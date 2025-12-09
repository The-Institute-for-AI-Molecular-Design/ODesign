import os
from pathlib import Path
import logging

import wandb
from ml_collections.config_dict import ConfigDict
from contextlib import nullcontext
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from src.api.model_interface import LossInput, ODesignOutput, GroundTruth
from src.model.odesign import ODesign
from src.model.modules.loss import ODesignLoss
from src.utils.inference.dumper import DataDumper
from src.utils.model.misc import is_loss_nan_check
from src.utils.train.distributed import DIST_WRAPPER
from src.utils.train.lr_scheduler import get_lr_scheduler
from src.utils.train.metrics import SimpleMetricAggregator
from src.utils.model.torch_utils import (
    autocasting_disable_decorator,
    to_device,
    filter_state_dict,
)
from src.utils.permutation.permutation import SymmetricPermutation

logger = logging.getLogger(__name__)


class TrainRunner(object):
    def __init__(
        self,
        configs: ConfigDict,
        ckpt_dir: str | Path,
        error_dir: str | Path,
        eval_dump_dir: str | Path,
        device: torch.device,
        train_dl: DataLoader,
        test_dls: dict[str, DataLoader],
        model: ODesign,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss: ODesignLoss,
        symmetric_permutation: SymmetricPermutation,
        train_metric_wrapper: SimpleMetricAggregator,
    ) -> None:
        
        self.configs = configs
        self.ckpt_dir = ckpt_dir
        self.error_dir = error_dir
        self.eval_dump_dir = eval_dump_dir
        self.device = device
        self.train_dl = train_dl
        self.test_dls = test_dls
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.symmetric_permutation = symmetric_permutation
        self.train_metric_wrapper = train_metric_wrapper

        # Step means effective step considering accumulation
        self.step = 0
        # Global_step equals to self.step * self.iters_to_accumulate
        self.global_step = 0
        self.start_step = 0
        # Add for grad accumulation, it can increase real batch size
        self.iters_to_accumulate = self.configs.iters_to_accumulate    

        self.load_checkpoint()

    def load_checkpoint(self):

        def _load_checkpoint(
            checkpoint_path: str,
            load_params_only: bool,
            skip_load_optimizer: bool = False,
            skip_load_step: bool = False,
            skip_load_scheduler: bool = False,
            load_step_for_scheduler: bool = True,
        ):
            if not os.path.exists(checkpoint_path):
                raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
            self.print(
                f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
            )
            checkpoint = torch.load(checkpoint_path, self.device)

            sample_key = [k for k in checkpoint["model"].keys()][0]
            self.print(f"Sampled key: {sample_key}")
            if sample_key.startswith("module.") and not (DIST_WRAPPER.world_size > 1):
                # DDP checkpoint has module. prefix
                checkpoint["model"] = {
                    k[len("module.") :]: v for k, v in checkpoint["model"].items()
                }

            # Handle shape mismatches by filtering out parameters with different shapes
            filtered_state_dict = filter_state_dict(
                model_state_dict=self.model.state_dict(),
                ckpt_state_dict=checkpoint["model"],
            )
            
            self.model.load_state_dict(
                state_dict=filtered_state_dict,
                strict=self.configs.load_strict,
            )

            if not load_params_only:
                if not skip_load_optimizer:
                    self.print(f"Loading optimizer state")
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                if not skip_load_step:
                    self.print(f"Loading checkpoint step")
                    self.step = checkpoint["step"] + 1
                    self.start_step = self.step
                    self.global_step = self.step * self.iters_to_accumulate
                if not skip_load_scheduler:
                    self.print(f"Loading scheduler state")
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                elif load_step_for_scheduler:
                    assert (
                        not skip_load_step
                    ), "if load_step_for_scheduler is True, you must load step first"
                    # reinitialize LR scheduler using the updated optimizer and step
                    self.lr_scheduler = get_lr_scheduler(
                        self.configs.lr_scheduler,
                        self.optimizer,
                        last_epoch=self.step - 1,
                    )

            self.print(f"Finish loading checkpoint, current step: {self.step}")

        # Load model
        if self.configs.load_checkpoint_path:
            _load_checkpoint(
                self.configs.load_checkpoint_path,
                self.configs.load_params_only,
                skip_load_optimizer=self.configs.skip_load_optimizer,
                skip_load_scheduler=self.configs.skip_load_scheduler,
                skip_load_step=self.configs.skip_load_step,
                load_step_for_scheduler=self.configs.load_step_for_scheduler,
            )
    
    def save_checkpoint(self):
        if DIST_WRAPPER.rank == 0:
            path = f"{self.ckpt_dir}/{self.step}.pt"
            checkpoint = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.lr_scheduler.state_dict()
                    if self.lr_scheduler is not None
                    else None
                ),
                "step": self.step,
            }
            torch.save(checkpoint, path)
            self.print(f"Saved checkpoint to {path}")

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def model_forward(
        self,
        batch: dict,
        mode: str = "train",
    ) -> tuple[ODesignOutput, GroundTruth, LossInput]:
        assert mode in ["train", "eval"]
        pred_output, ground_truth, loss_input = self.model(
            feature_data=batch["feature_data"],
            label_data=batch["label_data"],
            label_full_data=batch["label_full_data"],
            mode=mode,
            current_step=self.step if mode == "train" else None,
            symmetric_permutation=self.symmetric_permutation,
        )
        return pred_output, ground_truth, loss_input

    def get_loss(
        self,
        loss_input: LossInput,
        pred_output: ODesignOutput,
        ground_truth: GroundTruth,
        mode: str = "train",
    ) -> tuple[torch.Tensor, dict]:
        assert mode in ["train", "eval"]

        loss, loss_dict = autocasting_disable_decorator(self.configs.model.skip_amp.loss)(
            self.loss
        )(
            loss_input=loss_input,
            pred_output=pred_output,
            ground_truth=ground_truth,
            mode=mode,
        )
        return loss, loss_dict

    @torch.no_grad()
    def evaluate(self):
        # Init Metric Aggregator
        simple_metric_wrapper = SimpleMetricAggregator(["avg"])
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.model.dtype]
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.model.eval()

        for test_name, test_dl in self.test_dls.items():
            self.print(f"Testing on {test_name}")
            evaluated_pids = []
            total_batch_num = len(test_dl)
            for index, batch in enumerate(tqdm(test_dl)):
                batch = to_device(batch, self.device)
                pid = batch["basic"]["pdb_id"]

                if index + 1 == total_batch_num and DIST_WRAPPER.world_size > 1:
                    # Gather all pids across ranks for avoiding duplicated evaluations when drop_last = False
                    all_data_ids = DIST_WRAPPER.all_gather_object(evaluated_pids)
                    dedup_ids = set(sum(all_data_ids, []))
                    if pid in dedup_ids:
                        print(
                            f"Rank {DIST_WRAPPER.rank}: Drop data_id {pid} as it is already evaluated."
                        )
                        break
                evaluated_pids.append(pid)

                simple_metrics = {}
                with enable_amp:
                    # Model forward
                    pred_output, ground_truth, loss_input = self.model_forward(batch, mode="eval")
                    # Loss forward
                    _, loss_dict = self.get_loss(loss_input, pred_output, ground_truth, mode="eval")
                    simple_metrics.update(loss_dict)

                    if self.configs.eval_dump:
                        dumper = DataDumper(base_dir=self.eval_dump_dir)
                        dumper.dump(
                            dataset_name="",
                            pdb_id=pid,
                            seed=self.configs.seed,
                            pred_dict=pred_output.to_dict(),
                            atom_array=batch["atom_array"],
                            entity_poly_type=batch["basic"]["entity_poly_type"],
                        )                    

                # Metrics
                for key, value in simple_metrics.items():
                    simple_metric_wrapper.add(
                        key, value, namespace=test_name
                    )

                del batch, simple_metrics
                if index % 5 == 0:
                    # Release some memory periodically
                    torch.cuda.empty_cache()

            metrics = simple_metric_wrapper.calc()
            self.print(f"Step {self.step}, eval {test_name}: {metrics}")
            if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                wandb.log(metrics, step=self.step)

    def update(self):
        # Clip the gradient
        if self.configs.grad_clip_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.grad_clip_norm
            )

    def train_step(self, batch: dict):
        self.model.train()
        # FP16 training has not been verified yet
        train_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.model.dtype]
        enable_amp = (
            torch.autocast(
                device_type="cuda", dtype=train_precision, cache_enabled=False
            )
            if torch.cuda.is_available()
            else nullcontext()
        )

        scaler = torch.GradScaler(
            device="cuda" if torch.cuda.is_available() else "cpu",
            enabled=(self.configs.model.dtype == "float16"),
        )

        with enable_amp:
            pred_output, ground_truth, loss_input = self.model_forward(batch, mode="train")
            loss, loss_dict = self.get_loss(loss_input, pred_output, ground_truth, mode="train")

        if self.configs.model.dtype in ["bf16", "fp32"]:
            if is_loss_nan_check(loss):
                self.print(f"Skip iteration with NaN loss: {self.step} steps")
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        scaler.scale(loss / self.iters_to_accumulate).backward()

        # For simplicity, the global training step is used
        if (self.global_step + 1) % self.iters_to_accumulate == 0:
            self.print(
                f"self.step {self.step}, self.iters_to_accumulate: {self.iters_to_accumulate}"
            )
            # Unscales the gradients of optimizer's assigned parameters in-place
            scaler.unscale_(self.optimizer)
            # Do grad clip only
            self.update()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
        for key, value in loss_dict.items():
            if "loss" not in key:
                continue
            self.train_metric_wrapper.add(key, value, namespace="train")
        torch.cuda.empty_cache()

    def progress_bar(self, desc: str = ""):
        if DIST_WRAPPER.rank != 0:
            return
        if self.global_step % (
            self.configs.eval_interval * self.iters_to_accumulate
        ) == 0 or (not hasattr(self, "_ipbar")):
            # Start a new progress bar
            self._pbar = tqdm(
                range(
                    self.global_step
                    % (self.iters_to_accumulate * self.configs.eval_interval),
                    self.iters_to_accumulate * self.configs.eval_interval,
                )
            )
            self._ipbar = iter(self._pbar)

        step = next(self._ipbar)
        self._pbar.set_description(
            f"[step {self.step}: {step}/{self.iters_to_accumulate * self.configs.eval_interval}] {desc}"
        )
        return

    def run(self):
        """
        Main entry for the TrainRunner.

        This function handles the training process, evaluation, logging, and checkpoint saving.
        """
        if self.configs.eval_only or self.configs.eval_first or self.configs.eval_dump:
            self.evaluate()
            if self.configs.eval_only or self.configs.eval_dump:
                return

        while True:
            for batch in self.train_dl:
                is_update_step = (self.global_step + 1) % self.iters_to_accumulate == 0
                is_last_step = (self.step + 1) == self.configs.max_steps
                step_need_log = (self.step + 1) % self.configs.log_interval == 0

                step_need_eval = (
                    self.configs.eval_interval > 0
                    and (self.step + 1) % self.configs.eval_interval == 0
                )
                step_need_save = (
                    self.configs.checkpoint_interval > 0
                    and (self.step + 1) % self.configs.checkpoint_interval == 0
                )

                is_last_step &= is_update_step
                step_need_log &= is_update_step
                step_need_eval &= is_update_step
                step_need_save &= is_update_step

                batch = to_device(batch, self.device)
                self.progress_bar()
                self.train_step(batch)
                if step_need_log or is_last_step:
                    metrics = self.train_metric_wrapper.calc()
                    self.print(f"Step {self.step} train: {metrics}")
                    last_lr = self.lr_scheduler.get_last_lr()
                    if DIST_WRAPPER.rank == 0:
                        if self.configs.use_wandb:
                            lr_dict = {"train/lr": last_lr[0]}
                            for group_i, group_lr in enumerate(last_lr):
                                lr_dict[f"train/group{group_i}_lr"] = group_lr
                            wandb.log(lr_dict, step=self.step)
                        self.print(f"Step {self.step}, lr: {last_lr}")
                    if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
                        wandb.log(metrics, step=self.step)

                if step_need_save or is_last_step:
                    self.save_checkpoint()

                if step_need_eval or is_last_step:
                    self.evaluate()
                self.global_step += 1
                if self.global_step % self.iters_to_accumulate == 0:
                    self.step += 1
                if self.step >= self.configs.max_steps:
                    self.print(f"Finish training after {self.step} steps")
                    break
            if self.step >= self.configs.max_steps:
                break