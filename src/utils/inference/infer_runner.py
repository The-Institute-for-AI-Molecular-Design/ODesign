import os
from pathlib import Path
import logging
import traceback
from contextlib import nullcontext
from os.path import join as opjoin
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from ml_collections.config_dict import ConfigDict

from src.utils.inference.dumper import DataDumper
from src.model.odesign import ODesign
from src.model.modules.invfold.evaluation_tools.tools import *
from src.utils.train.distributed import DIST_WRAPPER
from src.utils.misc import seed_everything
from src.utils.model.torch_utils import to_device

logger = logging.getLogger(__name__)


class InferRunner(object):
    def __init__(
        self,
        configs: ConfigDict,
        dump_dir: str | Path,
        error_dir: str | Path,
        device: torch.device,
        infer_dl: DataLoader,
        model: ODesign,
        dumper: DataDumper,
    ) -> None:
        
        self.configs = configs
        self.dump_dir = dump_dir
        self.error_dir = error_dir
        self.device = device
        self.infer_dl = infer_dl
        self.model = model
        self.dumper = dumper

        self.load_checkpoint()
        self.load_invfold_module()

    def load_checkpoint(self) -> None:
        checkpoint_path = f"{os.getenv('CKPT_ROOT_DIR')}/{self.configs.infer_model_name}.pt"
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }

        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def load_invfold_module(self) -> None:
        checkpoint_path = f"{os.getenv('CKPT_ROOT_DIR')}/oinvfold_{self.configs.design_modality}.ckpt"
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(f"Loading inverse folding module from {checkpoint_path}")

        oinvfold, _ = reload_model(
            self.configs.design_modality,
            checkpoint_path=checkpoint_path,
            configs=self.configs.model.invfold_module,
            device=self.device,
        )

        self.model.invfold_module = oinvfold
        self.print(f"Finish loading inverse folding module.")

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_inference_configs(self, N_token: int):
        # Setting the default inference configs for different N_token and N_atom
        # when N_token is larger than 3000, the default config might OOM even on a
        # A100 80G GPUS,
        if N_token > 3840:
            self.configs.model.skip_amp.sample_diffusion = False
        else:
            self.configs.model.skip_amp.sample_diffusion = True
        
        self.model.configs = self.configs

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
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

        data = to_device(data, self.device)

        # Backbone Generation
        with enable_amp:
            pred_backbone_output, _, _ = self.model(
                feature_data=data["feature_data"],
                label_full_data=None,
                label_data=data["label_data"],
                mode="inference",
            )

        # Inverse Folding
        inv_samples = parse_invfold(
            atom_array=data["atom_array"],
            pred_output=pred_backbone_output,
            design_modality=self.configs.design_modality,
            sample_name=data["sample_name"],             
        )        
        all_sequence_variants = []  
        sequence_variants = None
        num_cand = None
        for inv_sample in inv_samples:
            smp = inv_sample[0]

            pred_seqs, scores, _, _, _ = inference(
                model=self.model.invfold_module,
                sample_input=smp,
                design_modality=self.configs.design_modality,
                topk=self.configs.invfold_topk,
                temp=self.configs.invfold_temp,
                use_beam=self.configs.invfold_use_beam,
                device=self.device,
            )

            if not pred_seqs:
                all_sequence_variants.append([])
                continue

            res_ids = smp["res_ids"]           # (L_design,)
            chain_ids = smp["chain_ids"]       # (L_design,)
            res_atom_indices = smp["res_atom_indices"] 
            ch = str(chain_ids[0])            

            num_cand = len(pred_seqs)
            sequence_variants = [
                {"per_chain": {}, "scores": {}} for _ in range(num_cand)
            ]

            for cand_idx in range(num_cand):
                seq_c = pred_seqs[cand_idx]    

                sequence_variants[cand_idx]["per_chain"][ch] = {
                    "res_ids": res_ids,
                    "atom_indices": res_atom_indices,
                    "new_seq": seq_c,
                }
                sequence_variants[cand_idx]["scores"][ch] = scores[cand_idx]

            all_sequence_variants.append(sequence_variants)

        return pred_backbone_output, all_sequence_variants
    
    def run(self) -> None:
        num_data = len(self.infer_dl.dataset)
        for seed in self.configs.seeds:
            seed_everything(seed=seed, deterministic=self.configs.deterministic)
            for batch in self.infer_dl:
                try:
                    data, data_error_message = batch[0]
                    atom_array = data.get("atom_array", None)
                    sample_name = data["sample_name"]

                    if len(data_error_message) > 0:
                        logger.info(data_error_message)
                        with open(opjoin(self.error_dir, f"{sample_name}.txt"), "a") as f:
                            f.write(data_error_message)
                        continue

                    logger.info(
                        (
                            f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                            f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                            f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                        )
                    )
                    self.update_inference_configs(data["N_token"].item())

                    pred_backbone_output, all_sequence_variants = self.predict(data)

                    self.dumper.dump(
                        dataset_name="",
                        pdb_id=sample_name,
                        seed=seed,
                        pred_dict=pred_backbone_output.to_dict(),
                        all_sequence_variants=all_sequence_variants,
                        design_modality=self.configs.design_modality,
                        atom_array=atom_array,
                        entity_poly_type=data["entity_poly_type"],
                    )

                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {data['sample_name']} succeeded.\n"
                        f"Results saved to {self.dump_dir}"
                    )
                    torch.cuda.empty_cache()
                except Exception as e:
                    error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                    logger.info(error_message)
                    # Save error info
                    with open(opjoin(self.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(error_message)
                    if hasattr(torch.cuda, "empty_cache"):
                        torch.cuda.empty_cache()

                        