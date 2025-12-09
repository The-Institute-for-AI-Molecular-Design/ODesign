import os
import sys
from pathlib import Path
import logging
import datetime
project_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(project_dir))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import wandb
from ml_collections.config_dict import ConfigDict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.data.dataloader import get_dataloaders
from src.model.odesign import ODesign
from src.model.modules.loss import ODesignLoss
from src.utils.model.misc import count_parameters
from src.utils.train.distributed import DIST_WRAPPER
from src.utils.train.optimizer import get_optimizer
from src.utils.train.lr_scheduler import get_lr_scheduler
from src.utils.train.metrics import SimpleMetricAggregator
from src.utils.train.train_runner import TrainRunner
from src.utils.permutation.permutation import SymmetricPermutation
from src.utils.misc import print_configs, seed_everything

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path=str(project_dir/"configs"), config_name="train"
)
def main(configs : DictConfig) -> None:
    # Load configs
    print_configs(configs)
    os.environ["DATA_ROOT_DIR"] = configs.data_root_dir
    configs = ConfigDict(
        OmegaConf.to_container(configs.exp, resolve=True)
    )
    output_dir = HydraConfig.get().runtime.output_dir
    ckpt_dir = Path(output_dir) / "checkpoints"
    error_dir = Path(output_dir) / "errors"
    eval_dump_dir = Path(output_dir) / "eval_dump"
    if DIST_WRAPPER.rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        if configs.eval_dump:
            eval_dump_dir.mkdir(parents=True, exist_ok=True)

    # Set up distributed env
    logger.info(
        f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
        + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
    )    
    device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
    torch.cuda.set_device(device)
    if DIST_WRAPPER.world_size > 1:
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    seed_everything(
        seed=configs.seed + DIST_WRAPPER.rank,
        deterministic=configs.deterministic,
    )

    # Init wandb
    if configs.use_wandb and DIST_WRAPPER.rank == 0:
        wandb.init(
            project=configs.project,
            name=configs.exp_name,
            config=vars(configs),
            id=configs.wandb_id or None,
            mode=configs.wandb_mode,
        )

    # Init dataloader
    train_dl, test_dls = get_dataloaders(
        configs,
        DIST_WRAPPER.world_size,
        seed=configs.seed,
        error_dir=error_dir,
    )    

    # Init model
    model = ODesign(configs).to(device)
    if DIST_WRAPPER.world_size > 1:
        model = DDP(
            model,
            find_unused_parameters=configs.model.find_unused_parameters,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )        
    if DIST_WRAPPER.rank == 0:
        logger.info(f"Model Parameters: {count_parameters(model)}")
    torch.cuda.empty_cache()

    # Init optimizer and lr scheduler
    optimizer = get_optimizer(configs.optimizer, model)
    lr_scheduler = get_lr_scheduler(configs.lr_scheduler, optimizer)

    # Init loss
    loss = ODesignLoss(configs)

    # Set up permutation
    symmetric_permutation = SymmetricPermutation(
        configs, error_dir=error_dir
    )    

    # Init metric wrapper
    train_metric_wrapper = SimpleMetricAggregator(["avg"])

    # Init runner
    train_runner = TrainRunner(
        configs=configs,
        ckpt_dir=ckpt_dir,
        error_dir=error_dir,
        eval_dump_dir=eval_dump_dir,
        device=device,
        train_dl=train_dl,
        test_dls=test_dls,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss=loss,
        symmetric_permutation=symmetric_permutation,
        train_metric_wrapper=train_metric_wrapper,
    )

    train_runner.run()


if __name__ == "__main__":
    main()