import os
import random
import rich
from rich.panel import Panel
from rich.syntax import Syntax
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from src.utils.train.distributed import DIST_WRAPPER


def print_configs(configs: DictConfig) -> None:
    if DIST_WRAPPER.rank == 0:
        panel = Panel(
            Syntax(
                OmegaConf.to_yaml(configs, resolve=True),
                "yaml",
                background_color="default",
                line_numbers=True,
                indent_guides=True,
            ),
            title="Configs",
            expand=False,
        )
        rich.print(panel)


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic=True applies to CUDA convolution operations, and nothing else.
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) affects all the normally-nondeterministic operations listed here https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html?highlight=use_deterministic#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"