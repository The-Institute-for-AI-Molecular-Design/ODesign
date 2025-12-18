import os
import sys
from pathlib import Path
import logging
project_dir = Path(__file__).absolute().parent.parent
sys.path.append(str(project_dir))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)

import torch
import torch.distributed as dist
from ml_collections.config_dict import ConfigDict

from src.utils.inference.dumper import DataDumper
from src.utils.inference.infer_runner import InferRunner
from src.data.dataloader import get_inference_dataloader
from src.model.odesign import ODesign
from src.utils.train.distributed import DIST_WRAPPER
from src.utils.misc import print_configs

logger = logging.getLogger(__name__)

from biotite.structure import AtomArray
def fast_repr(self):
    # 只返回简单的摘要，不进行繁重的字符串拼接
    return f"<{self.__class__.__name__} shape={self.shape} (Monkey Patched)>"

AtomArray.__repr__ = fast_repr


@hydra.main(
    version_base="1.3", config_path=str(project_dir/"configs"), config_name="inference"
)
def main(configs : DictConfig) -> None:
    # Load configs
    print_configs(configs)
    os.environ["DATA_ROOT_DIR"] = configs.data_root_dir
    os.environ["CKPT_ROOT_DIR"] = configs.ckpt_root_dir
    configs = ConfigDict(
        OmegaConf.to_container(configs.exp, resolve=True)
    )
    dump_dir = HydraConfig.get().runtime.output_dir
    configs.dump_dir = dump_dir
    error_dir = Path(dump_dir) / "errors"
    if DIST_WRAPPER.rank == 0:
        error_dir.mkdir(parents=True, exist_ok=True)

    # Set up distributed env
    logger.info(
        f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
        + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
    )    
    device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
    torch.cuda.set_device(device)
    if DIST_WRAPPER.world_size > 1:
        dist.init_process_group(backend="nccl")

    # Init dataloader
    infer_dl = get_inference_dataloader(configs=configs)

    # Init model
    model = ODesign(configs).to(device)

    # Init dumper
    dumper = DataDumper(base_dir=dump_dir)

    # Init runner
    infer_runner = InferRunner(
        configs=configs,
        dump_dir=dump_dir,
        error_dir=error_dir,
        device=device,
        infer_dl=infer_dl,
        model=model,
        dumper=dumper,
    )

    infer_runner.run()
    

if __name__ == "__main__":
    main()
