# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings

import torch
from torch.optim.lr_scheduler import LRScheduler


# The Alphafold3 Learning Rate Scheduler As in 5.4
class AlphaFold3LRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        warmup_steps: int = 1000,
        lr: float = 1.8e-3,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_every_n_steps
        self.lr = lr
        self.decay_factor = decay_factor
        super(AlphaFold3LRScheduler, self).__init__(
            optimizer=optimizer, last_epoch=last_epoch
        )

    def _get_step_lr(self, step):
        if step <= self.warmup_steps:
            lr = step / self.warmup_steps * self.lr
        else:
            decay_count = step // self.decay_steps
            lr = self.lr * (self.decay_factor**decay_count)
        return lr

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]


def get_lr_scheduler(
    configs, optimizer: torch.optim.Optimizer, **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the learning rate scheduler based on the configuration.

    Args:
        configs: Configuration object containing scheduler settings.
        optimizer (torch.optim.Optimizer): The optimizer to which the scheduler will be attached.
        **kwargs: Additional keyword arguments to be passed to the scheduler.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler.

    Raises:
        ValueError: If the specified learning rate scheduler is invalid.
    """
    if configs.type == "af3":
        lr_scheduler = AlphaFold3LRScheduler(
            optimizer,
            warmup_steps=configs.warmup_steps,
            decay_every_n_steps=configs.decay_every_n_steps,
            decay_factor=configs.decay_factor,
            lr=configs.lr,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid lr scheduler: [{configs.type}]")
    return lr_scheduler
