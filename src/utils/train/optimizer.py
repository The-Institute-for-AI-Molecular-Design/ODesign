import torch


def get_optimizer(
    configs, model: torch.nn.Module
) -> torch.optim.Optimizer:

    param_groups = [{"params": model.parameters(), "lr": configs.lr}]

    if configs.type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=(configs.beta1, configs.beta2),
        )
    else:
        raise ValueError(f"Invalid optimizer: [{configs.type}]")

    return optimizer