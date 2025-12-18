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

"""
PyTorch utility functions.

This module provides various utility functions for PyTorch operations including:
- Gradient utilities (norm computation)
- Device management (moving tensors/dicts to device)
- Distance computations (cdist wrapper)
- Masking and averaging operations
- Weight initialization
- Tensor manipulation utilities
- Type conversion helpers
"""

from contextlib import nullcontext
from typing import Sequence, Union
import attr
import logging

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)


def grad_norm(params) -> float:
    """
    Compute the L2 norm of gradients across all parameters.
    
    Args:
        params: Iterable of parameters (e.g., model.parameters()).

    Returns:
        float: L2 norm of all parameter gradients. Returns 0.0 if no gradients exist.
    """
    total_norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def to_device(obj, device):
    """
    Recursively move tensor(s) to specified device.
    
    This function handles various data structures including:
    - Individual tensors
    - Dictionaries of tensors
    - Lists of tensors
    - Attr-decorated classes containing tensors
    
    Args:
        obj: Object to move. Can be torch.Tensor, dict, list, or attr class.
        device: Target device (e.g., 'cuda', 'cpu', torch.device object).

    Returns:
        Object with all tensors moved to the specified device.
        For attr classes, returns a new instance with updated tensor fields.
    """    
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = to_device(v, device)
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif attr.has(obj.__class__):
        # Handle attr classes (like OFeatureData, OLabelData)
        updates = {}
        for field in attr.fields(obj.__class__):
            value = getattr(obj, field.name)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    updates[field.name] = value.to(device)
                elif isinstance(value, dict):
                    updates[field.name] = to_device(value, device)
                elif isinstance(value, list):
                    updates[field.name] = to_device(value, device)
                elif attr.has(value.__class__):
                    updates[field.name] = to_device(value, device)
        if updates:
            return attr.evolve(obj, **updates)
        return obj
    elif isinstance(obj, list):
        return [to_device(item, device) for item in obj]
    else:
        return obj


def detach_if(t: torch.Tensor, detach: bool) -> torch.Tensor:
    """
    Conditionally detach a tensor from the computation graph.
    
    Args:
        t (torch.Tensor): Input tensor.
        detach (bool): Whether to detach the tensor.

    Returns:
        torch.Tensor: Detached tensor if detach=True, otherwise original tensor.
    """
    if detach:
        return t.detach()
    return t


def cdist(a: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between points.
    
    This is a wrapper around torch.cdist that forces the use of the more
    accurate (but potentially slower) computation mode that avoids using
    matrix multiplication for Euclidean distance.
    
    Args:
        a (torch.Tensor): First set of points.
            Shape: [..., N, D]
        b (torch.Tensor, optional): Second set of points. If None, computes
            pairwise distances within a. Defaults to None.
            Shape: [..., M, D]

    Returns:
        torch.Tensor: Pairwise distances.
            Shape: [..., N, M] if b is provided, otherwise [..., N, N]
    """
    # for tensor shape [1, 512 * 14, 3], donot_use_mm_for_euclid_dist mode costs 0.0489s,
    # while use_mm_for_euclid_dist_if_necessary costs 0.0419s on cpu. On GPU there two costs
    # will be neglectible. So there is no need to sacrifice accuracy for speed here.
    return torch.cdist(
        a,
        b if b is not None else a,
        compute_mode="donot_use_mm_for_euclid_dist",
    )


def batch_avg_with_mask(
    value: torch.Tensor,
    mask: torch.Tensor,
    avg_dim: Union[int, tuple[int]] = None,
    batch_reduction: str = "mean",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute masked average over specified dimensions with optional batch reduction.
    
    This function computes averages while respecting a binary mask, where 1 indicates
    valid values and 0 indicates masked (ignored) values.

    Args:
        value (torch.Tensor): Input values to average.
            Shape: [batch_size, ...]
        mask (torch.Tensor): Binary mask with same shape as value.
            1 for valid values, 0 for masked values.
            Shape: [batch_size, ...]
        avg_dim (Union[int, tuple[int]], optional): Dimensions over which to average.
            If None, averages over all dims except batch. Defaults to None.
        batch_reduction (str, optional): How to reduce the batch dimension.
            Options: 'mean', 'sum', 'none'. Defaults to "mean".
        eps (float, optional): Small constant to avoid division by zero. 
            Defaults to 1e-12.

    Returns:
        torch.Tensor: Masked averages.
            Shape depends on batch_reduction:
            - 'mean' or 'sum': scalar
            - 'none': [batch_size]
            
    Raises:
        Exception: If batch_reduction is not one of 'mean', 'sum', or 'none'.
    """
    if avg_dim is None:
        avg_dim = tuple(range(1, len(value.shape)))
    avg = (value * mask).sum(dim=avg_dim) / (mask.sum(dim=avg_dim) + eps)
    if batch_reduction == "mean":
        return avg.mean()
    elif batch_reduction == "sum":
        return avg.sum()
    elif batch_reduction == "none":
        return avg
    else:
        raise Exception(f"Invalid batch_reduction: {batch_reduction}")


def eye_mask(L: int, device=None, opposite: bool = False) -> torch.Tensor:
    """
    Create an identity matrix or its complement.
    
    Args:
        L (int): Size of the square matrix.
        device: Device on which to create the tensor. Defaults to None.
        opposite (bool, optional): If True, returns complement (1 - eye).
            Defaults to False.

    Returns:
        torch.Tensor: Identity matrix or its complement.
            Shape: [L, L]
    """
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    else:
        return torch.eye(L, device=device)


def glorot_uniform(t: torch.Tensor) -> None:
    """
    Initialize tensor with Glorot (Xavier) uniform initialization.
    
    Initializes weights using a uniform distribution with range computed
    based on the fan-in and fan-out of the tensor. This helps maintain
    gradient variance across layers.
    
    Args:
        t (torch.Tensor): Tensor to initialize. Modified in-place.
            Supports 2D (Linear), 3D (Conv1D), or higher dimensional tensors.
            
    Note:
        Formula: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    """
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m, bias: str = "zero") -> None:
    """
    Initialize a single parameter or module with Glorot uniform weights.
    
    Args:
        m: Parameter or module to initialize.
        bias (str, optional): Bias initialization method. 
            Options: 'zero', 'normal'. Defaults to "zero".
    """
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            if bias == "zero":
                m.bias.data.zero_()
            else:
                assert bias == "normal"
                m.bias.data.normal_()
        glorot_uniform(m.weight.data)


def weights_init(m: nn.Module, bias: str = "zero") -> None:
    """
    Recursively initialize all parameters in a module with Glorot uniform.
    
    Traverses all submodules and top-level parameters, applying
    Glorot uniform initialization to weights and specified initialization
    to biases.
    
    Args:
        m (nn.Module): Module to initialize.
        bias (str, optional): Bias initialization method.
            Options: 'zero', 'normal'. Defaults to "zero".
    """
    for p in m.modules():
        if isinstance(p, nn.ParameterList) or isinstance(p, nn.ModuleList):
            for pp in p:
                _param_init(pp, bias)
        else:
            _param_init(p, bias)

    for name, p in m.named_parameters():
        if not "." in name:  # top-level parameters
            _param_init(p, bias)


def permute_last_dims(t: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """
    Permute the last N dimensions of a tensor while keeping other dimensions unchanged.
    
    This is useful for reordering the innermost dimensions without affecting
    batch or outer dimensions.

    Args:
        t (torch.Tensor): Input tensor with at least len(dims) dimensions.
        dims (Sequence[int]): Desired ordering of last dimensions.
            All values should be negative, e.g., (-1, -2) to swap last two dims.

    Returns:
        torch.Tensor: Tensor with last dimensions permuted according to dims.
        
    Example:
        >>> t = torch.randn(2, 3, 4, 5)
        >>> permute_last_dims(t, (-1, -2))  # Swap last two dims
        torch.Size([2, 3, 5, 4])
    """
    num_dims = len(t.shape)
    prefix_dims = list(range(num_dims - len(dims)))
    last_dims = [num_dims + d for d in dims]
    return torch.permute(t, prefix_dims + last_dims)


def flatten_tensors(tensors) -> torch.Tensor:
    """
    Flatten a list of tensors into a single 1D tensor.
    
    Args:
        tensors: Iterable of tensors to flatten and concatenate.

    Returns:
        torch.Tensor: Single 1D tensor containing all values from input tensors.
    """
    return torch.cat([t.view(-1) for t in tensors], dim=0)


def unflatten_tensors(flat_tensor: torch.Tensor, shapes: list) -> list[torch.Tensor]:
    """
    Unflatten a 1D tensor into a list of tensors with specified shapes.
    
    This is the inverse operation of flatten_tensors.
    
    Args:
        flat_tensor (torch.Tensor): 1D tensor to unflatten.
        shapes (list): List of shapes for each output tensor.

    Returns:
        list[torch.Tensor]: List of tensors reshaped according to shapes.
    """
    tensors = []
    offset = 0
    for shape in shapes:
        numel = shape.numel()
        tensors.append(flat_tensor[offset : offset + numel].view(shape))
        offset += numel
    return tensors


def map_values_to_list(data: dict, recursive: bool = True) -> dict:
    """
    Convert tensor and array values in a dictionary to Python lists.
    
    Useful for JSON serialization or logging. Handles bfloat16 tensors
    by converting to float32 first.
    
    Args:
        data (dict): Dictionary with tensor/array values.
        recursive (bool, optional): Whether to recursively process nested dicts.
            Defaults to True.

    Returns:
        dict: Dictionary with all tensors/arrays converted to lists.
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, dict) and recursive:
            data[k] = map_values_to_list(v, recursive)
    return data


def round_values(data: dict, recursive: bool = True) -> dict:
    """
    Round numeric values in a dictionary to 2 decimal places.
    
    Handles tensors, numpy arrays, and lists. Converts bfloat16 tensors
    to float32 before rounding.
    
    Args:
        data (dict): Dictionary with numeric values.
        recursive (bool, optional): Whether to recursively process nested dicts.
            Defaults to True.

    Returns:
        dict: Dictionary with all numeric values rounded to 2 decimals.
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, np.ndarray):
            data[k] = np.round(v, 2)
        elif isinstance(v, list):
            data[k] = list(np.round(np.array(v), 2))
        elif isinstance(v, dict) and recursive:
            data[k] = round_values(v, recursive)
    return data


def autocasting_disable_decorator(disable_casting: bool):
    """
    Decorator to conditionally disable automatic mixed precision (AMP) casting.
    
    When disable_casting=True, this decorator:
    1. Disables CUDA autocast context
    2. Converts all float tensor arguments to float32
    3. Recursively handles dicts, lists, and attr-decorated classes
    
    Useful for operations requiring full precision for numerical stability.
    
    Args:
        disable_casting (bool): Whether to disable AMP and cast to float32.

    Returns:
        Function decorator that wraps the target function.
        
    Example:
        >>> @autocasting_disable_decorator(disable_casting=True)
        >>> def my_func(x: torch.Tensor) -> torch.Tensor:
        >>>     return x * 2  # Will compute in float32 even if AMP is enabled
    """
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if disable_casting
                else nullcontext()
            )

            # Helper function to recursively cast tensors in nested structures
            def conditioned_cast(obj):
                if not disable_casting:
                    return obj
                    
                if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
                    return obj.to(dtype=torch.float32)
                elif attr is not None and hasattr(obj, '__dict__') and hasattr(obj.__class__, '__attrs_attrs__'):
                    # Handle @define decorated classes (attrs classes)
                    try:
                        field_dict = {}
                        for field in attr.fields(obj.__class__):
                            field_value = getattr(obj, field.name)
                            field_dict[field.name] = conditioned_cast(field_value)
                        return obj.__class__(**field_dict)
                    except Exception:
                        # Fallback: if attrs reconstruction fails, return original object
                        return obj
                elif isinstance(obj, dict):
                    # Handle dictionaries
                    return {k: conditioned_cast(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    # Handle lists and tuples
                    casted_items = [conditioned_cast(item) for item in obj]
                    return type(obj)(casted_items)
                else:
                    return obj

            with _amp_context:
                return func(
                    *(conditioned_cast(v) for v in args),
                    **{k: conditioned_cast(v) for k, v in kwargs.items()},
                )

        return new_func

    return func_wrapper


def dict_to_tensor(feature_dict: dict) -> dict:
    """
    Convert numpy arrays in a dictionary to PyTorch tensors.
    
    Preserves dtypes by mapping numpy int/float types to appropriate
    torch types.
    
    Args:
        feature_dict (dict): Dictionary containing numpy arrays or tensors.

    Returns:
        dict: Dictionary with all arrays converted to tensors.
    """
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = feature_dict[k].dtype
            feature_dict[k] = torch.tensor(v)

            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)

    return feature_dict


def collate_fn_identity(x):
    """
    Identity collate function that returns input unchanged.
    
    Useful as a DataLoader collate_fn when no collation is needed.
    
    Args:
        x: Input data from DataLoader.

    Returns:
        Input x unchanged.
    """
    return x


def collate_fn_first(x):
    """
    Collate function that returns only the first element.
    
    Useful when batch size is 1 and you want to unwrap the batch dimension.
    
    Args:
        x: List/tuple from DataLoader (typically length 1).

    Returns:
        First element of x.
    """
    return x[0]


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """
    Append singleton dimensions to the end of a tensor.
    
    Useful for broadcasting operations where you need to match dimensionality
    without changing the actual data shape.
    
    Args:
        x (torch.Tensor): Input tensor.
        target_dims (int): Desired number of dimensions.

    Returns:
        torch.Tensor: Tensor with singleton dimensions appended.
        
    Raises:
        ValueError: If input already has more dims than target_dims.
        
    Example:
        >>> x = torch.randn(3, 4)  # 2D
        >>> y = append_dims(x, 4)  # Add 2 dims
        >>> y.shape
        torch.Size([3, 4, 1, 1])
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def cat_dict(dict_list: list[dict], key: str, dim: int = 0) -> torch.Tensor:
    """
    Concatenate a specific key's values from a list of dictionaries.
    
    Args:
        dict_list (list[dict]): List of dictionaries containing tensors.
        key (str): Key to extract from each dictionary.
        dim (int, optional): Dimension along which to concatenate. Defaults to 0.

    Returns:
        torch.Tensor: Concatenated tensor from all dictionaries.
    """
    return torch.cat([x[key] for x in dict_list], dim=dim)


def filter_state_dict(
    model_state_dict: dict,
    ckpt_state_dict: dict,
    verbose: bool = False,
) -> dict:
    """
    Filter checkpoint state dict to match model's expected parameters and shapes.
    
    Removes parameters that don't exist in the model or have mismatched shapes,
    allowing safe loading of partially compatible checkpoints.
    
    Args:
        model_state_dict (dict): State dict from the model (e.g., model.state_dict()).
        ckpt_state_dict (dict): State dict from checkpoint to be filtered.
        verbose (bool, optional): If True, logs warnings for skipped parameters.
            Defaults to False.

    Returns:
        dict: Filtered checkpoint state dict containing only compatible parameters.
        
    Note:
        Useful for transfer learning or loading checkpoints from models with
        slightly different architectures.
    """
    if not verbose:
        filtered_state_dict = {
            k: v
            for k, v in ckpt_state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
    else:
        filtered_state_dict = {}
        for k, v in ckpt_state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(
                        f"Skipping parameter {k} due to shape mismatch: "
                        f"checkpoint {v.shape} vs model {model_state_dict[k].shape}"
                    )
            else:
                logger.warning(f"Skipping unexpected parameter {k}")

    return filtered_state_dict