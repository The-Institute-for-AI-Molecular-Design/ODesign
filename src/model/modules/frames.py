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

import torch
import torch.nn.functional as F

from src.utils.model.misc import batched_gather


def express_coordinates_in_frame(
    coordinate: torch.Tensor, frames: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Express coordinates in local reference frames.
    
    Implements Algorithm 29 from AlphaFold3 SI. This function transforms
    atom coordinates into a local frame defined by three atoms (a, b, c),
    where b is the origin, and an orthonormal basis is constructed from
    the vectors (a-b) and (c-b).

    Args:
        coordinate (torch.Tensor): Atom coordinates to be transformed.
            Shape: [..., N_atom, 3]
        frames (torch.Tensor): Frame definitions specified by three atoms.
            Shape: [..., N_frame, 3[a,b,c], 3[x,y,z]]
        eps (float, optional): Small epsilon value for numerical stability 
            during normalization. Defaults to 1e-8.

    Returns:
        torch.Tensor: Coordinates expressed in local frame basis vectors.
            Shape: [..., N_frame, N_atom, 3]
            
    Note:
        The frame is defined by:
        - Origin: atom b
        - e1: normalized bisector of (a-b) and (c-b)
        - e2: normalized difference of (c-b) and (a-b)
        - e3: cross product of e1 and e2
    """
    # Extract frame atoms
    a, b, c = torch.unbind(frames, dim=-2)  # a, b, c shape: [..., N_frame, 3]
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    # Build orthonormal basis
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)  # [..., N_frame, 3]
    # Project onto frame basis
    d = coordinate[..., None, :, :] - b[..., None, :]  #  [..., N_frame, N_atom, 3]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[..., None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[..., None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )  # [..., N_frame, N_atom, 3]
    return x_transformed


def gather_frame_atom_by_indices(
    coordinate: torch.Tensor, frame_atom_index: torch.Tensor, dim: int = -2
) -> torch.Tensor:
    """
    Construct frames by gathering specific atoms from coordinates.
    
    This function extracts three atoms per frame according to the provided indices,
    forming the frame definitions needed for expressCoordinatesInFrame. It supports
    both batched and non-batched indexing operations.

    Args:
        coordinate (torch.Tensor): Input atom coordinates.
            Shape: [..., N_atom, 3]
        frame_atom_index (torch.Tensor): Indices specifying which three atoms 
            define each frame. Can be either batched or non-batched.
            Shape: [..., N_frame, 3] or [N_frame, 3]
        dim (int, optional): Dimension along which to select atoms. 
            Defaults to -2 (the atom dimension).

    Returns:
        torch.Tensor: Frame definitions with three atoms per frame.
            Shape: [..., N_frame, 3[a,b,c atoms], 3[x,y,z coordinates]]
            
    Note:
        - If frame_atom_index is 2D, uses simple index_select for efficiency
        - If frame_atom_index is batched, uses batched_gather for correct broadcasting
    """
    if len(frame_atom_index.shape) == 2:
        # the navie case
        x1 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 0]
        )  # [..., N_frame, 3]
        x2 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 1]
        )  # [..., N_frame, 3]
        x3 = torch.index_select(
            coordinate, dim=dim, index=frame_atom_index[:, 2]
        )  # [..., N_frame, 3]
        return torch.stack([x1, x2, x3], dim=dim)
    else:
        assert (
            frame_atom_index.shape[:dim] == coordinate.shape[:dim]
        ), "batch size dims should match"

    x1 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 0],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x2 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 1],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    x3 = batched_gather(
        data=coordinate,
        inds=frame_atom_index[..., 2],
        dim=dim,
        no_batch_dims=len(coordinate.shape[:dim]),
    )  # [..., N_frame, 3]
    return torch.stack([x1, x2, x3], dim=dim)
