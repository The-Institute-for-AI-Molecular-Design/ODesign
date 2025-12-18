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

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.api.model_interface import PairFormerInput
from src.model.modules.primitives import LinearNoBias
from src.model.modules.transformer import AtomAttentionEncoder
from src.model.modules.feature_dims import (
    INPUTFEATS_DIMS,
    RELPOSENC_DIMS,
)


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )

        self.input_feature_dims = INPUTFEATS_DIMS

    def _check_input(self, input_data: PairFormerInput):
        for name in self.input_feature_dims:
            if name not in input_data:
                raise ValueError(f"Input data must contain {name}.")
            elif input_data[name] is None:
                raise ValueError(f"Input data must contain {name} and it cannot be None.")

    def forward(
        self,
        input_data: PairFormerInput,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_data (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
        """
        self._check_input(input_data)
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_data=input_data,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_data.restype.shape[:-1]
        
        s_inputs = torch.cat(
            [a]
            + [
                input_data[name].reshape(*batch_shape, d)
                for name, d in self.input_feature_dims.items()
            ],
            dim=-1,
        )

        return s_inputs


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        
        self.input_feature_dims = RELPOSENC_DIMS
    
    def _check_input(self, input_data: PairFormerInput):
        for name in self.input_feature_dims:
            if name not in input_data:
                raise ValueError(f"Input data must contain {name}.")
            elif input_data[name] is None:
                raise ValueError(f"Input data must contain {name} and it cannot be None.")

    def forward(self, input_data: PairFormerInput) -> torch.Tensor:
        """
        Args:
            input_data (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """

        self._check_input(input_data)
        b_same_chain = (
            input_data.asym_id[..., :, None]
            == input_data.asym_id[..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            input_data.residue_index[..., :, None]
            == input_data.residue_index[..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            input_data.entity_id[..., :, None]
            == input_data.entity_id[..., None, :]
        ).long()  # [..., N_token, N_token]
        
        # cycpeptide
        offset = (
            input_data.residue_index[..., :, None]
            - input_data.residue_index[..., None, :]
        )

        asym_id = input_data.asym_id
        unique_asym_ids = torch.unique(asym_id)
        is_cyclic_token = input_data.is_cyclic_token
        if is_cyclic_token is not None and is_cyclic_token.any():            
            for asym_id_val in unique_asym_ids:
                asym_mask = (asym_id == asym_id_val)
                
                # Check if this asym_id corresponds to a cyclic entity
                if asym_mask.any() and is_cyclic_token[asym_mask][0]:
                    # Apply cyclic logic for this asym_id
                    len_chain = asym_mask.sum()
                    if len_chain > 0:
                        chain_mask = (
                            (asym_id[..., :, None] == asym_id_val) &
                            (asym_id[..., None, :] == asym_id_val)
                        )
                        
                        offset_plus = offset + len_chain
                        offset_minus = offset - len_chain
                        abs_offset = offset.abs()
                        abs_offset_plus = offset_plus.abs()
                        abs_offset_minus = offset_minus.abs()

                        choice_plus_or_minus = torch.where(
                            abs_offset_plus <= abs_offset_minus,
                            offset_plus,
                            offset_minus
                        )
                        c_offset = torch.where(
                            (abs_offset <= abs_offset_plus) & (abs_offset <= abs_offset_minus),
                            offset,
                            choice_plus_or_minus
                        )
                        offset = torch.where(chain_mask, c_offset, offset)
                                
        offset = (offset + self.r_max).clamp(0, 2 * self.r_max)
        d_residue = torch.where(
            b_same_chain.bool(),
            offset,
            (2 * self.r_max + 1) * torch.ones_like(offset)
        )
        
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        d_token = torch.clip(
            input=input_data.token_index[..., :, None]
            - input_data.token_index[..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = torch.clip(
            input=input_data.sym_id[..., :, None]
            - input_data.sym_id[..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        if self.training:
            p = self.linear_no_bias(
                torch.cat(
                    [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                    dim=-1,
                ).float()
            )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
            return p
        else:
            del d_chain, d_token, d_residue, b_same_chain, b_same_residue
            origin_shape = a_rel_pos.shape[:-1]
            Ntoken = a_rel_pos.shape[-2]
            a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
            chunk_num = 1 if Ntoken < 3200 else 8
            a_rel_pos_chunks = torch.chunk(
                a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
            )
            a_rel_token_chunks = torch.chunk(
                a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
            )
            b_same_entity_chunks = torch.chunk(
                b_same_entity.reshape(-1, 1), chunk_num, dim=-2
            )
            a_rel_chain_chunks = torch.chunk(
                a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
            )
            start = 0
            p = None
            for i in range(len(a_rel_pos_chunks)):
                data = torch.cat(
                    [
                        a_rel_pos_chunks[i],
                        a_rel_token_chunks[i],
                        b_same_entity_chunks[i],
                        a_rel_chain_chunks[i],
                    ],
                    dim=-1,
                ).float()
                result = self.linear_no_bias(data)
                del data
                if p is None:
                    p = torch.empty(
                        (a_rel_pos.shape[-2], self.c_z),
                        device=a_rel_pos.device,
                        dtype=result.dtype,
                    )
                p[start : start + result.shape[0]] = result
                start += result.shape[0]
                del result
            del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
            p = p.reshape(*origin_shape, -1)
            return p

class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )


class SubstructureEmbedder(nn.Module):
    """
    Implements Substructure Embedder
    """

    def __init__(
        self,
        n_classes: int,
        c_pair_dim: int,
        architecture: str = "mlp",
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            n_classes (int): Number of distance classes in input
            c_pair_dim (int): Output pair embedding dimension
            architecture (str): Either 'mlp' or 'transformer'
            hidden_dim (int): Hidden dimension for both architectures
            n_layers (int): Number of layers (MLP or Transformer)
            dropout (float): Dropout rate
        """
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture == "mlp":
            layers = []
            layers.append(LinearNoBias(n_classes, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(LinearNoBias(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(LinearNoBias(hidden_dim, c_pair_dim))

            self.network = nn.Sequential(*layers)

        elif self.architecture == "transformer":
            self.input_proj = LinearNoBias(n_classes, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = LinearNoBias(hidden_dim, c_pair_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input distance map
                shape: [..., N_token, N_token, N_classes]
        Returns:
            torch.Tensor: Output pair embeddings
                shape: [..., N_token, N_token, c_pair_dim]
        """
        if self.architecture == "mlp":
            return self.network(x)
        else:  # transformer
            # Reshape for transformer
            orig_shape = x.shape
            x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
            batch_size, n_token, _, n_classes = x.shape

            # Project and reshape to sequence
            x = self.input_proj(x)
            x = x.reshape(batch_size, n_token * n_token, -1)

            # Apply transformer
            x = self.transformer(x)

            # Project to output dim and reshape back
            x = self.output_proj(x)
            x = x.reshape(*orig_shape[:-1], -1)

            return x


class ConstraintEmbedder(nn.Module):
    """
    Implements Constraint Embedder
    """

    def __init__(
        self,
        pocket_embedder: dict[str:int],
        contact_embedder: dict[str:int],
        contact_atom_embedder: dict[str:int],
        substructure_embedder: dict[str:int],
        c_constraint_z: int,
        initialize_method: str = "zero",
        **kwarg,
    ) -> None:
        """

        Args:
            pocket_embedder (dict[str:int]): pocket embedder config
            contact_embedder (dict[str:int]): contact embedder config
            contact_atom_embedder (dict[str:int]): contact atom embedder config
            substructure_embedder (dict[str:int]): substructure embedder config
            c_constraint_z (int): constraint z dimension
            initialize_method (str): initialize method
        """
        super(ConstraintEmbedder, self).__init__()
        self.pocket_embedder_config = pocket_embedder
        self.contact_embedder_config = contact_embedder
        self.contact_atom_embedder_config = contact_atom_embedder
        self.substructure_embedder_config = substructure_embedder
        # pocket embedder
        if self.pocket_embedder_config.get("enable", False):
            self.pocket_z_embedder = LinearNoBias(
                in_features=self.pocket_embedder_config.get("c_z_input", 1),
                out_features=c_constraint_z,
            )

        # token contact embedder
        if self.contact_embedder_config.get("enable", False):
            self.contact_z_embedder = LinearNoBias(
                in_features=contact_embedder["c_z_input"], out_features=c_constraint_z
            )

        # atom contact embedder
        if self.contact_atom_embedder_config.get("enable", False):
            self.contact_atom_z_embedder = LinearNoBias(
                in_features=contact_atom_embedder["c_z_input"],
                out_features=c_constraint_z,
            )

        # substructure embedder
        if self.substructure_embedder_config.get("enable", False):
            self.substructure_z_embedder = SubstructureEmbedder(
                n_classes=self.substructure_embedder_config.get("n_classes", 4),
                c_pair_dim=c_constraint_z,
                architecture=self.substructure_embedder_config.get(
                    "architecture", "mlp"
                ),
                hidden_dim=self.substructure_embedder_config.get("hidden_dim", 256),
                n_layers=self.substructure_embedder_config.get("n_layers", 3),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initialize_method == "zero":
                    nn.init.zeros_(module.weight)

    def forward(
        self,
        constraint_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:
        """

        Args:
            constraint_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): dict of input features

        Returns:
            torch.Tensor: token embedding
                [..., N_token, c_s]
        """
        z_constraint = None

        if self.pocket_embedder_config.get("enable", False):
            z_constraint = self.pocket_z_embedder(constraint_feature_dict["pocket"])

        if self.contact_embedder_config.get("enable", False):
            z_contact = self.contact_z_embedder(constraint_feature_dict["contact"])
            z_constraint = (
                z_contact if z_constraint is None else z_constraint + z_contact
            )

        if self.contact_atom_embedder_config.get("enable", False):
            z_contact_atom = self.contact_atom_z_embedder(
                constraint_feature_dict["contact_atom"]
            )
            z_constraint = (
                z_contact_atom
                if z_constraint is None
                else z_constraint + z_contact_atom
            )

        # substructure embedder
        if self.substructure_embedder_config.get("enable", False):
            z_substructure = self.substructure_z_embedder(
                constraint_feature_dict["substructure"]
            )
            z_constraint = (
                z_substructure
                if z_constraint is None
                else z_constraint + z_substructure
            )
        return z_constraint