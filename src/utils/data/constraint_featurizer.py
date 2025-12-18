import copy
import logging
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biotite.structure import Atom, AtomArray, get_residue_starts
from scipy.spatial.distance import cdist

from src.utils.license_register import register_license
from src.utils.data.constants import ELEMS, STD_RESIDUES
from src.utils.data.tokenizer import AtomArrayTokenizer, Token, TokenArray

logger = logging.getLogger(__name__)


@register_license('bytedance2024')
class ConstraintFeaturizer(object):
    def __init__(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        pad_value: float = 0,
        generator=None,
    ):
        self.token_array = token_array
        self.atom_array = atom_array
        self.pad_value = pad_value
        self.generator = generator
        self._get_base_info()

    @staticmethod
    def one_hot_encoder(feature: torch.Tensor, num_classes: int):
        # Create mask for padding values (-1)
        pad_mask = feature == -1

        # Replace -1 with 0 temporarily for F.one_hot
        feature = torch.where(pad_mask, torch.zeros_like(feature), feature)

        # Convert to one-hot
        one_hot = F.one_hot(feature, num_classes=num_classes).float()

        # Zero out the one-hot vectors for padding positions
        one_hot[pad_mask] = 0.0

        return one_hot

    def encode(self, feature: torch.Tensor, feature_type: str, **kwargs):
        if feature_type == "one_hot":
            return ConstraintFeaturizer.one_hot_encoder(
                feature, num_classes=kwargs.get("num_classes", -1)
            )
        elif feature_type == "continuous":
            return feature
        else:
            raise RuntimeError(f"Invalid feature_type: {feature_type}")

    def _get_base_info(self):
        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]
        self.asymid = torch.tensor(centre_atoms.asym_id_int, dtype=torch.long)
        self.is_ligand = torch.tensor(centre_atoms.is_ligand, dtype=torch.bool)
        self.is_protein = torch.tensor(centre_atoms.is_protein, dtype=torch.bool)
        self.entity_type_dict = {"P": self.is_protein, "L": self.is_ligand}

    def _get_generation_basics(self, distance_type: str = "center_atom"):
        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]

        # is_resolved mask
        self.token_resolved_mask = torch.tensor(
            centre_atoms.is_resolved, dtype=torch.bool
        )
        self.token_resolved_maskmat = (
            self.token_resolved_mask[:, None] * self.token_resolved_mask[None, :]
        )

        # distance matrix
        if distance_type == "center_atom":
            # center atom distance
            self.token_distance = torch.tensor(
                cdist(centre_atoms.coord, centre_atoms.coord), dtype=torch.float64
            )
        elif distance_type == "any_atom":
            # any atom distance
            all_atom_resolved_mask = (
                self.atom_array.is_resolved[:, None]
                * self.atom_array.is_resolved[None, :]
            )
            all_atom_distance = cdist(self.atom_array.coord, self.atom_array.coord)
            all_atom_distance[~all_atom_resolved_mask] = np.inf

            token_atoms_num = [
                len(_atoms)
                for _atoms in self.token_array.get_annotation("atom_indices")
            ]
            atom_token_num = np.repeat(
                np.arange(len(self.token_array)), token_atoms_num
            )

            self.token_distance = torch.zeros(
                (len(centre_atoms), len(centre_atoms)), dtype=torch.float64
            )
            for i, j in np.ndindex(self.token_distance.shape):
                atom_pairs_mask = np.ix_(atom_token_num == i, atom_token_num == j)
                self.token_distance[i, j] = np.min(all_atom_distance[atom_pairs_mask])
        elif distance_type == "atom":
            raise ValueError(
                "Not implement in this class, please use ContactAtomFeaturizer"
            )
        else:
            raise ValueError(f"Not recognized distance_type: {distance_type}")

    def generate(self):
        pass

    def generate_spec_constraint(self):
        pass


@register_license('bytedance2024')
class ConditionFeaturizer(object):
    def __init__(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        pad_value: float = 0,
        generator=None,
    ):
        self.token_array = token_array
        self.atom_array = atom_array
        self.pad_value = pad_value
        self.generator = generator

        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        self.centre_atoms = self.atom_array[token_centre_atom_indices]
        self.asymid = torch.tensor(self.centre_atoms.asym_id_int, dtype=torch.long)

    @staticmethod
    def one_hot_encoder(feature, num_classes):
        return F.one_hot(feature, num_classes=num_classes).float()

    def encode(self, feature, feature_type, **kwargs):
        if feature_type == "one_hot":
            return ConstraintFeaturizer.one_hot_encoder(
                feature, num_classes=kwargs.get("num_classes", -1)
            )
        elif feature_type == "continuous":
            return feature
        else:
            raise RuntimeError(f"Invalid feature_type: {feature_type}")

    def generate_spec_constraint(
        self,
    ):
        pass


@register_license('odesign2025')
class DistConditionFeaturizer(ConditionFeaturizer):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def generate_spec_constraint(self, contact_specifics, feature_type):
        """
        parse constraint from user specification
        """

        contact_feature = torch.full(
            (self.asymid.shape[0], self.asymid.shape[0], 2),
            fill_value=self.pad_value,
            dtype=torch.float32,
        )
        for token_list_1, token_list_2, max_distance in contact_specifics:

            token_id_1 = token_list_1[
                torch.randint(
                    high=token_list_1.shape[0], size=(1,), generator=self.generator
                ).item()
            ]
            token_id_2 = token_list_2[
                torch.randint(
                    high=token_list_2.shape[0], size=(1,), generator=self.generator
                ).item()
            ]

            contact_feature[token_id_1, token_id_2, 1] = max_distance
            contact_feature[token_id_2, token_id_1, 1] = max_distance
            contact_feature[token_id_1, token_id_2, 0] = 0
            contact_feature[token_id_2, token_id_1, 0] = 0

        contact_feature = self.encode(
            feature=contact_feature, feature_type=feature_type
        )
        return contact_feature

    def generate_spec_constraint_max_distance(self):
        """
        Generate maximum distance constraint between each token pair.
        
        This method computes the maximum pairwise distance between atoms in token pairs
        and uses it as a constraint only between condition and non-condition tokens.
        
        Returns:
            dict: Dictionary with 'contact' key containing constraint features.
                Shape: [N_token, N_token, 2]
        """
        n_token = len(self.token_array)
        # Initialize max distance constraint to 0
        contact_feature = torch.full(
            (n_token, n_token, 2),
            fill_value=0,
            dtype=torch.float32,
        )
        # Pad token atom coordinates to (24, 3)
        token_coords = np.stack([
            np.pad(
                self.atom_array.coord[idx], 
                ((0, 24 - self.atom_array.coord[idx].shape[0]), (0, 0)), 
                mode="constant", 
                constant_values=np.nan
            ) 
            for idx in self.token_array.get_annotation("atom_indices")
        ], axis=0)
        
        # Compute maximum distance between token pairs
        max_distance = np.nanmax(
            np.linalg.norm(
                token_coords[:, None, :, None, :] - token_coords[None, :, None, :, :], 
                axis=-1
            ), 
            axis=-1
        )
        dist_threshold = 8
        max_distance = np.where(max_distance <= dist_threshold, max_distance, 0)
        contact_feature[:, :, 1] = torch.from_numpy(max_distance).float()
        
        # Apply mask: compute max distance only between condition and non-condition tokens
        mask = torch.from_numpy(~self.centre_atoms.condition_token_mask)
        mask_2d = torch.logical_xor(mask.unsqueeze(1), mask.unsqueeze(0)).unsqueeze(-1)
        contact_feature = torch.where(
            mask_2d, contact_feature, torch.zeros_like(contact_feature)
        )
        return {'contact': contact_feature}
    
    def generate_spec_constraint_distance(self):
        """
        Generate distance constraint between condition token pairs.
        
        Returns:
            dict: Dictionary with 'contact' key containing constraint features.
                Shape: [N_token, N_token, 2]
        """
        # Initialize distance constraint to 0
        n_token = len(self.token_array)
        contact_feature = torch.full(
            (n_token, n_token, 2),
            fill_value=0,
            dtype=torch.float32,
        )
        rep_pos = torch.from_numpy(self.centre_atoms.coord)
        contact_feature[..., 1] = torch.cdist(rep_pos, rep_pos)  # [N, N]
        
        # Only keep constraints for condition token pairs
        is_condition_atom = (
            torch.Tensor(self.centre_atoms.is_resolved).long() * 
            torch.from_numpy(self.centre_atoms.condition_token_mask).long()
        )  # [N]
        is_condition_atom_pair = torch.outer(is_condition_atom, is_condition_atom)  # [N, N]
        contact_feature = torch.where(
            is_condition_atom_pair.bool().unsqueeze(-1), 
            contact_feature, 
            torch.zeros_like(contact_feature)
        )  # [N, N, 2]
        return {'contact': contact_feature}
    
    def generate_spec_constraint_distogram(self):
        """
        Generate distance constraint (distogram) inside condition tokens.
        
        This method computes pairwise distances between representative atoms
        of condition tokens to provide distance constraints.
        
        Returns:
            torch.Tensor: Distance constraint tensor.
                Shape: [N_token, N_token, 1]
        """
        # Compute pairwise distances between representative atom positions
        rep_pos = torch.from_numpy(self.centre_atoms.coord)
        contact_feature = torch.cdist(rep_pos, rep_pos) # [N, N]
        contact_feature = contact_feature + torch.randn_like(contact_feature)
        is_antigen_token = torch.from_numpy(self.centre_atoms.is_antigen_chain[:, None] * self.centre_atoms.is_antigen_chain[None, :])
        contact_feature = torch.where(is_antigen_token.bool(), contact_feature, torch.zeros_like(contact_feature)).unsqueeze(-1)  # [N, N, 1]
        return contact_feature