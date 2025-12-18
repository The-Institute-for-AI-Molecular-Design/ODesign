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

import os

import torch

from src.utils.permutation import atom_permutation, chain_permutation
from src.api.model_interface import (
    PermutationInput,
    ODesignOutput,
    GroundTruth,
)

class SymmetricPermutation(object):
    """
    A symmetric permutation class for chain and atom permutations.

    Attributes:
        configs: Configuration settings for the permutation process.
        error_dir (str, optional): Directory to save error data. Defaults to None.
    """

    def __init__(self, configs, error_dir: str = None):
        self.configs = configs
        if error_dir is not None:
            self.chain_error_dir = os.path.join(error_dir, "chain_permutation")
            self.atom_error_dir = os.path.join(error_dir, "atom_permutation")
        else:
            self.chain_error_dir = None
            self.atom_error_dir = None

    def permute_label_to_match_mini_rollout(
        self,
        mini_coord: torch.Tensor,
        input_feature_dict: dict,
        label_dict: dict,
        label_full_dict: dict,
    ):
        """
        Apply permutation to label structure to match the predicted structure.
        This is mainly used to align label structure to the mini-rollout structure during training.

        Args:
            mini_coord (torch.Tensor): Coordinates of the predicted mini-rollout structure.
            input_feature_dict (dict): Input feature dictionary.
            label_dict (dict): Label dictionary.
            label_full_dict (dict): Full label dictionary.
        """

        assert mini_coord.dim() == 3

        log_dict = {}
        # 1. ChainPermutation: permute ground-truth chains to match mini-rollout prediction
        permuted_label_dict, chain_perm_log_dict, _, _ = chain_permutation.run(
            pred_coord=mini_coord[0],  # Only accepts a single structure
            input_feature_dict=input_feature_dict,
            label_full_dict=label_full_dict,
            max_num_chains=-1,
            permute_label=True,
            permute_by_pocket=False,
            error_dir=self.chain_error_dir,
            **self.configs.chain_permutation.configs,
        )
        if self.configs.chain_permutation.train.mini_rollout:
            label_dict.update(permuted_label_dict)
            log_dict.update(
                {
                    f"minirollout_perm/Chain-{k}": v
                    for k, v in chain_perm_log_dict.items()
                }
            )
        else:
            # Log only, not update the label_dict
            log_dict.update(
                {
                    f"minirollout_perm/Chain.F-{k}": v
                    for k, v in chain_perm_log_dict.items()
                }
            )

        # 2. AtomPermutation: permute ground-truth atoms to match mini-rollout prediction
        permuted_label_dict, atom_perm_log_dict, _ = atom_permutation.run(
            pred_coord=mini_coord[0],
            true_coord=label_dict["coordinate"],
            true_coord_mask=label_dict["coordinate_mask"],
            ref_space_uid=input_feature_dict["ref_space_uid"],
            atom_perm_list=input_feature_dict["atom_perm_list"],
            permute_label=True,
            error_dir=self.atom_error_dir,
            global_align_wo_symmetric_atom=self.configs.atom_permutation.global_align_wo_symmetric_atom,
        )

        if self.configs.atom_permutation.train.mini_rollout:
            label_dict.update(permuted_label_dict)
            log_dict.update(
                {f"minirollout_perm/Atom-{k}": v for k, v in atom_perm_log_dict.items()}
            )
        else:
            # Log only, not update the label_dict
            log_dict.update(
                {
                    f"minirollout_perm/Atom.F-{k}": v
                    for k, v in atom_perm_log_dict.items()
                }
            )

        return label_dict, log_dict

    def permute_diffusion_sample_to_match_label(
        self,
        input_data: PermutationInput,
        model_output: ODesignOutput,
        ground_truth: GroundTruth,
        stage: str,
        permute_by_pocket: bool = False,
    ):
        """
        Apply per-sample permutation to predicted structures to correct symmetries.
        Permutations are performed independently for each diffusion sample.

        Args:
            input_data (PermutationInput): Input feature.
            model_output (ODesignOutput): Prediction dictionary.
            label_dict (dict): Label dictionary.
            stage (str): Current stage of the diffusion process, in ['train', 'test'].
            permute_by_pocket (bool): Whether to permute by pocket (for PoseBusters dataset). Defaults to False.
        """

        assert model_output.coordinate.size(-2) == ground_truth.coordinate.size(
            -2
        ), "Cannot perform per-sample permutation on predicted structures if the label structure has more atoms."

        log_dict = {}
        permute_pred_indices, permute_label_indices = [], []
        if (
            stage != "train"
        ):  # During training stage, the label_dict is cropped after mini-rollout permutation.
            # In this case, chain permutation is not handled.

            # ChainPermutation: permute predicted chains to match label structure.

            (
                permuted_pred_dict,
                chain_perm_log_dict,
                permute_pred_indices,
                _,
            ) = chain_permutation.run(
                pred_coord=model_output.coordinate,
                input_data=input_data,
                ground_truth=ground_truth,
                max_num_chains=-1,
                permute_label=False,
                permute_by_pocket=permute_by_pocket
                and self.configs.chain_permutation.permute_by_pocket,
                error_dir=self.chain_error_dir,
                **self.configs.chain_permutation.configs,
            )
            if self.configs.chain_permutation.get(stage).diffusion_sample:
                model_output.update(permuted_pred_dict)
                log_dict.update(
                    {
                        f"sample_perm/Chain-{k}": v
                        for k, v in chain_perm_log_dict.items()
                    }
                )
            else:
                # Log only, not update the pred_dict.
                log_dict.update(
                    {
                        f"sample_perm/Chain.F-{k}": v
                        for k, v in chain_perm_log_dict.items()
                    }
                )

        # AtomPermutation: permute predicted atoms to match label structure.
        # Permutations are performed independently for each diffusion sample.
        if permute_by_pocket and self.configs.atom_permutation.permute_by_pocket:
            if ground_truth.pocket_mask.dim() == 2:
                # the 0-the pocket is assumed to be the `main` pocket
                pocket_mask = ground_truth.pocket_mask[0]
                ligand_mask = ground_truth.interested_ligand_mask[0]
            else:
                pocket_mask = ground_truth.pocket_mask
                ligand_mask = ground_truth.interested_ligand_mask
            chain_mask = self.get_chain_mask_from_atom_mask(
                pocket_mask + ligand_mask,
                atom_to_token_idx=input_data.atom_to_token_idx,
                token_asym_id=input_data.asym_id,
            )
            alignment_mask = pocket_mask
        else:
            chain_mask = 1
            alignment_mask = None

        permuted_pred_dict, atom_perm_log_dict, atom_perm_pred_indices = (
            atom_permutation.run(
                pred_coord=model_output.coordinate,
                true_coord=ground_truth.coordinate,
                true_coord_mask=ground_truth.coordinate_mask * chain_mask,
                ref_space_uid=input_data.ref_space_uid,
                atom_perm_list=input_data.atom_perm_list,
                permute_label=False,
                alignment_mask=alignment_mask,
                error_dir=self.atom_error_dir,
                global_align_wo_symmetric_atom=self.configs.atom_permutation.global_align_wo_symmetric_atom,
            )
        )
        if permute_pred_indices:
            # Update `permute_pred_indices' according to the results of atom permutation
            updated_permute_pred_indices = []
            assert len(permute_pred_indices) == len(atom_perm_pred_indices)
            for chain_perm_indices, atom_perm_indices in zip(
                permute_pred_indices, atom_perm_pred_indices
            ):
                updated_permute_pred_indices.append(
                    chain_perm_indices[atom_perm_indices]
                )
            permute_pred_indices = updated_permute_pred_indices
        elif atom_perm_pred_indices is not None:
            permute_pred_indices = [
                atom_perm_indices for atom_perm_indices in atom_perm_pred_indices
            ]

        if self.configs.atom_permutation.get(stage).diffusion_sample:
            model_output.update(permuted_pred_dict)
            log_dict.update(
                {f"sample_perm/Atom-{k}": v for k, v in atom_perm_log_dict.items()}
            )
        else:
            # Log only, not update the pred_dict.
            log_dict.update(
                {f"sample_perm/Atom.F-{k}": v for k, v in atom_perm_log_dict.items()}
            )

        return model_output, log_dict, permute_pred_indices, permute_label_indices

    @staticmethod
    def get_chain_mask_from_atom_mask(
        atom_mask: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        token_asym_id: torch.Tensor,
    ):
        """
        Generate a chain mask from an atom mask.

        This method maps atoms to their corresponding token indices and then to their asym IDs. It then filters these asym IDs based on the atom mask and returns a mask indicating which atoms belong to the filtered chains.

        Args:
            atom_mask (torch.Tensor): A boolean atom mask. Shape: [N_atom].
            atom_to_token_idx (torch.Tensor): A tensor mapping each atom to its corresponding token index. Shape: [N_atom].
            token_asym_id (torch.Tensor): A tensor containing the asym ID for each token. Shape: [N_token].

        Returns:
            torch.Tensor: Chain mask. Shape: [N_atom].

        """

        atom_asym_id = token_asym_id[atom_to_token_idx.long()].long()
        assert atom_asym_id.size(0) == atom_mask.size(0)
        masked_asym_id = torch.unique(atom_asym_id[atom_mask.bool()])
        return torch.isin(atom_asym_id, masked_asym_id)

    @staticmethod
    def get_asym_id_match(
        permute_indices: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
        token_asym_id: torch.Tensor,
    ) -> dict[int, int]:
        """Function to match asym IDs between original and permuted structure.

        Args:
            permute_indices (torch.Tensor): indices that specify the permuted ordering of atoms.
                [N_atom]
            atom_to_token_idx (torch.Tensor):  each entry maps an atom to its corresponding token index.
                [N_atom]
            token_asym_id (torch.Tensor): contains the asym ID for each token.
                [N_token]
        Returns:
            asym_id_match (Dict[int])
                A dictionary where the key is the original asym ID and the value is the permuted asym ID.
        """
        token_asym_id = token_asym_id.long()
        atom_to_token_idx = atom_to_token_idx.long()

        # Get the asym IDs for the original atoms
        original_atom_asym_id = token_asym_id[atom_to_token_idx]

        # Permute these IDs using the provided indices
        permuted_atom_asym_id = original_atom_asym_id[permute_indices]
        unique_asym_ids = torch.unique(original_atom_asym_id)

        asym_id_match = {}
        for ori_aid in unique_asym_ids:
            ori_aid = ori_aid.item()
            asym_mask = original_atom_asym_id == ori_aid
            perm_aid = permuted_atom_asym_id[asym_mask]

            assert (
                len(torch.unique(perm_aid)) == 1
            ), "Permuted asym ID must be unique for each original ID."

            asym_id_match[ori_aid] = perm_aid[0].item()

        return asym_id_match

    @staticmethod
    def permute_summary_confidence(
        summary_confidence_list: list[dict],
        permute_pred_indices: list[torch.Tensor],  # [N_atom]
        atom_to_token_idx: torch.Tensor,  # [N_atom]
        token_asym_id: torch.Tensor,  # [N_token]
        chain_keys: list[str] = [
            "chain_ptm",
            "chain_iptm",
            "chain_plddt",
            "chain_gpde",
        ],
        chain_pair_keys: list[str] = [
            "chain_pair_iptm",
            "chain_pair_iptm_global",
            "chain_pair_plddt",
            "chain_pair_gpde",
        ],
    ):
        """
        Permute summary confidence based on predicted indices.

        Args:
            summary_confidence_list (list[dict]): List of summary confidence dictionaries.
            permute_pred_indices (list[torch.Tensor]): List of predicted indices for permutation.
            atom_to_token_idx (torch.Tensor): Mapping from atoms to token indices.
            token_asym_id (torch.Tensor): Asym ID for each token.
            chain_keys (list[str], optional): Keys for chain-level confidence metrics. Defaults to ["chain_ptm", "chain_iptm", "chain_plddt"].
            chain_pair_keys (list[str], optional): Keys for chain pair-level confidence metrics. Defaults to ["chain_pair_iptm", "chain_pair_iptm_global", "chain_pair_plddt"].
        """

        assert len(summary_confidence_list) == len(permute_pred_indices)

        def _permute_one_sample(summary_confidence, permute_indices):
            # asym_id_match : {ori_asym_id: permuted_asym_id}
            asym_id_match = SymmetricPermutation.get_asym_id_match(
                permute_indices=permute_indices,
                atom_to_token_idx=atom_to_token_idx,
                token_asym_id=token_asym_id,
            )
            id_indices = torch.arange(len(asym_id_match), device=permute_indices.device)
            for i, j in asym_id_match.items():
                id_indices[j] = i

            # fix chain_id (asym_id) in summary_confidence
            for key in chain_keys:
                assert summary_confidence[key].dim() == 1
                summary_confidence[key] = summary_confidence[key][id_indices]
            for key in chain_pair_keys:
                assert summary_confidence[key].dim() == 2
                summary_confidence[key] = summary_confidence[key][:, id_indices]
                summary_confidence[key] = summary_confidence[key][id_indices, :]
            return summary_confidence, asym_id_match

        asym_id_match_list = []
        permuted_summary_confidence_list = []
        for i, (summary_confidence, perm_indices) in enumerate(
            zip(summary_confidence_list, permute_pred_indices)
        ):
            summary_confidence, asym_id_match = _permute_one_sample(
                summary_confidence, perm_indices
            )
            permuted_summary_confidence_list.append(summary_confidence)
            asym_id_match_list.append(asym_id_match)

        return permuted_summary_confidence_list, asym_id_match_list

    def permute_inference_output(
        self,
        input_data: PermutationInput,
        model_output: ODesignOutput,
        ground_truth: GroundTruth,
        permute_by_pocket: bool = False,
    ):
        """
        Permute predicted coordinates during inference.

        Args:
            input_feature_dict (dict): Input features dictionary.
            pred_dict (dict): Predicted dictionary.
            label_dict (dict): Label dictionary.
            permute_by_pocket (bool, optional): Whether to permute by pocket. Defaults to False.
        """
        # 1. Permute predicted coordinates
        model_output, log_dict, _, _ = (
            self.permute_diffusion_sample_to_match_label(
                input_data=input_data,
                model_output=model_output,
                ground_truth=ground_truth,
                stage="test",
                permute_by_pocket=permute_by_pocket,
            )
        )

        return model_output, log_dict
