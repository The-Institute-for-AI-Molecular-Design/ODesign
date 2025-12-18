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
from pathlib import Path

import pickle
import numpy as np
import torch
from biotite.structure import AtomArray
import biotite.structure.io as strucio
import copy

from src.utils.data.constants import AA1_TO_3, DNA1_TO_3, RNA1_TO_3
from src.utils.data.misc import save_structure_cif
from src.utils.model.torch_utils import round_values


class DataDumper:
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        all_sequence_variants: list[dict],
        design_modality: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        prediction_save_dir = os.path.join(dump_dir, "predictions")
        os.makedirs(prediction_save_dir, exist_ok=True)

        self._save_structure_sequence(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
            all_sequence_variants=all_sequence_variants,
            design_modality=design_modality,
            sorted_indices=None,
            b_factor=None,
        )

        # save atom_array for benchmark and traceback
        output_array_path = os.path.join(
            dump_dir,
            "traceback.pkl",
        )
        with open(output_array_path, "wb") as f:
            pickle.dump(atom_array, f)

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data based on the dataset name, sample name, and seed.
        """
        dump_dir = os.path.join(
            self.base_dir, dataset_name, sample_name, f"seed_{seed}"
        )
        return dump_dir

    def _save_structure_sequence(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
        all_sequence_variants: list[dict],
        design_modality: str,
        sorted_indices: None,
        b_factor: torch.Tensor = None,
    ):
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        if sorted_indices is None:
            sorted_indices = range(N_sample)  # do not rank the output file

        for idx, rank in enumerate(sorted_indices):
            if all_sequence_variants is None or idx >= len(all_sequence_variants):
                variants_for_struct = []
            else:
                variants_for_struct = all_sequence_variants[idx] or []

            if not variants_for_struct:
                variants_for_struct = [None]

            num_seq_vars = len(variants_for_struct)
            for seq_var_idx in range(num_seq_vars):
                new_atom_array = copy.deepcopy(atom_array)

                variant = variants_for_struct[seq_var_idx]
                if variant is not None:
                    per_chain_edits = variant.get("per_chain", {})
                    self._apply_sequence_variant_to_atom_array(
                        new_atom_array,
                        per_chain_edits=per_chain_edits,
                        design_modality=design_modality,
                    )    

                    output_fpath = os.path.join(
                        prediction_save_dir,
                        f"{sample_name}_seed_{seed}_bb_{rank}_seq_{seq_var_idx}.cif",
                    )                    

                    if b_factor is not None:
                        # b_factor.shape == [N_sample, N_atom]
                        new_atom_array.set_annotation("b_factor", np.round(b_factor[idx], 2))

                    save_structure_cif(
                        atom_array=new_atom_array,
                        pred_coordinate=pred_coordinates[idx],
                        output_fpath=output_fpath,
                        entity_poly_type=entity_poly_type,
                        pdb_id=sample_name,
                    )

    def _apply_sequence_variant_to_atom_array(
        self,
        atom_array: AtomArray,
        per_chain_edits: dict[str, dict],
        design_modality: str,
    ):
        kind = design_modality.strip().lower()  # 'protein' / 'rna' / 'dna' / 'ligand'

        # ---------- ligand ----------
        if kind == "ligand":

            if not hasattr(atom_array, "element"):
                return

            elem = np.asarray(atom_array.element).astype(str)

            for ch, edit in per_chain_edits.items():
                atom_indices_list = edit.get("atom_indices", None)
                new_seq = edit.get("new_seq", "")

                if atom_indices_list is None:
                    continue

                if isinstance(new_seq, str):
                    tokens = new_seq.strip().split()
                else:
                    tokens = list(new_seq)

                flat_indices: list[int] = []
                for arr in atom_indices_list:
                    idxs = np.asarray(arr, dtype=int).reshape(-1)
                    flat_indices.extend(idxs.tolist())

                L = min(len(flat_indices), len(tokens))
                if L == 0:
                    continue

                for i in range(L):
                    idx = flat_indices[i]
                    e = tokens[i]
                    if not e:
                        continue
                    elem[idx] = e.upper()

            atom_array.element = elem

            return

        # ---------- protein / DNA / RNA:  1-letter → 3-letter patch ----------
        if kind == "protein":
            mapping = AA1_TO_3
        elif kind == "dna":
            mapping = DNA1_TO_3
        elif kind == "rna":
            mapping = RNA1_TO_3
        else:
            return

        has_cano = hasattr(atom_array, "cano_seq_resname")
        if has_cano:
            cano = np.asarray(atom_array.cano_seq_resname).astype(str)
        res_name = np.asarray(atom_array.res_name).astype(str)

        for ch, edit in per_chain_edits.items():
            atom_indices_list = edit.get("atom_indices", None)
            new_seq = edit.get("new_seq", "")

            if atom_indices_list is None:
                continue

            assert isinstance(new_seq, str), f"new_seq must be str, got {type(new_seq)}"

            L = min(len(atom_indices_list), len(new_seq))
            if L == 0:
                continue

            for i in range(L):
                aa1 = new_seq[i].upper()
                res3 = mapping.get(aa1, None)
                if res3 is None:
                    continue 

                idxs = np.asarray(atom_indices_list[i], dtype=int).reshape(-1)
                if idxs.size == 0:
                    continue

                res_name[idxs] = res3
                if has_cano:
                    cano[idxs] = res3

        atom_array.res_name = res_name
        if has_cano:
            atom_array.cano_seq_resname = cano

            