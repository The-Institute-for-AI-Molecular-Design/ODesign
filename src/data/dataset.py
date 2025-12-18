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
Dataset classes for protein structure prediction and generation.

This module provides dataset classes for handling protein structure data in various contexts:

Classes:
    BaseSingleDataset: Main dataset class for single data sources, supporting:
        - Data loading from mmCIF and bioassembly files
        - Data cropping and sampling strategies
        - MSA and template feature extraction
        - Data conditioning and masking for generation tasks
        - Feature and label generation for training/evaluation
    
    WeightedMultiDataset: Combines multiple datasets with weighted sampling for training
    
    InferenceDataset: Dataset class for inference/prediction tasks from JSON inputs

Functions:
    get_msa_featurizer: Create MSA featurizer from configuration
    get_weighted_pdb_weight: Calculate sample weight based on cluster size and composition
    calc_weights_for_df: Calculate weights for all samples in a dataframe
    get_sample_weights: Get sample weights for weighted or uniform sampling
    get_datasets: Create training and testing datasets from configuration

Key Features:
    - Support for proteins, nucleic acids, and ligands
    - Multiple cropping strategies (contiguous, spatial)
    - MSA feature extraction for proteins and RNA
    - Template feature extraction
    - Data augmentation (shuffling, masking)
    - Weighted sampling to handle data imbalance
    - Error handling and logging
"""

import json
import os
import logging
import warnings
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from src.utils.license_register import register_license
from src.utils.data.constants import EvaluationChainInterface
from src.utils.data.data_pipeline import DataPipeline
from src.utils.data.featurizer import Featurizer
from src.utils.data.msa_featurizer import MSAFeaturizer
from src.utils.data.tokenizer import TokenArray
from src.utils.data.cropping import CropData
from src.utils.data.file_io import read_indices_csv
from src.utils.data.mask_generator import MaskGenerator
from src.utils.data.misc import (
    get_data_shape_dict,
)
from src.utils.inference.inference_utils import SampleDictToFeatures
from src.api.data_interface import (
    OFeatureData,
    OLabelData,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


class BaseSingleDataset(Dataset):
    """
    Dataset class for a single data source.
    
    This class handles protein structure data from a single data source, supporting
    data cropping, MSA feature extraction, template feature extraction, and more.
    The __getitem__ method returns a dictionary containing features and labels.
    
    Main functionalities:
        - Read structure data from mmCIF files and bioassembly files
        - Support data cropping and sampling strategies
        - Extract MSA and template features
        - Generate training/testing features and labels
        - Support data conditioning and mask generation
    
    Attributes:
        mmcif_dir: Directory containing mmCIF files
        bioassembly_dict_dir: Directory containing bioassembly dictionary files
        indices_fpath: Path to the indices CSV file
        cropping_configs: Configuration dictionary for cropping
        msa_featurizer: MSA feature extractor
        template_featurizer: Template feature extractor
        data_condition: Whether to enable data conditioning
        mask_method: Masking method ('entity', 'token', 'atom', 'all')
        mask_type: Set of mask types
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        template_featurizer: Optional[Any] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Initialize the BaseSingleDataset.

        Args:
            mmcif_dir: Path to directory containing mmCIF files
            bioassembly_dict_dir: Path to directory containing bioassembly dictionary files (optional)
            indices_fpath: Path to the indices CSV file
            cropping_configs: Dictionary containing cropping configurations (crop size, method weights, etc.)
            msa_featurizer: MSA feature extractor instance (optional)
            template_featurizer: Template feature extractor instance (optional)
            name: Name of the dataset (optional)
            **kwargs: Additional configuration parameters including:
                - data_condition: Whether to enable data conditioning
                - mask_method: Masking method ('entity', 'token', 'atom', 'all')
                - mask_type: Set of mask types
                - return_atom_token_array: Whether to return atom and token arrays
                - use_hotspot_residue: Whether to use hotspot residues
                - ref_pos_augment: Whether to augment reference positions
                - shuffle_mols: Whether to shuffle molecules
                - shuffle_sym_ids: Whether to shuffle symmetry IDs
                - max_n_token: Maximum number of tokens limit
                - pdb_list: List of PDB IDs for filtering
                - exclusion_dict: Dictionary of exclusion rules
                - limits: Limit on number of data entries
                - error_dir: Directory for saving error logs

        Returns:
            None
        """
        super(BaseSingleDataset, self).__init__()
        self.data_condition = kwargs.get("data_condition", False)
        self.mask_method = kwargs.get("mask_method", "")
        self.mask_type = kwargs.get("mask_type", set())
        self.return_atom_token_array = kwargs.get("return_atom_token_array", False)
        self.use_hotspot_residue = kwargs.get("use_hotspot_residue", False)
        self.bad_lig_types = [
            'glycans', 'excluded_ligand', 'IONS_GLYCANS_LIGAND_EXCLUSION_PBV2_COMMON_NATURAL_LIGANDS'
        ]

        self.print_config_table()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)

        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.pdb_list = kwargs.get("pdb_list", None)
        if len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.msa_featurizer = msa_featurizer
        self.template_featurizer = template_featurizer

        # Read data
        self.indices_list = self.read_indices_list(indices_fpath)

    @register_license('odesign2025')
    def print_config_table(self):
        """
        Print dataset configuration information in table format.
        
        Displays current dataset configuration parameters including data conditioning,
        masking method, mask types, whether to return atom and token arrays, and
        whether to use hotspot residues.

        Args:
            None

        Returns:
            None, prints configuration table directly to console
        """
        # Create configuration table
        config_data = [
            ['Parameter', 'Value'],
            ['Data condition', str(self.data_condition)],
            ['Mask method', str(self.mask_method)],
            ['Mask type', str(self.mask_type)],
            ['Return atom and token array', str(self.return_atom_token_array)],
            ['Use hotspot residue', str(self.use_hotspot_residue)]
        ]
        
        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in config_data) for i in range(len(config_data[0]))]
        
        # Print table
        print("\n" + "=" * (sum(col_widths) + 7))
        print("Dataset Configuration")
        print("=" * (sum(col_widths) + 7))
        
        for i, row in enumerate(config_data):
            formatted_row = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
            print(f"| {formatted_row} |")
            if i == 0:  # Add separator after header
                print("|" + "-" * (sum(col_widths) + 5) + "|")
        
        print("=" * (sum(col_widths) + 7) + "\n")

    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Read a list of PDB IDs from a file or directly from a list.
        
        This method supports two input formats:
            1. A list of PDB IDs (returned directly)
            2. A file path containing PDB IDs (one per line)

        Args:
            pdb_list: Either a list of PDB IDs or a file path string containing PDB IDs

        Returns:
            Optional[list]: List of PDB IDs if input is valid, None if input is None
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    @register_license('odesign2025')
    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Read and process indices list from a CSV file.
        
        This method reads the indices file and applies multiple filtering operations based on 
        configuration parameters:
            1. Filter by molecule type based on data conditioning and mask type
            2. Filter by PDB ID list
            3. Filter by maximum token count
            4. Filter by exclusion rules dictionary
            5. Group by PDB ID (if enabled)
            6. Sort by token count (if enabled)
            7. Filter evaluation chain interfaces (if enabled)
            8. Limit number of entries (if set)

        Args:
            indices_fpath: Path to the CSV file containing indices information

        Returns:
            Processed DataFrame, either a single DataFrame or a list of DataFrames grouped by PDB ID
            
        Data filtering workflow:
            - For data conditioning mode, filters data containing specific molecule types based on mask_type
            - If only ligand masking, excludes specific ligand types (e.g., glycans, ions)
            - Prints data statistics after each filtering step
        """
        indices_list = read_indices_csv(indices_fpath)

        if self.data_condition and self.mask_type & set(["ligand", "prot", "nuc"]):
            INCLUDE_TYPES = list(self.mask_type)
            indices_list = indices_list[
                (
                    indices_list.mol_1_type.isin(INCLUDE_TYPES) |
                    indices_list.mol_2_type.isin(INCLUDE_TYPES)
                )
            ]
            if 'ligand' in self.mask_type and len(self.mask_type) == 1:
                indices_list = indices_list[
                    (
                        (~indices_list.sub_mol_1_type.isin(self.bad_lig_types)) & 
                        (~indices_list.sub_mol_2_type.isin(self.bad_lig_types))
                    )
                ]

        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
        self.print_data_stats(indices_list)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                indices_list = indices_list[
                    indices_list["eval_type"].apply(
                        lambda x: x in EvaluationChainInterface
                    )
                ]
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
        return indices_list

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Print statistics about the dataset including molecular group type distribution.
        
        This method computes and logs statistics about:
            - Molecule type combinations (e.g., prot-prot, prot-ligand)
            - Data distribution across different molecule groups
            - Cluster distribution for each molecule group
            - Total number of unique PDB IDs

        Args:
            df: DataFrame containing the indices list with columns:
                - mol_1_type, mol_2_type: Molecule types for the two interacting molecules
                - cluster_id: Cluster identifier (optional)
                - pdb_id: PDB identifier

        Returns:
            None, logs statistics to logger
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        df["mol_group_type"] = df.apply(
            lambda row: "_".join(
                sorted(
                    [
                        str(row["mol_1_type"]),
                        str(row["mol_2_type"]).replace("nan", "intra"),
                    ]
                )
            ),
            axis=1,
        )

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Save error data for a specific index to a JSON file in the error directory.
        
        This method saves the sample indices and error message when data processing fails,
        allowing for later analysis and debugging.

        Args:
            idx: Index of the data sample that caused the error
            error_message: Error message string to be saved

        Returns:
            None, saves error information to a JSON file in the error directory
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    def __getitem__(self, idx: int):
        """
        Retrieve a data sample by processing the given index.
        
        This method processes the data sample at the specified index. If an error occurs
        during processing, it will either save the error data or randomly sample another
        index (if random_sample_if_failed is enabled). It will retry up to 10 times.

        Args:
            idx: Index of the data sample to retrieve

        Returns:
            dict: Dictionary containing the processed data sample with keys:
                - 'feature_data': OFeatureData object containing input features
                - 'label_data': OLabelData object containing labels
                - 'label_full_data': OLabelData object containing full complex labels
                - 'basic': Dictionary containing basic information (PDB ID, dimensions, etc.)
                - 'atom_array': AtomArray object (if return_atom_token_array is True)
                - 'token_array': TokenArray object (if return_atom_token_array is True)
        """
        # Try at most 10 times
        for _ in range(10):
            try:
                data = self.process_one(idx, return_atom_token_array=self.return_atom_token_array)
                return data
            except Exception as e:
                error_message = f"{e} at idx {idx}:\n{traceback.format_exc()}"
                self.save_error_data(idx, error_message)

                if self.random_sample_if_failed:
                    logger.exception(f"[skip data {idx}] {error_message}")
                    # Random sample an index
                    idx = random.choice(range(len(self.indices_list)))
                    continue
                else:
                    raise Exception(e)
        return data

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get bioassembly data for the specified index.
        
        This method retrieves the bioassembly data dictionary corresponding to the sample
        index, containing atom arrays, token arrays, and other structural information.

        Args:
            idx: Data sample index

        Returns:
            tuple: Tuple containing three elements:
                - sample_indice: Sample indices information (pandas Series) containing:
                    - pdb_id: PDB identifier
                    - chain_1_id, chain_2_id: Chain identifiers
                    - type: Data type ('chain' or 'interface')
                    - mol_1_type, mol_2_type: Molecule types
                - bioassembly_dict: Bioassembly data dictionary containing:
                    - 'atom_array': AtomArray object with atomic information
                    - 'token_array': TokenArray object with token information
                    - 'pdb_id': PDB identifier
                    - 'entity_poly_type': Entity polymer type information
                - bioassembly_dict_fpath: Path to the bioassembly dictionary file
        """
        sample_indice = self._get_sample_indice(idx=idx)
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        Reassign discontinuous chain IDs to be continuous.
        
        In experiments using training sets, pre-stored AtomArray may have discontinuous
        chain IDs due to filtering. This method reassigns them to be continuous.
        
        Example: (3x6u)
            asym_id_int = [0, 1, 2, ... 18, 20] -> [0, 1, 2, ..., 18, 19]

        Args:
            atom_array: AtomArray object with potentially discontinuous chain IDs

        Returns:
            AtomArray: AtomArray with reassigned continuous chain IDs for:
                - asym_id_int: Asymmetric unit ID
                - entity_id_int: Entity ID
                - sym_id_int: Symmetry ID
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            map_dict = {i: idx for idx, i in enumerate(array_uniq)}
            new_array = np.vectorize(map_dict.get)(array)
            return new_array

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array based on molecule IDs.
        
        Atoms/tokens with the same mol_id are shuffled as an integrated component,
        maintaining their internal structure while randomizing the order of molecules.

        Args:
            token_array: TokenArray object to be shuffled
            atom_array: AtomArray object to be shuffled

        Returns:
            tuple: Tuple containing two elements:
                - token_array: Shuffled TokenArray
                - atom_array: Shuffled AtomArray
                
        Note:
            Molecules are shuffled as complete units - all tokens and atoms belonging
            to the same molecule are kept together.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        token_array, atom_array, _, _ = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random symmetry IDs for chains of the same entity ID.
        
        This method randomly shuffles symmetry IDs within each entity, which is useful
        for data augmentation and preventing the model from overfitting to specific
        chain orderings.
        
        Example:
            When entity_id = 0:
                sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
            When entity_id = 1:
                sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]

        Args:
            atom_array: AtomArray object with symmetry ID information

        Returns:
            AtomArray: AtomArray with randomly shuffled symmetry IDs within each entity
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            map_dict = dict(zip(x_unique, x_shuffled))
            new_x = np.vectorize(map_dict.get)(x)
            return new_x.copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    @register_license('odesign2025')
    def process_one(
        self, idx: int, return_atom_token_array: bool = False
    ) -> dict[str, dict]:
        """
        Process a single data sample with full pipeline.
        
        This method retrieves bioassembly data, applies various transformations
        (shuffling, cropping, masking), extracts features and labels, and optionally
        returns the processed atom and token arrays.
        
        Processing pipeline:
            1. Retrieve bioassembly data
            2. Filter to reference chains only (if enabled)
            3. Shuffle molecules (if enabled)
            4. Shuffle symmetry IDs (if enabled)
            5. Reassign continuous chain IDs (if enabled)
            6. Crop data to specified size
            7. Apply masking (if data conditioning is enabled)
            8. Extract features and labels

        Args:
            idx: Index of the data sample to process
            return_atom_token_array: Whether to return the processed atom and token arrays

        Returns:
            dict: Dictionary containing:
                - 'feature_data': OFeatureData object with input features including:
                    - Atom coordinates and features
                    - Token features
                    - MSA features
                    - Template features
                    - Hotspot features
                - 'label_data': OLabelData object with labels for training
                - 'label_full_data': OLabelData object with full complex labels
                - 'basic': Dictionary with basic information:
                    - 'pdb_id': PDB identifier
                    - 'entity_poly_type': Entity polymer type information
                    - 'N_asym', 'N_token', 'N_atom', 'N_msa': Dimension counts
                    - 'chain_id': List of chain identifiers
                - 'atom_array': AtomArray object (if return_atom_token_array=True)
                - 'token_array': TokenArray object (if return_atom_token_array=True)
        """

        sample_indice, bioassembly_dict, bioassembly_dict_fpath = (
            self._get_bioassembly_data(idx=idx)
        )

        if self.use_reference_chains_only:
            # Get the reference chains
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            # Remove other chains from the bioassembly_dict
            # Remove them safely using the crop method
            token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
            )
            token_chain_id = bioassembly_dict["atom_array"][
                token_centre_atom_indices
            ].chain_id
            is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _ = (
                CropData.select_by_token_indices(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
                )
            )

        if self.shuffle_mols:
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"] = (
                self._shuffle_array_based_on_mol_id(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                )
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        focus_on_ligand = (
            (sample_indice.mol_1_type == 'ligand')
            or 
            (sample_indice.mol_2_type == 'ligand')
        ) and (
            sample_indice.sub_mol_1_type not in self.bad_lig_types
        ) and (
            sample_indice.sub_mol_2_type not in self.bad_lig_types
        ) and (
            'ligand' in self.mask_type
        )

        # Crop
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_template_features,
            _,
            ref_chain_indices,
        ) = self.crop(
            focus_on_ligand=focus_on_ligand,
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            **self.cropping_configs,
        )

        # Create backbone subsample mask
        # Mask methods include three types: entity, token, atom, all
        # entity: Mask by chain unit, keep backbone atoms, mask other side chain atoms
        # token: Mask by token unit, keep backbone atoms, mask other side chain atoms
        # atom: Mask the entire sequence, keep backbone atoms, and additionally keep some atoms as condition, mask other side chain atoms
        # all: Mask the entire sequence, only keep backbone atoms
        # condition_atom_mask: Kept condition atoms, including all atoms of condition tokens, as well as backbone and tip atoms of condition tokens
        # condition_token_mask: Kept condition tokens, including all atoms of condition tokens
        masked_asym_ids = []
        if self.data_condition:
            mask_generator = MaskGenerator(
                cropped_token_array, 
                cropped_atom_array, 
                self.mask_type, 
                ref_chain_indices=ref_chain_indices if focus_on_ligand else None
            )
            (
                cropped_atom_array, 
                cropped_token_array
            ) = mask_generator.apply_mask(
                data_condition=self.data_condition,
                mask_method=self.mask_method,
            )
            mask_method = mask_generator.mask_method
            del mask_generator
        else:
            mask_method = None
            cropped_atom_array.set_annotation(
                'condition_token_mask', 
                np.ones(len(cropped_atom_array), dtype=bool)
            )

        feature_data, label_data, label_full_data = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            masked_asym_ids=masked_asym_ids,
            data_condition=self.data_condition,
            mask_method=mask_method,
            is_spatial_crop="spatial" in crop_method.lower(),
        )

        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "entity_poly_type": bioassembly_dict["entity_poly_type"],
            "N_asym": torch.tensor([feature_data.num_asym]),
            "N_token": torch.tensor([feature_data.num_token]),
            "N_atom": torch.tensor([feature_data.num_atom]),
            "N_msa": torch.tensor([feature_data.num_msa]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feature_data.prot_pair_num_alignments]),
            "N_msa_prot_unpair": torch.tensor([feature_data.prot_unpair_num_alignments]),
            "N_msa_rna_pair": torch.tensor([feature_data.rna_pair_num_alignments]),
            "N_msa_rna_unpair": torch.tensor([feature_data.rna_unpair_num_alignments]),
        }

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feature_data[f"is_{mol_type}"]
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feature_data.atom_to_token_idx[mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "feature_data": feature_data,
            "label_data": label_data,
            "label_full_data": label_full_data,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["atom_array"] = cropped_atom_array
            data["token_array"] = cropped_token_array
        return data

    @register_license('odesign2025')
    def crop(
        self,
        focus_on_ligand: bool,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop bioassembly data based on specified configurations.
        
        This method crops the structure data to a specified size using various cropping
        strategies (e.g., contiguous, spatial) with weighted sampling.

        Args:
            focus_on_ligand: Whether to focus cropping on ligand regions
            sample_indice: Sample indices information (pandas Series)
            bioassembly_dict: Bioassembly data dictionary containing atom and token arrays
            crop_size: Target number of tokens after cropping
            method_weights: Weights for different cropping methods
            contiguous_crop_complete_lig: Whether to keep complete ligands in contiguous cropping
            spatial_crop_complete_lig: Whether to keep complete ligands in spatial cropping
            drop_last: Whether to drop the last incomplete crop
            remove_metal: Whether to remove metal ions

        Returns:
            tuple: Tuple containing:
                - crop_method: Name of the cropping method used (str)
                - cropped_token_array: Cropped TokenArray object
                - cropped_atom_array: Cropped AtomArray object
                - cropped_msa_features: Dictionary of cropped MSA features
                - cropped_template_features: Dictionary of cropped template features
                - _: Placeholder (unused)
                - ref_chain_indices: Indices of reference chains in the cropped data
        """
        return DataPipeline.crop(
            focus_on_ligand=focus_on_ligand,
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieve sample indices for a given index.
        
        If the dataset is grouped by PDB ID, returns the first row of the PDB group.
        Otherwise, returns the row at the specified index.

        Args:
            idx: Index of the data sample to retrieve

        Returns:
            pd.Series: Pandas Series containing the sample indices with fields:
                - pdb_id: PDB identifier
                - chain_1_id: First chain identifier
                - chain_2_id: Second chain identifier (may be NaN for single chains)
                - type: Data type ('chain' or 'interface')
                - mol_1_type: Type of first molecule ('prot', 'nuc', 'ligand')
                - mol_2_type: Type of second molecule
                - num_tokens: Number of tokens
                - eval_type: Evaluation type
                - cluster_id: Cluster identifier
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice

    def _get_pdb_indice(self, idx: int) -> pd.core.series.Series:
        """
        Retrieve PDB indices for a given index.
        
        If the dataset is grouped by PDB ID, returns the entire PDB group.
        Otherwise, returns a single row at the specified index.

        Args:
            idx: Index of the data sample

        Returns:
            pd.DataFrame or pd.Series: PDB indices information
        """
        if self.group_by_pdb_id:
            pdb_indice = self.indices_list[idx].copy()
        else:
            pdb_indice = self.indices_list.iloc[idx : idx + 1].copy()
        return pdb_indice

    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieve evaluation chain/interface masks for a given index.
        
        This method generates masks for evaluating specific chains or interfaces,
        filtering for evaluation types defined in EvaluationChainInterface.

        Args:
            idx: Index of the data sample
            atom_array_chain_id: Array containing chain IDs of the atom array

        Returns:
            tuple: Tuple containing four elements:
                - eval_type: Array of evaluation types (np.ndarray) with values from
                    EvaluationChainInterface (e.g., 'protein-protein', 'protein-ligand')
                - cluster_id: Array of cluster IDs (np.ndarray)
                - chain_1_mask: Boolean tensor indicating atoms belonging to first chain,
                    shape [N_eval, N_atom] (torch.Tensor)
                - chain_2_mask: Boolean tensor indicating atoms belonging to second chain,
                    shape [N_eval, N_atom] (torch.Tensor)
                    
        Raises:
            ValueError: If no valid chain/interface for evaluation is found
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].apply(lambda x: x in EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the PDB."
            )

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    @register_license('odesign2025')
    def get_constraint_feature(
        self,
        idx,
        atom_array,
        token_array,
        msa_features,
        max_entity_mol_id,
        full_atom_array,
    ):
        """
        Generate constraint features for the data sample.
        
        This method generates constraint features using the constraint generator,
        which may include distance constraints, contact constraints, or other
        structural constraints for guided generation.

        Args:
            idx: Index of the data sample
            atom_array: AtomArray object with atomic information
            token_array: TokenArray object with token information
            msa_features: Dictionary of MSA features
            max_entity_mol_id: Maximum entity molecule ID
            full_atom_array: Full AtomArray object (before cropping)

        Returns:
            tuple: Tuple containing:
                - token_array: Updated TokenArray object
                - atom_array: Updated AtomArray object
                - features_dict: Dictionary containing:
                    - 'constraint_feature': Constraint feature dictionary
                    - Other feature information
                    - 'constraint_log_info': Logging information dictionary
                - msa_features: Updated MSA features dictionary
                - full_atom_array: Updated full AtomArray object
        """
        sample_indice = self._get_sample_indice(idx=idx)
        pdb_indice = self._get_pdb_indice(idx=idx)
        features_dict = {}
        (
            token_array,
            atom_array,
            msa_features,
            constraint_feature_dict,
            feature_info,
            log_dict,
            full_atom_array,
        ) = self.constraint_generator.generate(
            atom_array,
            token_array,
            sample_indice,
            pdb_indice,
            msa_features,
            max_entity_mol_id,
            full_atom_array,
        )
        features_dict["constraint_feature"] = constraint_feature_dict
        features_dict.update(feature_info)
        features_dict["constraint_log_info"] = log_dict
        return token_array, atom_array, features_dict, msa_features, full_atom_array

    @register_license('odesign2025')
    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
        data_condition: bool = False,
        masked_asym_ids: list[int] = None,
        mask_method: str = "",
    ) -> tuple[OFeatureData, OLabelData, OLabelData]:
        """
        Extract features and labels for a given data point.
        
        This method uses a Featurizer object to extract input features and labels,
        then applies additional processing steps to generate MSA features, template
        features, hotspot features, and full complex labels.

        Args:
            idx: Index of the data point
            token_array: TokenArray object representing the sequence
            atom_array: AtomArray object containing atomic information
            msa_features: Dictionary of MSA features from MSA featurizer
            template_features: Dictionary of template features from template featurizer
            full_atom_array: Full AtomArray object containing all atoms (before cropping)
            is_spatial_crop: Whether spatial cropping was applied (default: True)
            data_condition: Whether data conditioning is enabled (default: False)
            masked_asym_ids: List of masked asymmetric unit IDs (default: None)
            mask_method: Masking method used ('entity', 'token', 'atom', 'all')

        Returns:
            tuple: Tuple containing three OFeatureData/OLabelData objects:
                - feature_data: OFeatureData object with all input features including:
                    - Atom positions and features (ref_pos, ref_mask, ref_element, etc.)
                    - Token features (token_index, residue_index, etc.)
                    - MSA features (msa, msa_mask, etc.)
                    - Template features (template_pseudo_beta, template_mask, etc.)
                    - Hotspot features (hotspot_mask, etc.)
                    - Dimension information (num_token, num_atom, num_asym, etc.)
                - label_data: OLabelData object with training labels:
                    - atom_gt_coords: Ground truth atom coordinates
                    - token_gt_coords: Ground truth token coordinates
                    - eval_type, cluster_id: Evaluation metadata (if find_eval_chain_interface)
                    - chain_1_mask, chain_2_mask: Chain masks for evaluation
                - label_full_data: OLabelData object with full complex labels for multi-chain
                    permutation evaluation

        Raises:
            ValueError: If required data cannot be found or processed
        """
        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
            data_condition=data_condition,
            is_distillation=self.is_distillation,
            template_features=template_features,
            msa_features=msa_features,
            mask_method=mask_method,
            use_hotspot_residue=self.use_hotspot_residue,            
        )
        features_dict = feat.get_all_input_features()
        feature_data = OFeatureData.from_feature_dict(features_dict)

        default_feat_shape_dict, _ = get_data_shape_dict(
            num_token=feature_data.num_token,
            num_atom=feature_data.num_atom,
            num_msa=feature_data.default_num_msa,
            num_templ=feature_data.default_num_templ,
            num_pocket=feature_data.default_num_pocket,
        )

        # prepare MSA features 
        msa_features = feat.get_msa_features(
            features_dict=features_dict,
            feat_shape=default_feat_shape_dict
        )
        feature_data.update(msa_features)

        # prepare template features
        template_features = feat.get_template_features(
            feat_shape=default_feat_shape_dict
        )
        feature_data.update(template_features)

        # prepare hotspot features
        hotspot_features = feat.get_hotspot_features(
            feat_shape=default_feat_shape_dict,
            interface_minimal_distance=8,
            min_distance=True
        )
        feature_data.update(hotspot_features)

        feature_data.update(masked_asym_ids=masked_asym_ids)

        labels_dict = feat.get_labels()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            eval_type, cluster_id, chain_1_mask, chain_2_mask = (
                self._get_eval_chain_interface_mask(
                    idx=idx, atom_array_chain_id=full_atom_array.chain_id
                )
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        label_data = OLabelData.from_label_dict(labels_dict)
        label_full_data = OLabelData.from_label_dict(label_full_dict)

        return feature_data, label_data, label_full_data


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Create and return an MSAFeaturizer object based on provided configurations.
    
    This function constructs an MSA featurizer with protein and RNA MSA configurations
    if MSA is enabled in the config.

    Args:
        configs: ConfigDict containing MSA featurizer configurations including:
            - data.msa.enable: Whether MSA is enabled
            - data.msa.prot: Protein MSA configuration
            - data.msa.rna: RNA MSA configuration
            - data.msa.merge_method: Method for merging MSAs
            - data.msa.max_size: Maximum MSA size per stage
            - data.msa.enable_rna_msa: Whether RNA MSA is enabled
        dataset_name: Name of the dataset (e.g., 'pdb', 'cameo')
        stage: Stage of the dataset ('train' or 'test')

    Returns:
        MSAFeaturizer: MSAFeaturizer object configured for the dataset, or None if MSA
            is not enabled in the configurations
    """
    if "msa" in configs["data"] and configs["data"]["msa"]["enable"]:
        msa_info = configs["data"]["msa"]
        msa_args = deepcopy(msa_info)

        if "msa" in (dataset_config := configs["data"][dataset_name]):
            for k, v in dataset_config["msa"].items():
                if k not in ["prot", "rna"]:
                    msa_args[k] = v
                else:
                    for kk, vv in dataset_config["msa"][k].items():
                        msa_args[k][kk] = vv

        prot_msa_args = msa_args["prot"]
        prot_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        rna_msa_args = msa_args["rna"]
        rna_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        return MSAFeaturizer(
            prot_msa_args=prot_msa_args,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=configs.data.msa.enable_rna_msa,
        )

    else:
        return None


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with sampling weights.
    
    This class combines multiple datasets with individual datapoint weights and overall
    dataset weights for weighted sampling during training.
    
    Attributes:
        datasets: List of Dataset objects
        dataset_names: List of dataset names
        datapoint_weights: List of weight lists for each datapoint in each dataset
        dataset_sample_weights: Tensor of overall weights for each dataset
        merged_datapoint_weights: Tensor of all datapoint weights merged across datasets
        dataset_indices: List mapping merged indices to dataset indices
        within_dataset_indices: List mapping merged indices to within-dataset indices
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        """
        Initialize the WeightedMultiDataset.
        
        Args:
            datasets: List of Dataset objects to be combined
            dataset_names: List of dataset names corresponding to the datasets
            datapoint_weights: List of lists containing sampling weights for each
                datapoint in each dataset. Each inner list corresponds to one dataset.
            dataset_sample_weights: List of torch tensors containing overall sampling
                weights for each dataset

        Returns:
            None
        """
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        self.init_datasets()

    def init_datasets(self):
        """
        Calculate global weights of each datapoint across all datasets for sampling.
        
        This method:
            1. Normalizes datapoint weights within each dataset
            2. Scales by dataset-level weights
            3. Merges all weights into a single tensor
            4. Creates index mappings for efficient sampling

        Returns:
            None, updates instance attributes:
                - merged_datapoint_weights: Tensor of all normalized weights
                - weight: Total weight sum
                - dataset_indices: Mapping from merged index to dataset index
                - within_dataset_indices: Mapping from merged index to within-dataset index
        """
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        """
        Return the total number of datapoints across all datasets.

        Returns:
            int: Total number of datapoints
        """
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        """
        Retrieve a datapoint by its merged index.
        
        Maps the merged index to the appropriate dataset and within-dataset index.

        Args:
            index: Merged index across all datasets

        Returns:
            dict: Dictionary containing feature_data, label_data, label_full_data, and basic info
        """
        return self.datasets[self.dataset_indices[index]][
            self.within_dataset_indices[index]
        ]


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Calculate sample weight for a chain/interface in a weighted PDB dataset.
    
    The weight is calculated as:
        weight = beta * sum(alpha_i * count_i) / (cluster_size + eps)
    
    Where:
        - beta: Weight multiplier for data type (chain vs interface)
        - alpha_i: Weight multiplier for each molecule type
        - count_i: Number of molecules of each type
        - cluster_size: Number of similar structures in the cluster

    Args:
        data_type: Type of data, either 'chain' or 'interface'
        cluster_size: Cluster size of this chain/interface (number of similar structures)
        chain_count: Dictionary with counts of each molecule type, e.g., 
            {"prot": int, "nuc": int, "ligand": int}
        eps: Small epsilon value to avoid division by zero (default: 1e-9)
        beta_dict: Dictionary containing beta values for 'chain' and 'interface'.
            If None, defaults to {"chain": 0.5, "interface": 1}
        alpha_dict: Dictionary containing alpha values for different chain types.
            If None, defaults to {"prot": 3, "nuc": 3, "ligand": 1}

    Returns:
        float: Calculated weight for the given chain/interface. Higher weights indicate
            higher sampling probability. Weights are inversely proportional to cluster
            size (downweight overrepresented structures) and proportional to molecule
            counts weighted by their alpha values.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame, 
    beta_dict: dict[str, Any], 
    alpha_dict: dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate sample weights for each example in the dataframe.
    
    This function computes sampling weights for each row in the indices dataframe,
    accounting for:
        - Data type (chain vs interface)
        - Cluster size (to reduce redundancy)
        - Molecule type composition
        - Entity member count in the same assembly
    
    The weight formula is:
        weight = get_weighted_pdb_weight(...) / pdb_sorted_entity_id_member_num

    Args:
        indices_df: Pandas DataFrame containing the indices with columns:
            - pdb_id: PDB identifier
            - assembly_id: Assembly identifier
            - entity_1_id, entity_2_id: Entity identifiers
            - cluster_id: Cluster identifier for grouping similar structures
            - type: Data type ('chain' or 'interface')
            - mol_1_type, mol_2_type: Molecule types
        beta_dict: Dictionary containing beta values for different data types
            (e.g., {"chain": 0.5, "interface": 1})
        alpha_dict: Dictionary containing alpha values for different molecule types
            (e.g., {"prot": 3, "nuc": 3, "ligand": 1})

    Returns:
        pd.DataFrame: DataFrame with an additional 'weights' column containing the
            calculated sampling weights for each example
    """
    # Specific to assembly, and entities (chain or interface)
    indices_df["pdb_sorted_entity_id"] = indices_df.apply(
        lambda x: f"{x['pdb_id']}_{x['assembly_id']}_{'_'.join(sorted([str(x['entity_1_id']), str(x['entity_2_id'])]))}",
        axis=1,
    )

    entity_member_num_dict = {}
    for pdb_sorted_entity_id, sub_df in indices_df.groupby("pdb_sorted_entity_id"):
        # Number of repeatative entities in the same assembly
        entity_member_num_dict[pdb_sorted_entity_id] = len(sub_df)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.apply(
        lambda x: entity_member_num_dict[x["pdb_sorted_entity_id"]], axis=1
    )

    cluster_size_record = {}
    for cluster_id, sub_df in indices_df.groupby("cluster_id"):
        cluster_size_record[cluster_id] = len(set(sub_df["pdb_sorted_entity_id"]))

    weights = []
    for _, row in indices_df.iterrows():
        data_type = row["type"]
        cluster_size = cluster_size_record[row["cluster_id"]]
        chain_count = {"prot": 0, "nuc": 0, "ligand": 0}
        for mol_type in [row["mol_1_type"], row["mol_2_type"]]:
            if chain_count.get(mol_type) is None:
                continue
            chain_count[mol_type] += 1
        # Weight specific to (assembly, entity(chain/interface))
        weight = get_weighted_pdb_weight(
            data_type=data_type,
            cluster_size=cluster_size,
            chain_count=chain_count,
            beta_dict=beta_dict,
            alpha_dict=alpha_dict,
        )
        weights.append(weight)
    indices_df["weights"] = weights / indices_df["pdb_sorted_entity_id_member_num"]
    return indices_df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Compute sample weights based on the specified sampler type.
    
    This function calculates sampling weights for datapoints using either weighted
    sampling (based on cluster size, molecule types, etc.) or uniform sampling.

    Args:
        sampler_type: Type of sampler to use:
            - 'weighted': Calculate weights based on cluster size and molecule composition
            - 'uniform': Use equal weights for all samples
        indices_df: Pandas DataFrame containing the indices (required for both sampler types)
        beta_dict: Dictionary containing beta values for different data types.
            Default: {"chain": 0.5, "interface": 1}
        alpha_dict: Dictionary containing alpha values for different molecule types.
            Default: {"prot": 3, "nuc": 3, "ligand": 1}
        force_recompute_weight: Whether to force recomputation of weights even if
            a 'weights' column already exists in indices_df (default: False)

    Returns:
        Union[pd.Series, list[float]]: Sample weights as either:
            - pd.Series of float32 weights (for weighted sampling)
            - list of uniform weights (for uniform sampling)

    Raises:
        ValueError: If an unknown sampler type is provided (not 'weighted' or 'uniform')
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Create training and testing datasets from configuration.
    
    This function constructs training and testing datasets based on the provided
    configuration, including:
        - Creating individual BaseSingleDataset instances for each dataset
        - Computing sample weights for training datasets
        - Combining training datasets into a WeightedMultiDataset
        - Creating separate test datasets

    Args:
        configs: ConfigDict containing dataset configurations including:
            - configs.data: Data configuration section
            - configs.train_sets: List of training dataset names
            - configs.test_sets: List of testing dataset names
            - configs.data.train_sampler.train_sample_weights: Overall weights for training datasets
            - configs.data[dataset_name]: Configuration for each dataset including:
                - base_info: Basic dataset information (paths, filters, etc.)
                - cropping_configs: Cropping configuration
                - sampler_configs: Sampling configuration (for training)
                - msa, template configurations
        error_dir: Directory where error logs will be saved (optional)

    Returns:
        tuple: Tuple containing two elements:
            - train_dataset: WeightedMultiDataset combining all training datasets with
                their respective sampling weights
            - test_datasets: Dictionary mapping test dataset names to BaseSingleDataset
                instances {test_name: test_dataset}
    """

    @staticmethod
    @register_license('odesign2025')
    def _get_dataset_param(data_config, dataset_name: str, stage: str):
        """
        Extract and organize dataset parameters from configuration.
        
        This internal helper function extracts dataset-specific parameters from the
        configuration and organizes them into a dictionary suitable for initializing
        a BaseSingleDataset.

        Args:
            data_config: Data configuration section from configs.data
            dataset_name: Name of the dataset (e.g., 'pdb', 'cameo')
            stage: Stage of the dataset ('train' or 'test')

        Returns:
            dict: Dictionary of parameters for BaseSingleDataset initialization including:
                - name: Dataset name
                - base_info fields: mmcif_dir, indices_fpath, etc.
                - cropping_configs: Cropping configuration
                - error_dir: Error log directory
                - msa_featurizer: MSA featurizer instance
                - template_featurizer: Template featurizer instance (None currently)
                - lig_atom_rename, shuffle_mols, shuffle_sym_ids: Boolean flags
                - data_condition, mask_type: Sets for conditioning and masking
                - mask_method: Masking method string
                - return_atom_token_array: Whether to return arrays
                - use_hotspot_residue: Whether to use hotspot features
                - limits: Data limit
                - ref_pos_augment: Whether to augment reference positions
        """
        config_dict = data_config[dataset_name].to_dict()
        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "template_featurizer": None,
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "data_condition": set(config_dict.get("data_condition", [])),
            "mask_type": set(config_dict.get("mask_type", [])),
            "mask_method": config_dict.get("mask_method", ""),
            "return_atom_token_array": config_dict.get("return_atom_token_array", False),
            "use_hotspot_residue": config_dict.get("use_hotspot_residue", False),
            "limits": config_dict.get("limits", -1),
            "ref_pos_augment": config_dict.get("ref_pos_augment", True),
        }

    data_config = configs.data
    logger.info(f"Using train sets {configs.train_sets}")
    assert len(configs.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in configs.train_sets:
        dataset_param = _get_dataset_param(
            data_config, dataset_name=train_name, stage="train"
        )
        train_dataset = BaseSingleDataset(**dataset_param)
        train_datasets.append(train_dataset)
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=configs.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )

    test_datasets = {}
    test_sets = configs.test_sets
    for test_name in test_sets:
        dataset_param = _get_dataset_param(
            data_config, dataset_name=test_name, stage="test"
        )
        test_dataset = BaseSingleDataset(**dataset_param)
        test_datasets[test_name] = test_dataset
    return train_dataset, test_datasets


class InferenceDataset(Dataset):
    """
    Dataset class for inference/prediction tasks.
    
    This class processes input samples from a JSON file for model inference,
    converting them into features without requiring ground truth labels.
    
    Attributes:
        input_json_path: Path to input JSON file containing sample definitions
        dump_dir: Directory for dumping intermediate results
        use_msa: Whether to use MSA features
        data_condition: Set of data conditioning types
        inputs: List of sample dictionaries loaded from JSON
    """
    
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        data_condition: set,
        use_msa: bool = True,
    ) -> None:
        """
        Initialize the InferenceDataset.

        Args:
            input_json_path: Path to JSON file containing input sample definitions.
                Each sample should contain sequence, entity information, etc.
            dump_dir: Directory for dumping intermediate results and outputs
            data_condition: Set of data conditioning types (e.g., {'protein', 'ligand'})
            use_msa: Whether to use MSA features during inference (default: True)

        Returns:
            None
        """
        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.data_condition = data_condition

        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

    def process_one(
        self,
        single_sample_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - OFeatureData object.
                - OLabelData object.
                - AtomArray object.
                - TokenArray object.
        """
        sample2feat = SampleDictToFeatures(
            single_sample_dict=single_sample_dict,
            data_condition=self.data_condition,
            use_msa=self.use_msa
        )
        feature_data, label_data, atom_array, token_array = sample2feat.get_feature_and_label()    

        data = {
            "feature_data": feature_data,
            "label_data": label_data,
            "atom_array": atom_array,
            "token_array": token_array,
            "entity_poly_type": sample2feat.entity_poly_type,
        }

        data.update(
            {
                "N_asym": torch.tensor([feature_data.num_asym]),
                "N_token": torch.tensor([feature_data.num_token]),
                "N_atom": torch.tensor([feature_data.num_atom]),
                "N_msa": torch.tensor([feature_data.num_msa]),
            }
        )

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feature_data[f"is_{mol_type}"]
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feature_data.atom_to_token_idx[mol_type_mask]))
            data[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            data[f"N_{abbr_type}_token"] = torch.tensor([n_token])
        
        return data

    def __len__(self) -> int:
        """
        Return the number of samples in the inference dataset.

        Returns:
            int: Number of input samples
        """
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        """
        Retrieve and process an inference sample by index.
        
        This method loads a sample from the input JSON, converts it to features,
        and handles any errors that occur during processing.

        Args:
            index: Index of the sample to retrieve

        Returns:
            tuple: Tuple containing two elements:
                - data: Dictionary containing:
                    - 'feature_data': OFeatureData object with input features
                    - 'label_data': OLabelData object (may be empty for inference)
                    - 'atom_array': AtomArray object
                    - 'token_array': TokenArray object
                    - 'entity_poly_type': Entity polymer type information
                    - 'N_asym', 'N_token', 'N_atom', 'N_msa': Dimension counts
                    - 'N_{mol_type}_atom', 'N_{mol_type}_token': Per-molecule-type counts
                    - 'sample_name': Name of the sample
                    - 'sample_index': Index of the sample
                - error_message: String containing error message if processing failed,
                    empty string if successful
        """
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data = {}
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, error_message