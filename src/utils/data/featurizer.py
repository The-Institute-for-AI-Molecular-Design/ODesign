import copy
import logging
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
from biotite.structure import Atom, AtomArray, get_residue_starts, BondType
from sklearn.neighbors import KDTree

from src.utils.license_register import register_license
from src.utils.data.ccd import get_ccd_ref_info
from src.utils.data.constants import (
    BOND_TYPE,
    BACKBONE_ATOM_NAMES, APPEND_NA_BACKBONE_ATOM_NAMES,
    PROTEIN_REP_RESIDUE, RNA_REP_RESIDUE, DNA_REP_RESIDUE,
    STD_RESIDUES, STD_RESIDUES_WITH_GAP,
    get_all_elems,
)
from src.utils.data.parser import AddAtomArrayAnnot
from src.utils.data.tokenizer import Token, TokenArray
from src.utils.data.misc import (
    get_atom_level_token_mask,
    get_ligand_polymer_bond_mask,
    exponential_decay,
)
from src.utils.data.geometry import angle_3p, random_transform
from src.utils.data.constraint_featurizer import DistConditionFeaturizer
from scipy.spatial.distance import cdist
from src.utils.model.torch_utils import dict_to_tensor

logger = logging.getLogger(__name__)


class Featurizer(object):
    """
    Main featurizer class for converting molecular structures into model-ready features.
    
    This class processes cropped token and atom arrays to generate various features required
    for the AlphaFold3-style model, including:
        - Token-level features (residue type, entity IDs, etc.)
        - Atom-level features (reference positions, elements, charges, etc.)
        - Structural features (bonds, frames, masks, etc.)
        - MSA and template features
        - Training labels
    
    Attributes:
        cropped_token_array: TokenArray after spatial cropping.
        cropped_atom_array: AtomArray after spatial cropping.
        ref_pos_augment: Whether to apply random rotation/translation on reference positions.
        lig_atom_rename: Whether to rename ligand atom names to avoid information leakage.
        data_condition: Data conditioning mode for partial structure generation.
        is_distillation: Whether in distillation mode (affects resolution features).
        template_features: Pre-computed template features dictionary.
        msa_features: Pre-computed MSA features dictionary.
        mask_method: Masking method for MSA features ('entity' or other).
        use_hotspot_residue: Whether to use hotspot residue identification.
        inference_mode: Whether in inference mode (affects feature generation).
    """
    
    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
        data_condition: bool = False,
        is_distillation: bool = False,
        template_features: dict = {},
        msa_features: dict = {},
        mask_method: str = 'entity',
        use_hotspot_residue: bool = True,
        inference_mode: bool = False,
    ) -> None:
        """
        Initializes the Featurizer with cropped molecular data and configuration options.
        
        Args:
            cropped_token_array: TokenArray object after spatial cropping.
            cropped_atom_array: AtomArray object after spatial cropping.
            ref_pos_augment: Whether to apply random rotation and translation to reference
                positions for data augmentation. Defaults to True.
            lig_atom_rename: Whether to rename ligand atom names to prevent information
                leakage during training. Defaults to False.
            data_condition: Whether to use data conditioning for partial structure generation.
                Can be False or a set containing 'data', 'diffusion', 'constraint_distogram'.
                Defaults to False.
            is_distillation: Whether in distillation mode, which affects resolution features.
                Defaults to False.
            template_features: Pre-computed template features dictionary. Defaults to {}.
            msa_features: Pre-computed MSA features dictionary. Defaults to {}.
            mask_method: Method for masking MSA features. 'entity' masks by entity,
                other values mask all tokens. Defaults to 'entity'.
            use_hotspot_residue: Whether to identify and use hotspot residues for
                interface prediction. Defaults to True.
            inference_mode: Whether in inference mode, which affects feature generation
                and masking strategies. Defaults to False.
        
        Returns:
            None
        """
        self.cropped_token_array = cropped_token_array

        self.cropped_atom_array = cropped_atom_array
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename
        self.data_condition = data_condition
        self.is_distillation = is_distillation
        self.template_features = template_features
        self.msa_features = msa_features
        self.mask_method = mask_method
        self.use_hotspot_residue = use_hotspot_residue
        self.inference_mode = inference_mode

    @staticmethod
    @register_license('bytedance2024')
    def encoder(
        encode_def_dict_or_list: Optional[Union[dict, list[str]]], input_list: list[str]
    ) -> torch.Tensor:
        """
        Encodes a list of input values into one-hot binary format using an encoding definition.
        
        This method creates a one-hot encoded tensor where each input value is mapped to a
        binary vector according to the encoding definition (dict or list).
        
        Args:
            encode_def_dict_or_list: Encoding definition that maps values to indices.
                Can be either:
                    - dict: Maps values (keys) to indices (values), e.g., {'A': 0, 'B': 1}
                    - list: Values in order, indices are implicit, e.g., ['A', 'B']
            input_list: List of input values to be one-hot encoded.
        
        Returns:
            torch.Tensor: One-hot encoded tensor.
                Shape: [len(input_list), num_categories]
                
        Raises:
            TypeError: If encode_def_dict_or_list is neither a list nor a dict.
            AssertionError: If dict indices are not continuous starting from 0.
        """
        num_keys = len(encode_def_dict_or_list)
        if isinstance(encode_def_dict_or_list, dict):
            items = encode_def_dict_or_list.items()
            assert (
                num_keys == max(encode_def_dict_or_list.values()) + 1
            ), "Do not use discontinuous number, which might causing potential bugs in the code"
        elif isinstance(encode_def_dict_or_list, list):
            items = ((key, idx) for idx, key in enumerate(encode_def_dict_or_list))
        else:
            raise TypeError(
                "encode_def_dict_or_list must be a list or dict, "
                f"but got {type(encode_def_dict_or_list)}"
            )
        onehot_dict = {
            key: [int(i == idx) for i in range(num_keys)] for key, idx in items
        }
        onehot_encoded_data = [onehot_dict[item] for item in input_list]
        onehot_tensor = torch.Tensor(onehot_encoded_data)
        return onehot_tensor

    @staticmethod
    @register_license('bytedance2024')
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        One-hot encodes residue types according to AlphaFold3 SI Table 5.
        
        Encodes sequences with 32 possible values:
            - 20 standard amino acids + 1 unknown amino acid
            - 4 RNA nucleotides + 1 unknown RNA nucleotide
            - 4 DNA nucleotides + 1 unknown DNA nucleotide
            - 1 gap character
        
        Ligands are represented as "unknown amino acid" (UNK).
        
        Reference: AlphaFold3 SI Table 5 "restype"
        
        Args:
            restype_list: List of residue type strings (e.g., ['ALA', 'GLY', 'UNK']).
                Ligand residues should be represented as "UNK" in the input list.
        
        Returns:
            torch.Tensor: One-hot encoded residue types.
                Shape: [len(restype_list), 32]
        """
        return Featurizer.encoder(STD_RESIDUES_WITH_GAP, restype_list)

    @staticmethod
    @register_license('bytedance2024')
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        One-hot encodes element symbols according to AlphaFold3 SI Table 5.
        
        Encodes element atomic numbers for atoms in the reference conformer,
        supporting elements up to atomic number 128.
        
        Reference: AlphaFold3 SI Table 5 "ref_element"
        
        Args:
            elem_list: List of element symbol strings (e.g., ['C', 'N', 'O']).
        
        Returns:
            torch.Tensor: One-hot encoded element types.
                Shape: [len(elem_list), 128]
        """
        return Featurizer.encoder(get_all_elems(), elem_list)

    @staticmethod
    @register_license('bytedance2024')
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        One-hot encodes atom name characters according to AlphaFold3 SI Table 5.
        
        Encodes unique atom names in the reference conformer character by character.
        Each character is encoded as ord(c) - 32, and atom names are padded to length 4.
        Supports 64 possible character values (ASCII 32-95: printable characters).
        
        Reference: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        
        Args:
            atom_names: List of atom name strings (e.g., ['CA', 'N', 'C']).
                Names shorter than 4 characters are right-padded with spaces.
        
        Returns:
            torch.Tensor: Character-encoded atom names.
                Shape: [len(atom_names), 4, 64]
                Each atom name is represented by 4 characters, each one-hot encoded.
        """
        onehot_dict = {}
        for index, key in enumerate(range(64)):
            onehot = [0] * 64
            onehot[index] = 1
            onehot_dict[key] = onehot
        # [N_atom, 4, 64]
        mol_encode = []
        for atom_name in atom_names:
            # [4, 64]
            atom_encode = []
            for name_str in atom_name.ljust(4):
                atom_encode.append(onehot_dict[ord(name_str) - 32])
            mol_encode.append(atom_encode)
        onehot_tensor = torch.Tensor(mol_encode)
        return onehot_tensor

    @staticmethod
    @register_license('bytedance2024')
    def get_prot_nuc_frame(token: Token, centre_atom: Atom) -> tuple[int, list[int]]:
        """
        Constructs coordinate frame for protein/DNA/RNA tokens using backbone atoms.
        
        For standard polymers, frames are constructed from three backbone atoms:
            - Proteins: [N, CA, C] atoms
            - DNA/RNA: [C1', C3', C4'] atoms
        
        Reference: AlphaFold3 SI Chapter 4.3.2
        
        Args:
            token: Token object containing atom information.
            centre_atom: Biotite Atom object representing the token's center atom.
        
        Returns:
            tuple[int, list[int]]: A tuple containing:
                - has_frame: 1 if the token has a valid frame, 0 otherwise
                - frame_atom_index: List of 3 atom indices [a, b, c] used to construct
                    the frame, or [-1, -1, -1] if frame is invalid
        """
        if centre_atom.mol_type == "protein":
            # For protein
            abc_atom_name = ["N", "CA", "C"]
        else:
            # For DNA and RNA
            abc_atom_name = [r"C1'", r"C3'", r"C4'"]

        idx_in_atom_indices = []
        for i in abc_atom_name:
            if centre_atom.mol_type == "protein" and "N" not in token.atom_names:
                return 0, [-1, -1, -1]
            elif centre_atom.mol_type != "protein" and "C1'" not in token.atom_names:
                return 0, [-1, -1, -1]
            idx_in_atom_indices.append(token.atom_names.index(i))
        # Protein/DNA/RNA always has frame
        has_frame = 1
        frame_atom_index = [token.atom_indices[i] for i in idx_in_atom_indices]
        return has_frame, frame_atom_index

    @staticmethod
    @register_license('bytedance2024')
    def get_lig_frame(
        token: Token,
        centre_atom: Atom,
        lig_res_ref_conf_kdtree: dict[str, tuple[KDTree, list[int]]],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[int, list[int]]:
        """
        Constructs coordinate frame for ligand tokens using reference conformer geometry.
        
        For ligands, the frame is constructed from the reference conformer by finding:
            - b: The token's center atom
            - a: The closest atom to b in the reference conformer
            - c: The second closest atom to b in the reference conformer
        
        Frame validity is checked for:
            - Sufficient atoms (>= 3)
            - Valid reference positions
            - Non-colinear geometry (angle not in [0-25° or 155-180°])
        
        Reference: AlphaFold3 SI Chapter 4.3.2
        
        Args:
            token: Token object containing atom information.
            centre_atom: Biotite Atom object representing the token's center atom.
            lig_res_ref_conf_kdtree: Dictionary mapping ref_space_uid to (KDTree, atom_indices).
                The KDTree is built from reference conformer positions.
            ref_pos: Reference conformer atom positions.
                Shape: [N_atom, 3]
            ref_mask: Mask indicating valid reference positions.
                Shape: [N_atom]
        
        Returns:
            tuple[int, list[int]]: A tuple containing:
                - has_frame: 1 if the token has a valid frame, 0 otherwise
                - frame_atom_index: List of 3 atom indices [a, b, c] used to construct
                    the frame, or [-1, b, -1] if frame is invalid
        """
        kdtree, atom_ids = lig_res_ref_conf_kdtree[centre_atom.ref_space_uid]
        b_ref_pos = ref_pos[token.centre_atom_index]
        b_idx = token.centre_atom_index
        if kdtree is None:
            # Atom num < 3
            frame_atom_index = [-1, b_idx, -1]
            has_frame = 0
        else:
            _dist, ind = kdtree.query([b_ref_pos], k=3)
            a_idx, c_idx = atom_ids[ind[0][1]], atom_ids[ind[0][2]]
            frame_atom_index = [a_idx, b_idx, c_idx]

            # Check if reference confomrer vaild
            has_frame = all([ref_mask[idx] for idx in frame_atom_index])

            # Colinear check
            if has_frame:
                vec1 = ref_pos[frame_atom_index[1]] - ref_pos[frame_atom_index[0]]
                vec2 = ref_pos[frame_atom_index[2]] - ref_pos[frame_atom_index[1]]
                # ref_pos can be all zeros, in which case has_frame=0
                is_zero_norm = np.isclose(
                    np.linalg.norm(vec1, axis=-1), 0
                ) or np.isclose(np.linalg.norm(vec2, axis=-1), 0)
                if is_zero_norm:
                    has_frame = 0
                else:
                    theta_degrees = angle_3p(
                        *[ref_pos[idx] for idx in frame_atom_index]
                    )
                    is_colinear = theta_degrees <= 25 or theta_degrees >= 155
                    if is_colinear:
                        has_frame = 0
        return has_frame, frame_atom_index

    @staticmethod
    @register_license('odesign2025')
    def get_token_frame(
        token_array: TokenArray,
        atom_array: AtomArray,
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> TokenArray:
        """
        Constructs coordinate frames for all tokens in the token array.
        
        The atoms (a_i, b_i, c_i) used to construct token i's frame depend on chain type:
            - Protein tokens: Use backbone atoms (N, Cα, C)
            - DNA/RNA tokens: Use sugar atoms (C1', C3', C4')
            - Other tokens (ligands, glycans, ions): Use closest atoms from reference conformer
                - b_i: The token atom itself
                - a_i: Closest atom to b_i
                - c_i: Second closest atom to b_i
        
        Frame validity requirements:
            - Three atoms must exist
            - Atoms must not be close to colinear (< 25° or > 155°)
            - Valid reference positions must be available
        
        Note: Frames are constructed from the reference conformer, not the native structure.
        
        Reference: AlphaFold3 SI Chapter 4.3.2
        
        Args:
            token_array: Array of Token objects.
            atom_array: Biotite AtomArray containing all atoms.
            ref_pos: Reference conformer atom positions.
                Shape: [N_atom, 3]
            ref_mask: Mask indicating valid reference positions.
                Shape: [N_atom]
        
        Returns:
            TokenArray: Updated token array with frame annotations:
                - has_frame: 1 if the token has a valid frame, 0 otherwise
                - frame_atom_index: List of 3 atom indices [a, b, c] for each token
        """
        token_array_w_frame = token_array
        atom_level_token_mask = get_atom_level_token_mask(token_array, atom_array)

        # Construct a KDTree for queries to avoid redundant distance calculations
        lig_res_ref_conf_kdtree = {}
        # Ligand and non-standard residues need to use ref to identify frames
        lig_atom_array = atom_array[
            (atom_array.mol_type == "ligand")
            | (~np.isin(atom_array.res_name, list(STD_RESIDUES.keys())))
            | atom_level_token_mask
        ]
        for ref_space_uid in np.unique(lig_atom_array.ref_space_uid):
            # The ref_space_uid is the unique identifier ID for each residue.
            atom_ids = np.where(atom_array.ref_space_uid == ref_space_uid)[0]
            if len(atom_ids) >= 3:
                kdtree = KDTree(ref_pos[atom_ids], metric="euclidean")
            else:
                # Invalid frame
                kdtree = None
            lig_res_ref_conf_kdtree[ref_space_uid] = (kdtree, atom_ids)

        has_frame = []
        for token in token_array_w_frame:
            centre_atom = atom_array[token.centre_atom_index]
            if (
                centre_atom.mol_type != "ligand"
                and centre_atom.res_name in STD_RESIDUES
                and len(token.atom_indices) > 1
            ):
                has_frame, frame_atom_index = Featurizer.get_prot_nuc_frame(
                    token, centre_atom
                )

            else:
                has_frame, frame_atom_index = Featurizer.get_lig_frame(
                    token, centre_atom, lig_res_ref_conf_kdtree, ref_pos, ref_mask
                )

            token.has_frame = has_frame
            token.frame_atom_index = frame_atom_index
        return token_array_w_frame

    @register_license('odesign2025')
    def get_token_features(self) -> dict[str, torch.Tensor]:
        """
        Generates token-level features for model input.
        
        Extracts and encodes features at the token level, including residue types,
        identifiers, and indices. Applies data conditioning masks if specified.
        
        Reference: AlphaFold3 SI Chapter 2.8
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of token features:
                - token_index: Sequential token indices [N_token]
                - residue_index: Residue IDs from structure [N_token]
                - asym_id: Asymmetric unit IDs [N_token]
                - entity_id: Entity IDs [N_token]
                - sym_id: Symmetry IDs [N_token]
                - restype: One-hot encoded residue types [N_token, 32]
        """
        token_features = {}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        restype = centre_atoms.cano_seq_resname
        
        if self.data_condition & set(['data']):
            token_mask = ~(centre_atoms.get_annotation('condition_token_mask'))
            restype[token_mask & (centre_atoms.mol_type == 'protein')] = '-P'
            restype[token_mask & (centre_atoms.mol_type == 'rna')] = '-N'
            restype[token_mask & (centre_atoms.mol_type == 'dna')] = '-N'
            restype[token_mask & (centre_atoms.mol_type == 'ligand')] = '-L'

        restype_onehot = self.restype_onehot_encoded(restype)

        token_features["token_index"] = torch.arange(
            0, len(self.cropped_token_array)
        )
        token_features["residue_index"] = torch.Tensor(
            centre_atoms.res_id.astype(int)
        ).long()
        token_features["asym_id"] = torch.Tensor(
            centre_atoms.asym_id_int
        ).long()
        token_features["entity_id"] = torch.Tensor(
            centre_atoms.entity_id_int
        ).long()
        token_features["sym_id"] = torch.Tensor(
            centre_atoms.sym_id_int
        ).long()
        token_features["restype"] = restype_onehot

        return token_features

    @register_license('bytedance2024')
    def get_chain_perm_features(self) -> dict[str, torch.Tensor]:
        """
        Generates chain permutation features for symmetry-aware alignment.
        
        These features use molecular-level identifiers (mol_id, entity_mol_id, mol_atom_index)
        instead of asymmetric unit identifiers (asym_id, entity_id, residue_index) to enable
        proper handling of symmetric chains during multi-chain permutation alignment.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of chain permutation features:
                - mol_id: Molecular chain ID for each atom [N_atom]
                - mol_atom_index: Atom index within its molecule [N_atom]
                - entity_mol_id: Entity-level molecular ID [N_atom]
        """
        chain_perm_features = {}
        chain_perm_features["mol_id"] = torch.Tensor(
            self.cropped_atom_array.mol_id
        ).long()
        chain_perm_features["mol_atom_index"] = torch.Tensor(
            self.cropped_atom_array.mol_atom_index
        ).long()
        chain_perm_features["entity_mol_id"] = torch.Tensor(
            self.cropped_atom_array.entity_mol_id
        ).long()
        return chain_perm_features

    @register_license('odesign2025')
    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Renames ligand atom names to prevent information leakage during training.
        
        Ligand atom names in structures often contain chemical information that could
        leak ground truth data. This method renames ligand atoms systematically:
            - Format: ELEMENT + COUNT (e.g., C1, C2, N1, O1)
            - Non-ligand atoms retain their original names
        
        Returns:
            np.ndarray: Array of renamed atom names.
                Shape: [N_atom]
        """
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        new_atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            res_mol_type = self.cropped_atom_array.mol_type[start]
            if res_mol_type != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.cropped_atom_array.element[start:stop]:
                elem_count[elem] += 1
                new_res_atom_names.append(f"{elem.upper()}{elem_count[elem]}")
            new_atom_names[start:stop] = new_res_atom_names
        return new_atom_names

    @register_license('odesign2025')
    def get_reference_features(self) -> dict[str, torch.Tensor]:
        """
        Generates reference conformer features for model input.
        
        Extracts and processes features from reference conformers, including:
            - Reference atomic positions (with optional augmentation)
            - Element types
            - Atomic charges
            - Atom names
            - Coordinate frames for tokens
        
        For data conditioning mode, applies masking to certain features to prevent
        information leakage during partial structure generation.
        
        Reference: AlphaFold3 SI Chapter 2.8
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of reference features:
                - ref_pos: Reference conformer positions [N_atom, 3]
                - ref_mask: Valid reference position mask [N_atom]
                - ref_element: One-hot encoded elements [N_atom, 128]
                - ref_charge: Atomic charges [N_atom]
                - ref_atom_name_chars: Encoded atom names [N_atom, 4, 64]
                - ref_space_uid: Reference space unique IDs [N_atom]
                - has_frame: Frame validity flags [N_token]
                - frame_atom_index: Frame atom indices [N_token, 3]
        """
        ref_pos = []
        rep_res_info = {
            PROTEIN_REP_RESIDUE: get_ccd_ref_info(PROTEIN_REP_RESIDUE),
            DNA_REP_RESIDUE: get_ccd_ref_info(DNA_REP_RESIDUE),
            RNA_REP_RESIDUE: get_ccd_ref_info(RNA_REP_RESIDUE)
        }
        unique_ref_space_uids = np.unique(
            self.cropped_atom_array.ref_space_uid, return_index=True
        )[1]
        sorted_indices = np.sort(unique_ref_space_uids)
        for ref_space_uid in self.cropped_atom_array.ref_space_uid[sorted_indices]:
            tmp_atom_array = self.cropped_atom_array[
                self.cropped_atom_array.ref_space_uid == ref_space_uid
            ]
            tmp_ref_pos = tmp_atom_array.ref_pos
            res_name = tmp_atom_array.res_name[0]
            if self.data_condition & set(['data']):
                if (~tmp_atom_array.get_annotation('condition_token_mask')).all():
                    if tmp_atom_array.mol_type[0] == 'protein':
                        rep_residue = PROTEIN_REP_RESIDUE
                    elif tmp_atom_array.mol_type[0] == 'dna':
                        rep_residue = DNA_REP_RESIDUE
                    elif tmp_atom_array.mol_type[0] == 'rna':
                        rep_residue = RNA_REP_RESIDUE
                    else:  # ligand
                        rep_residue = res_name
                        tmp_ref_pos = np.zeros_like(tmp_ref_pos)
                        ref_pos.append(tmp_ref_pos)
                        continue
                        
                    if res_name != rep_residue:
                        # replace the backbone atoms with the representative residue
                        rep_residue_info = rep_res_info[rep_residue]
                        backbone_atoms_list = (
                            BACKBONE_ATOM_NAMES + 
                            [APPEND_NA_BACKBONE_ATOM_NAMES.get(res_name, 'Nothing')]
                        )
                        bb_atom_idx = np.where(
                            np.isin(tmp_atom_array.atom_name, backbone_atoms_list)
                        )[0]
                        bb_atom_names = np.char.replace(
                            tmp_atom_array.atom_name[bb_atom_idx], 'N9', 'N1'
                        )
                        rep_atom_idx = [
                            rep_residue_info['atom_map'][bb_atom_name] 
                            for bb_atom_name in bb_atom_names
                        ]
                        tmp_ref_pos[bb_atom_idx] = rep_residue_info['coord'][rep_atom_idx]
               
            res_ref_pos = random_transform(
                tmp_ref_pos,
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )
            ref_pos.append(res_ref_pos)
        ref_pos = np.concatenate(ref_pos)

        ref_features = {}
        ref_features["ref_pos"] = torch.Tensor(ref_pos)
        ref_features["ref_mask"] = torch.Tensor(self.cropped_atom_array.ref_mask).long()
        token_mask = ~self.cropped_atom_array.get_annotation('condition_token_mask')
        element_array = copy.deepcopy(self.cropped_atom_array.element)
        if self.data_condition & set(['data']):
            lig_mask = (token_mask) & (self.cropped_atom_array.mol_type == 'ligand')
            nuc_mask = (
                (token_mask) & 
                (np.isin(self.cropped_atom_array.mol_type, ['dna', 'rna'])) & 
                (np.isin(self.cropped_atom_array.atom_name, ['N1', 'N9']))
            )
            if ref_pos[lig_mask].any():
                print('The reference position of ligand atoms should not be all zeros.')
            element_array[lig_mask] = '-'
        ref_features["ref_element"] = Featurizer.elem_onehot_encoded(
            element_array
        ).long()

        charge_array = copy.deepcopy(self.cropped_atom_array.charge)
        if self.data_condition & set(['data']):
            charge_array[token_mask] = 0
        ref_features["ref_charge"] = torch.Tensor(
            charge_array
        ).long()

        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.cropped_atom_array.atom_name

        atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        if self.data_condition & set(['data']):
            atom_names[lig_mask] = '-'
            atom_names[nuc_mask] = 'N'

        ref_features["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
            atom_names
        ).long()
        ref_features["ref_space_uid"] = torch.Tensor(
            self.cropped_atom_array.ref_space_uid
        ).long()

        token_array_with_frame = self.get_token_frame(
            token_array=self.cropped_token_array,
            atom_array=self.cropped_atom_array,
            ref_pos=ref_features["ref_pos"],
            ref_mask=ref_features["ref_mask"],
        )
        ref_features["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()  # [N_token]
        ref_features["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()  # [N_token, 3]
        return ref_features

    @register_license('odesign2025')
    def get_bond_features(self) -> dict[str, torch.Tensor]:
        """
        Generates bond connectivity features at the token level.
        
        Creates a 2D adjacency matrix indicating bonds between tokens. The bonding
        information is filtered to include only:
            - Polymer-ligand bonds
            - Ligand-ligand bonds
            - Bonds < 2.4 Å during training (for polymer-ligand)
        
        Standard polymer-polymer bonds (within and between standard residues) are excluded
        as they are implicitly known from the sequence.
        
        Reference: AlphaFold3 SI Chapter 2.8
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of bond features:
                - token_bonds: Token-level bond adjacency matrix [N_token, N_token]
                - token_pair_gen_mask: Mask for token pairs to generate [N_token, N_token]
                - token_bond_gen_mask: Mask for bonds to generate [N_token, N_token]
        """
        bond_array = self.cropped_atom_array.bonds.as_array()
        bond_atom_i = bond_array[:, 0]
        bond_atom_j = bond_array[:, 1]
        ref_space_uid = self.cropped_atom_array.ref_space_uid
        if hasattr(self.cropped_atom_array, 'atomized'):
            polymer_mask = (
                np.isin(self.cropped_atom_array.mol_type, ["protein", "dna", "rna"]) & 
                ~self.cropped_atom_array.atomized
            )
        else:
            polymer_mask = np.isin(
                self.cropped_atom_array.mol_type, ["protein", "dna", "rna"]
            )
        std_res_mask = (
            np.isin(self.cropped_atom_array.res_name, list(STD_RESIDUES.keys()))
            & polymer_mask
        )
        unstd_res_mask = ~std_res_mask & polymer_mask
        # The polymer-polymer (std-std, std-unstd, and inter-unstd) bonds 
        # will not be included in token_bonds.
        std_std_bond_mask = std_res_mask[bond_atom_i] & std_res_mask[bond_atom_j]
        std_unstd_bond_mask = (
            std_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) | (std_res_mask[bond_atom_j] & unstd_res_mask[bond_atom_i])
        inter_unstd_bond_mask = (
            unstd_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) & (ref_space_uid[bond_atom_i] != ref_space_uid[bond_atom_j])

        if self.data_condition & set(['data']):
            masked_atom_mask = ~self.cropped_atom_array.get_annotation(
                'condition_token_mask'
            )
            if hasattr(self.cropped_atom_array, 'atomized'):
                atomized_mask = self.cropped_atom_array.get_annotation('atomized')
                masked_bonds = (
                    (masked_atom_mask[bond_atom_i] | masked_atom_mask[bond_atom_j]) & 
                    ~(atomized_mask[bond_atom_i] | atomized_mask[bond_atom_j])
                )
            else:
                masked_bonds = (
                    masked_atom_mask[bond_atom_i] | masked_atom_mask[bond_atom_j]
                )
        else:
            masked_bonds = np.zeros_like(bond_atom_i, dtype=bool)
        
        kept_bonds = bond_array[
            ~(std_std_bond_mask | std_unstd_bond_mask | inter_unstd_bond_mask | masked_bonds)
        ]
        
        # -1 means the atom is not in any token
        atom_idx_to_token_idx = np.zeros(len(self.cropped_atom_array), dtype=int) - 1
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_idx_to_token_idx[atom_idx] = idx
        assert np.all(atom_idx_to_token_idx >= 0), "Some atoms are not in any token"
        num_tokens = len(self.cropped_token_array)
        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        bond_token_i, bond_atom_j = (
            atom_idx_to_token_idx[kept_bonds[:, 0]],
            atom_idx_to_token_idx[kept_bonds[:, 1]],
        )
        for i, j in zip(bond_token_i, bond_atom_j):
            token_adj_matrix[i, j] = 1
            token_adj_matrix[j, i] = 1
        bond_features = {"token_bonds": torch.Tensor(token_adj_matrix)}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]
        if hasattr(centre_atoms, 'atomized'):
            condition_or_atomized = (
                centre_atoms.get_annotation('condition_token_mask') | 
                centre_atoms.get_annotation('atomized')
            )
            condition_mask_tensor = torch.Tensor(condition_or_atomized).bool()
            bond_features["token_pair_gen_mask"] = torch.logical_not(
                condition_mask_tensor[:, None] & condition_mask_tensor[None, :]
            )
        else:
            condition_mask = centre_atoms.get_annotation('condition_token_mask')
            condition_mask_tensor = torch.Tensor(condition_mask).bool()
            bond_features["token_pair_gen_mask"] = torch.logical_not(
                condition_mask_tensor[:, None] & condition_mask_tensor[None, :]
            )

        if hasattr(centre_atoms, 'atomized'):
            ligand_or_atomized = centre_atoms.is_ligand | centre_atoms.atomized
            ligand_mask_tensor = torch.Tensor(ligand_or_atomized).bool()
            is_ligand_pair = (
                ligand_mask_tensor[:, None] & ligand_mask_tensor[None, :]
            )
        else:
            ligand_mask_tensor = torch.Tensor(centre_atoms.is_ligand).bool()
            is_ligand_pair = (
                ligand_mask_tensor[:, None] & ligand_mask_tensor[None, :]
            )

        token_bond_gen_mask = bond_features["token_pair_gen_mask"] * is_ligand_pair
        upper_tri_mask = torch.triu(
            torch.ones_like(token_bond_gen_mask, dtype=torch.bool),
            diagonal=1
        )
        bond_features["token_bond_gen_mask"] = token_bond_gen_mask & upper_tri_mask

        return bond_features

    @register_license('odesign2025')
    def get_identity_features(self) -> dict[str, torch.Tensor]:
        """
        Generates atom identity and type features for model input.
        
        Extracts atom-to-token mappings and molecular type indicators that are
        essential for model processing but not explicitly listed in AlphaFold3 SI.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of identity features:
                - atom_to_token_idx: Mapping from atoms to tokens [N_atom]
                - atom_to_tokatom_idx: Atom index within its token [N_atom]
                - is_protein: Binary flag for protein atoms [N_atom]
                - is_ligand: Binary flag for ligand atoms [N_atom]
                - is_dna: Binary flag for DNA atoms [N_atom]
                - is_rna: Binary flag for RNA atoms [N_atom]
        """
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # Ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = [
            atom_to_token_idx_dict[atom_idx]
            for atom_idx in range(len(self.cropped_atom_array))
        ]

        identity_features = {}
        identity_features["atom_to_token_idx"] = torch.Tensor(atom_to_token_idx).long()
        identity_features["atom_to_tokatom_idx"] = torch.Tensor(
            self.cropped_atom_array.tokatom_idx
        ).long()

        identity_features["is_protein"] = torch.Tensor(
            self.cropped_atom_array.is_protein
        ).long()
        identity_features["is_ligand"] = torch.Tensor(
            self.cropped_atom_array.is_ligand
        ).long()
        identity_features["is_dna"] = torch.Tensor(self.cropped_atom_array.is_dna).long()
        identity_features["is_rna"] = torch.Tensor(self.cropped_atom_array.is_rna).long()

        return identity_features
    
    @register_license('bytedance2024')
    def get_resolution_features(self) -> dict[str, torch.Tensor]:
        """
        Extracts structure resolution information if available.
        
        Returns resolution from the structure metadata, or -1 if unavailable or
        in distillation mode (to prevent model from learning resolution-dependent biases).
        
        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - resolution: Structure resolution in Angstroms, or -1 if unavailable [1]
        """
        if "resolution" in self.cropped_atom_array._annot:
            resolution = torch.Tensor(
                [self.cropped_atom_array.resolution[0]]
            )
        else:
            resolution = torch.Tensor([-1])

        if self.is_distillation:
            resolution = torch.tensor([-1])

        return {"resolution": resolution}

    @staticmethod
    @register_license('odesign2025')
    def get_lig_pocket_mask(
        atom_array: AtomArray, 
        lig_label_asym_id: Union[str, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Identifies ligand binding pocket atoms for evaluation metrics.
        
        Defines the pocket as all heavy atoms within 10 Å of any ligand heavy atom,
        with additional restrictions:
            - Limited to the primary polymer chain (protein with most atoms near ligand)
            - For proteins, further restricted to backbone atoms (C, N, CA)
        
        Primary chain definition varies by context:
            - PoseBusters: Protein chain with most atoms within 10 Å of ligand
            - Bonded ligands: The bonded polymer chain
            - Modified residues: The chain containing the residue (excluding that residue)
        
        Reference: AlphaFold3 Methods.Metrics
        
        Args:
            atom_array: Biotite AtomArray containing all atoms in the complex.
            lig_label_asym_id: Label asymmetric ID(s) of the ligand(s) of interest.
                Can be a single string or list of strings.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - ligand_mask_by_pockets: Binary mask for ligand atoms [N_ligands, N_atom]
                - pocket_mask_by_pockets: Binary mask for pocket atoms [N_ligands, N_atom]
        """

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        prot_backbone = (
            atom_array.is_protein & np.isin(atom_array.atom_name, ["C", "N", "CA"])
        ).astype(bool)

        kdtree = KDTree(atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(
                lig_label_asym_id, atom_array.label_asym_id
            ), f"{lig_label_asym_id} is not in the label_asym_id of the cropped atom array."

            ligand_mask = atom_array.label_asym_id == lig_label_asym_id
            lig_pos = atom_array.coord[ligand_mask & atom_array.is_resolved]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                np.concatenate(kdtree.query_radius(lig_pos, 10.0))
            )
            near_atoms = [
                (
                    True
                    if ((i in near_atom_indices) and atom_array.is_resolved[i])
                    else False
                )
                for i in range(len(atom_array))
            ]

            # Get primary chain (protein backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & prot_backbone
            primary_chain_candidates_atoms = atom_array[primary_chain_candidates]

            max_atom = 0
            primary_chain_asym_id_int = None
            for asym_id_int in np.unique(primary_chain_candidates_atoms.asym_id_int):
                n_atoms = np.sum(
                    primary_chain_candidates_atoms.asym_id_int == asym_id_int
                )
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_asym_id_int = asym_id_int
            assert (
                primary_chain_asym_id_int is not None
            ), f"No primary chain found for ligand ({lig_label_asym_id=})."

            pocket_mask = primary_chain_candidates & (
                atom_array.asym_id_int == primary_chain_asym_id_int
            )
            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = torch.Tensor(
            np.array(ligand_mask_list).astype(int)
        ).long()
        pocket_mask_by_pockets = torch.Tensor(
            np.array(pocket_mask_list).astype(int)
        ).long()
        return ligand_mask_by_pockets, pocket_mask_by_pockets

    @register_license('odesign2025')
    def get_mask_features(self) -> dict[str, torch.Tensor]:
        """
        Generates various masking features for training and evaluation.
        
        Creates masks that control which atoms/tokens are used for different purposes:
            - PAE (Predicted Aligned Error) evaluation
            - Modified residue identification
            - Conditional generation (which atoms are fixed/conditioned)
            - Distogram and pLDDT metric computation
            - Bond prediction masking
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of mask features:
                - pae_rep_atom_mask: Mask for PAE representative atoms [N_atom]
                - modified_res_mask: Mask for modified residues [N_atom]
                - is_condition_atom: Mask for conditioned atoms (fixed during generation) [N_atom]
                - distogram_rep_atom_mask: Mask for distogram representative atoms [N_atom]
                - plddt_m_rep_atom_mask: Mask for pLDDT representative atoms [N_atom]
                - bond_mask: Mask for polymer-ligand bonds [N_atom, N_atom]
        """
        mask_features = {}

        mask_features["pae_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.centre_atom_mask
        ).bool()

        mask_features["modified_res_mask"] = torch.Tensor(
            self.cropped_atom_array.modified_res_mask
        ).bool()

        if self.data_condition & set(['diffusion']):
            condition_token_mask = self.cropped_atom_array.get_annotation(
                "condition_token_mask"
            )
            mask_features['is_condition_atom'] = torch.tensor(
                condition_token_mask
            ).bool()
        else:
            mask_features['is_condition_atom'] = torch.tensor(
                np.zeros(len(self.cropped_atom_array))
            ).bool()

        plddt_m_rep_atom_mask = torch.Tensor(
            self.cropped_atom_array.plddt_m_rep_atom_mask
        ).long()  # [N_atom]

        distogram_rep_atom_mask = torch.Tensor(
            self.cropped_atom_array.distogram_rep_atom_mask
        ).long()  # [N_atom]

        if (self.data_condition & set(['data'])) and not self.inference_mode:
            token_mask = ~self.cropped_atom_array.get_annotation(
                'condition_token_mask'
            )
            distogram_rep_atom_mask[token_mask] = plddt_m_rep_atom_mask[token_mask]
            distogram_rep_atom_mask[
                self.cropped_atom_array.mol_type == 'ligand'
            ] = 1
            if hasattr(self.cropped_atom_array, 'atomized'):
                distogram_rep_atom_mask[
                    self.cropped_atom_array.atomized == True
                ] = 1

        if distogram_rep_atom_mask.sum() != len(self.cropped_token_array):
            raise ValueError(
                f"distogram_rep_atom_mask sum: {distogram_rep_atom_mask.sum()}, "
                f"len(cropped_token_array): {len(self.cropped_token_array)}"
            )

        mask_features["distogram_rep_atom_mask"] = distogram_rep_atom_mask
        mask_features["plddt_m_rep_atom_mask"] = plddt_m_rep_atom_mask

        lig_polymer_bonds = get_ligand_polymer_bond_mask(self.cropped_atom_array)
        num_atoms = len(self.cropped_atom_array)
        bond_mask_mat = np.zeros((num_atoms, num_atoms))
        condition_token_mask = self.cropped_atom_array.get_annotation(
            'condition_token_mask'
        )
        for i, j, _ in lig_polymer_bonds:
            if condition_token_mask[i] and condition_token_mask[j]:
                bond_mask_mat[i, j] = 1
                bond_mask_mat[j, i] = 1
        mask_features["bond_mask"] = torch.Tensor(
            bond_mask_mat
        ).bool()

        return mask_features

    @register_license('odesign2025')
    def get_all_input_features(self) -> dict[str, torch.Tensor]:
        """
        Aggregates all input features required for model inference or training.
        
        Combines features from multiple sources:
            - Token features (residue types, IDs)
            - Bond connectivity
            - Reference conformer information
            - Identity features (atom types, mappings)
            - Resolution metadata
            - Permutation features (for symmetry handling)
            - Mask features (for evaluation and conditioning)
            - Optional constraint features (for guided generation)
            - Optional cyclic features (for inference)
        
        Returns:
            dict[str, torch.Tensor]: Comprehensive dictionary of all input features
                required by the model.
        """
        features = {}
        token_features = self.get_token_features()
        features.update(token_features)

        bond_features = self.get_bond_features()
        features.update(bond_features)

        reference_features = self.get_reference_features()
        features.update(reference_features)

        identity_features = self.get_identity_features()
        features.update(identity_features)

        resolution_features = self.get_resolution_features()
        features.update(resolution_features)

        chain_perm_features = self.get_chain_perm_features()
        features.update(chain_perm_features)

        atom_perm_features = self.get_atom_perm_features()
        features.update(atom_perm_features)

        mask_features = self.get_mask_features()
        features.update(mask_features)

        if self.data_condition & set(['constraint_distogram']):
            contact_featurizer = DistConditionFeaturizer(
                token_array=self.cropped_token_array,
                atom_array=self.cropped_atom_array,
                pad_value=0,
                generator=None,
            )
            features["constraint_feature"] = (
                contact_featurizer.generate_spec_constraint_distogram()
            )

        # if self.inference_mode:
        #     cyclic_features = self.get_cyclic_features()
        #     features.update(cyclic_features)
        
        return features

    @register_license('odesign2025')
    def get_labels(self) -> dict[str, torch.Tensor]:
        """
        Generates ground truth labels for model training.
        
        Extracts target values from the structure that the model will learn to predict:
            - Atomic coordinates (3D positions)
            - Coordinate validity masks
            - Canonical sequence residue names (for data conditioning)
            - Bond types between tokens (for ligand bond prediction)
            - Ligand bond masks
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of training labels:
                - coordinate: Ground truth atom positions [N_atom, 3]
                - coordinate_mask: Valid coordinate mask [N_atom]
                - cano_seq_resname_label: Canonical residue name labels [N_token]
                    (only in data conditioning mode)
                - token_bond_type_label: Bond types between tokens [N_token, N_token]
                - ligand_bond_mask: Binary mask for ligand bonds [N_atom, N_atom]
        """

        labels = {}

        labels["coordinate"] = torch.Tensor(
            self.cropped_atom_array.coord
        )  # [N_atom, 3]
        labels["coordinate_mask"] = torch.Tensor(
            self.cropped_atom_array.is_resolved.astype(int)
        ).long()  # [N_atom]
        assert labels["coordinate_mask"].any(), "coordinate_mask contains all 0"

        if self.data_condition & set(['data']):
            centre_atom_indices = self.cropped_token_array.get_annotation(
                'centre_atom_index'
            )
            cano_seq_resnames = self.cropped_atom_array[
                centre_atom_indices
            ].cano_seq_resname
            labels["cano_seq_resname_label"] = torch.Tensor([
                STD_RESIDUES.get(cano_seq_resname, 20) 
                for cano_seq_resname in cano_seq_resnames
            ]).long()

        bond_array = self.cropped_atom_array.bonds.as_array()
        bond_atom_i = bond_array[:, 0]
        bond_atom_j = bond_array[:, 1]
        is_ligand = self.cropped_atom_array.is_ligand.astype(bool)
        lig_lig_bond_mask = is_ligand[bond_atom_i] & is_ligand[bond_atom_j]
        lig_lig_bonds = bond_array[lig_lig_bond_mask]

        # Initialize atom to token mapping (-1 means atom is not in any token)
        atom_idx_to_token_idx = (
            np.zeros(len(self.cropped_atom_array), dtype=int) - 1
        )
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_idx_to_token_idx[atom_idx] = idx
        assert np.all(atom_idx_to_token_idx >= 0), (
            "Some atoms are not in any token"
        )
        num_tokens = len(self.cropped_token_array)
        token_bond_type_label = torch.full(
            (num_tokens, num_tokens), BOND_TYPE["NONE"]
        ).long()
        bond_token_i, bond_token_j, bond_types = (
            atom_idx_to_token_idx[lig_lig_bonds[:, 0]],
            atom_idx_to_token_idx[lig_lig_bonds[:, 1]],
            lig_lig_bonds[:, 2],
        )
        for i, j, bond_type in zip(bond_token_i, bond_token_j, bond_types):
            if BondType(bond_type).name == "NONE":
                token_bond_type_label[i, j] = BOND_TYPE[BondType(bond_type).name]
                token_bond_type_label[j, i] = BOND_TYPE[BondType(bond_type).name]
            else:
                token_bond_type_label[i, j] = 1
                token_bond_type_label[j, i] = 1

        labels["token_bond_type_label"] = token_bond_type_label

        num_atoms = len(self.cropped_atom_array)
        ligand_bond_mask = torch.full((num_atoms, num_atoms), 0).long()
        ligand_bond_atom_i = lig_lig_bonds[:, 0]
        ligand_bond_atom_j = lig_lig_bonds[:, 1]
        for i, j in zip(ligand_bond_atom_i, ligand_bond_atom_j):
            ligand_bond_mask[i, j] = 1
            ligand_bond_mask[j, i] = 1
        labels['ligand_bond_mask'] = ligand_bond_mask.to(torch.bool)

        return labels

    @register_license('bytedance2024')
    def get_atom_perm_features(self) -> dict[str, list[list[int]]]:
        """
        Generates atom permutation information for symmetry handling.
        
        Identifies valid atom permutations within residues based on chemical symmetry.
        Atoms connected to different residues (via inter-residue bonds) are fixed and
        excluded from permutation groups to maintain connectivity constraints.
        
        This is crucial for:
            - Proper loss computation with symmetric groups (e.g., terminal methyl groups)
            - Avoiding ambiguity in coordinate prediction for equivalent atoms
            - Ensuring structural validity across permutations
        
        Returns:
            dict[str, list[list[int]]]: Dictionary containing:
                - atom_perm_list: List of valid permutation indices for each atom.
                    Shape: [[perm_idx_1, perm_idx_2, ...] for each atom]
                    Each sub-list contains the permutation group indices for that atom.
        """
        self.cropped_atom_array = AddAtomArrayAnnot.add_res_perm(self.cropped_atom_array)

        atom_perm_list = []
        for i in self.cropped_atom_array.res_perm:
            # Decode list[str] -> list[list[int]]
            atom_perm_list.append([int(j) for j in i.split("_")])

        # Atoms connected to different residue are fixed.
        # Bonds array: [[atom_idx_i, atom_idx_j, bond_type]]
        idx_i = self.cropped_atom_array.bonds._bonds[:, 0]
        idx_j = self.cropped_atom_array.bonds._bonds[:, 1]
        diff_mask = (
            self.cropped_atom_array.ref_space_uid[idx_i]
            != self.cropped_atom_array.ref_space_uid[idx_j]
        )
        inter_residue_bonds = self.cropped_atom_array.bonds._bonds[diff_mask]
        fixed_atom_mask = np.isin(
            np.arange(len(self.cropped_atom_array)),
            np.unique(inter_residue_bonds[:, :2]),
        )

        # Get fixed atom permutation for each residue.
        fixed_atom_perm_list = []
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        for r_start, r_stop in zip(res_starts[:-1], res_starts[1:]):
            atom_res_perm = np.array(
                atom_perm_list[r_start:r_stop]
            )  # [N_res_atoms, N_res_perm]
            res_fixed_atom_mask = fixed_atom_mask[r_start:r_stop]

            if np.sum(res_fixed_atom_mask) == 0:
                # If all atoms in the residue are not fixed, e.g. ions
                fixed_atom_perm_list.extend(atom_res_perm.tolist())
                continue

            # Create a [N_res_atoms, N_res_perm] template of indices
            n_res_atoms, n_perm = atom_res_perm.shape
            indices_template = (
                atom_res_perm[:, 0].reshape(n_res_atoms, 1).repeat(n_perm, axis=1)
            )

            # Identify the column where the positions of the fixed atoms remain unchanged
            fixed_atom_perm = atom_res_perm[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            fixed_indices_template = indices_template[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            unchanged_columns_mask = np.all(
                fixed_atom_perm == fixed_indices_template, axis=0
            )

            # Remove the columns related to the position changes of fixed atoms.
            fiedx_atom_res_perm = atom_res_perm[:, unchanged_columns_mask]
            fixed_atom_perm_list.extend(fiedx_atom_res_perm.tolist())
        return {"atom_perm_list": fixed_atom_perm_list}

    @staticmethod
    @register_license('odesign2025')
    def get_gt_full_complex_features(
        atom_array: AtomArray,
        cropped_atom_array: AtomArray = None,
        get_cropped_asym_only: bool = True,
    ) -> tuple[dict[str, torch.Tensor], AtomArray]:
        """
        Extracts full ground truth complex features for multi-chain permutation alignment.
        
        Retrieves complete structure information for chains that appear in the cropped
        region, enabling proper evaluation and alignment across symmetric chain copies.
        
        The filtering behavior depends on get_cropped_asym_only:
            - True: Include only asymmetric units (chains) present in cropped region
                (favored for spatial cropping scenarios)
            - False: Include all chains from entities present in cropped region
                (includes symmetric copies not in the crop)
        
        Args:
            atom_array: Complete Biotite AtomArray for the full complex.
            cropped_atom_array: Spatially cropped AtomArray subset. If None, returns
                all atoms from atom_array. Defaults to None.
            get_cropped_asym_only: Whether to restrict to asymmetric units in crop.
                Defaults to True.
        
        Returns:
            tuple[dict[str, torch.Tensor], AtomArray]: A tuple containing:
                - Dictionary of ground truth features:
                    - coordinate: Atom positions [N_atom, 3]
                    - coordinate_mask: Valid position mask [N_atom]
                    - entity_mol_id: Entity-level molecular IDs [N_atom]
                    - mol_id: Molecular chain IDs [N_atom]
                    - mol_atom_index: Atom indices within molecules [N_atom]
                    - pae_rep_atom_mask: PAE representative atom mask [N_atom]
                - Filtered AtomArray containing selected atoms
        """
        gt_features = {}

        if cropped_atom_array is not None:
            # Get the cropped part of gt entities
            entity_atom_set = set(
                zip(
                    cropped_atom_array.entity_mol_id,
                    cropped_atom_array.mol_atom_index,
                )
            )
            mask = [
                (entity, atom) in entity_atom_set
                for (entity, atom) in zip(
                    atom_array.entity_mol_id, atom_array.mol_atom_index
                )
            ]

            if get_cropped_asym_only:
                # Restrict to asym chains appeared in cropped_atom_array
                asyms = np.unique(cropped_atom_array.mol_id)
                mask = mask * np.isin(atom_array.mol_id, asyms)
            atom_array = atom_array[mask]

        gt_features["coordinate"] = torch.Tensor(atom_array.coord)
        gt_features["coordinate_mask"] = torch.Tensor(atom_array.is_resolved).long()
        gt_features["entity_mol_id"] = torch.Tensor(atom_array.entity_mol_id).long()
        gt_features["mol_id"] = torch.Tensor(atom_array.mol_id).long()
        gt_features["mol_atom_index"] = torch.Tensor(atom_array.mol_atom_index).long()
        gt_features["pae_rep_atom_mask"] = torch.Tensor(
            atom_array.centre_atom_mask
        ).long()
        return gt_features, atom_array

    @register_license('odesign2025')
    def get_hotspot_features(
        self, 
        feat_shape: Optional[dict] = None, 
        interface_minimal_distance: float = 15.0, 
        min_distance: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Identifies hotspot residues at protein-ligand or protein-protein interfaces.
        
        Hotspot residues are defined as tokens near the interface between conditioned
        and generated entities. These are prioritized during evaluation and can guide
        model focus during training. Distance thresholds and sampling strategies ensure
        a balanced set of hotspot residues.
        
        Args:
            feat_shape: Optional dictionary specifying expected feature shapes.
                Used when use_hotspot_residue is False.
            interface_minimal_distance: Distance threshold (Angstroms) to define
                proximity at interfaces. Defaults to 15.0.
            min_distance: Whether to use minimum distance between token atoms (True)
                or center-to-center distance (False). Defaults to True.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - is_hotspot_residue: Binary mask for hotspot tokens [N_token]
        """
        if not self.use_hotspot_residue:
            is_hotspot_residue = torch.ones(
                feat_shape['is_hotspot_residue'], dtype=torch.bool
            )
            return {"is_hotspot_residue": is_hotspot_residue}

        centre_atom_array = self.cropped_atom_array[
            self.cropped_token_array.get_annotation('centre_atom_index')
        ]

        if self.inference_mode:
            is_hotspot_residue = torch.Tensor(
                centre_atom_array.is_hotspot_residue
            ).long()
            logger.info(
                f"Hotspot tokens: {is_hotspot_residue.sum()} "
                f"out of {len(is_hotspot_residue)}"
            )
            return {"is_hotspot_residue": is_hotspot_residue}

        if min_distance:
            token_coords = np.stack(
                [
                    np.pad(
                        self.cropped_atom_array.coord[idx],
                        ((0, 24 - self.cropped_atom_array.coord[idx].shape[0]), (0, 0)),
                        mode="constant", 
                        constant_values=np.nan
                    ) for idx in self.cropped_token_array.get_annotation("atom_indices")
                ], 
                axis=0
            )
            token_distance = np.nanmin(
                np.linalg.norm(
                    token_coords[:, None, :, None, :] - token_coords[None, :, None, :, :], 
                    axis=-1
                ), 
                axis=(-1,-2)
            )
        else:
            token_distance = cdist(
                centre_atom_array.coord,
                centre_atom_array.coord,
                "euclidean",
            )

        mask_distance = token_distance < interface_minimal_distance
        condition_token_mask = centre_atom_array.get_annotation(
            'condition_token_mask'
        )
        masked_entity_ids = set(
            centre_atom_array.entity_id_int[~condition_token_mask]
        )
        condition_entity_ids = set(
            centre_atom_array.entity_id_int[condition_token_mask]
        )
        shared_entity_ids = masked_entity_ids & condition_entity_ids
        
        is_masked_entity = np.isin(
            centre_atom_array.entity_id_int, list(masked_entity_ids)
        )[None, :]
        is_condition_entity = np.isin(
            centre_atom_array.entity_id_int, 
            list(condition_entity_ids - shared_entity_ids)
        )[:, None]
        mask_condition_generation = is_masked_entity & is_condition_entity
        mask = mask_distance & mask_condition_generation
        is_hotspot_residue = torch.from_numpy(mask.sum(axis=-1) > 0).long()
        
        is_hotspot_residue[~centre_atom_array.is_resolved] = 0
        condition_hotspot_mask = (
            (centre_atom_array.is_ligand != 1) & 
            (centre_atom_array.condition_token_mask)
        )
        condition_hotspot_res_idx = (
            is_hotspot_residue * condition_hotspot_mask
        ).nonzero().view(-1)
        token_distance_adjusted = (
            token_distance + np.eye(token_distance.shape[0]) * 10
        )
        token_distance = torch.from_numpy(token_distance_adjusted.min(axis=1))
        sorted_indices = torch.sort(
            token_distance[condition_hotspot_res_idx]
        ).indices
        condition_hotspot_res_idx = condition_hotspot_res_idx[sorted_indices]
        numbers = torch.arange(1, 21)
        weights = exponential_decay(numbers, peak_pos=4, decay_rate=0.3)
        sample = numbers[torch.multinomial(weights, 1)].item()
        canceled_condition_hotspot_res_idx = condition_hotspot_res_idx[sample:]
        is_hotspot_residue[canceled_condition_hotspot_res_idx] = 0
        
        if is_hotspot_residue.sum() == 0:
            print("No hotspot residues found in the cropped atom array.")
        
        return {"is_hotspot_residue": is_hotspot_residue}

    @register_license('odesign2025')
    def set_default_msa_features(
        self, features_dict: dict, feat_shape: dict
    ) -> dict[str, torch.Tensor]:
        """
        Creates default MSA features when no MSA data is available.
        
        Constructs minimal MSA features using only the input sequence, setting:
            - MSA as single-sequence (no homologs)
            - Zero deletion values and counts
            - Profile derived from input sequence
            - Zero alignment counts
        
        Args:
            features_dict: Dictionary of existing features containing residue types.
            feat_shape: Dictionary specifying expected shapes for MSA features.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of default MSA features:
                - msa: Single sequence MSA [1, N_token]
                - has_deletion: Zero deletion flags [1, N_token]
                - deletion_value: Zero deletion values [1, N_token]
                - profile: Sequence profile from restype [N_token, 32]
                - deletion_mean: Zero deletion means [N_token]
                - prot_pair_num_alignments: Zero count
                - prot_unpair_num_alignments: Zero count
                - rna_pair_num_alignments: Zero count
                - rna_unpair_num_alignments: Zero count
        """
        msa_features = {}

        tmp_restype = features_dict["restype"].clone()
        restype = tmp_restype[:, :32]
        restype[(tmp_restype[:, 32:] == 1).any(dim=1), -1] = 1
        msa_features["msa"] = torch.nonzero(restype)[:, 1].unsqueeze(
            0
        )
        assert msa_features["msa"].shape == feat_shape["msa"]
        msa_features["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        msa_features["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        msa_features["profile"] = restype

        assert msa_features["profile"].shape == feat_shape["profile"]
        msa_features["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            msa_features[key] = torch.tensor(0, dtype=torch.int32)

        return msa_features
    
    @register_license('odesign2025')
    def set_default_template_features(self, feat_shape: dict) -> dict[str, torch.Tensor]:
        """
        Creates default template features when no template data is available.
        
        Constructs minimal template features with:
            - All residues marked as gaps (type 31)
            - Zero masks (no valid atoms)
            - Zero positions
        
        Args:
            feat_shape: Dictionary specifying expected shapes for template features.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of default template features:
                - template_restype: Gap residue types [feat_shape]
                - template_all_atom_mask: Zero atom masks [feat_shape]
                - template_all_atom_positions: Zero positions [feat_shape]
        """
        template_features = {}

        template_features["template_restype"] = (
            torch.ones(feat_shape["template_restype"]) * 31
        )  # gap
        template_features["template_all_atom_mask"] = torch.zeros(
            feat_shape["template_all_atom_mask"]
        )
        template_features["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
        return template_features

    @register_license('odesign2025')
    def get_msa_features(
        self, features_dict: dict, feat_shape: dict
    ) -> dict[str, torch.Tensor]:
        """
        Retrieves or generates MSA features with appropriate masking.
        
        Either uses pre-computed MSA features or generates default features if unavailable.
        During training (non-inference mode), applies masking to conditioned tokens to
        prevent information leakage.
        
        Args:
            features_dict: Dictionary of existing features, used for masking.
            feat_shape: Dictionary specifying expected shapes for MSA features.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of MSA features with appropriate masking:
                - msa: Multiple sequence alignment [N_msa, N_token]
                - has_deletion: Deletion presence flags [N_msa, N_token]
                - deletion_value: Deletion gap lengths [N_msa, N_token]
                - profile: Sequence profile [N_token, 32]
                - deletion_mean: Mean deletion values [N_token]
                - alignment counts (various)
                - msa_token_mask: Token masking for training [N_token] (if not inference)
        """
        if len(self.msa_features) == 0:
            msa_features = self.set_default_msa_features(
                features_dict, feat_shape
            )
        else:
            msa_features = dict_to_tensor(self.msa_features)
        
        if not self.inference_mode:
            msa_token_mask = ~features_dict['is_condition_atom'][
                features_dict['distogram_rep_atom_mask'].bool()
            ]
            if self.mask_method != 'entity':
                msa_token_mask = torch.ones_like(msa_token_mask, dtype=torch.bool)
            msa_features["has_deletion"][:, msa_token_mask] = False
            msa_features["deletion_value"][:, msa_token_mask] = 0
            msa_features["deletion_mean"][msa_token_mask] = 0
            msa_features["profile"][msa_token_mask, :] = 0
            msa_features['msa_token_mask'] = msa_token_mask
        return msa_features

    @register_license('bytedance2024')
    def get_template_features(self, feat_shape: dict) -> dict[str, torch.Tensor]:
        """
        Retrieves or generates template features.
        
        Uses pre-computed template features if available, otherwise generates default
        features with zero values.
        
        Args:
            feat_shape: Dictionary specifying expected shapes for template features.
        
        Returns:
            dict[str, torch.Tensor]: Dictionary of template features:
                - template_restype: Template residue types
                - template_all_atom_mask: Template atom validity masks
                - template_all_atom_positions: Template atom positions
        """
        if len(self.template_features) == 0:
            template_features = self.set_default_template_features(feat_shape)
        else:
            template_features = dict_to_tensor(self.template_features)
        return template_features
    
    @register_license('odesign2025')
    def get_cyclic_features(self) -> dict[str, torch.Tensor]:
        """
        Identifies tokens that are part of cyclic structures.
        
        Extracts cyclicity information for each token, which is important for:
            - Proper handling of cyclic peptides
            - Constraint enforcement during structure generation
            - Evaluation of cyclic structure predictions
        
        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - is_cyclic_token: Binary flag for cyclic tokens [N_token]
        """
        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )        
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        is_cyclic_token = torch.Tensor(centre_atoms.if_cyc).long()

        return {"is_cyclic_token": is_cyclic_token}
    
