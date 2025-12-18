"""
Data interface definitions for ODesign dataset and feature processing.

This module defines standardized data structures for features and labels:
- OFeatureData: Complete feature set for model input
- OLabelData: Ground truth labels for training

These interfaces ensure type consistency and provide convenient property
accessors for data dimensions.
"""

from ._base import *
from typing import List
from attr import define
import attr
import numpy as np
from src.utils.license_register import register_license


@define(kw_only=True)
@register_license('odesign2025')
@auto_type_convert
class OFeatureData(DictAccessMixin):
    """
    Complete feature data structure for ODesign model input.
    
    This class contains all features extracted from protein/RNA/DNA/ligand structures,
    including token-level features, atomic features, MSA features, template features,
    and various masks for training and evaluation. Features are organized by category.
    
    Attributes:
        # Token-level features
        token_index (IndexType): Token index within the full sequence.
            Shape: [N_token]
            
        residue_index (IndexType): Residue index in original sequence (1-indexed).
            Shape: [N_token]
            
        asym_id (IndexType): Asymmetric unit ID (chain identifier).
            Shape: [N_token]
            
        entity_id (IndexType): Entity ID (molecule type identifier).
            Same entity ID = same sequence.
            Shape: [N_token]
            
        sym_id (IndexType): Symmetry operation ID.
            Shape: [N_token]
            
        restype (OneHotType): One-hot encoded residue/nucleotide/ligand type.
            Shape: [N_token, n_residue_types]
            
        # Bond features
        token_bonds (FeatureType): Bond features between token pairs.
            Encodes bond types and connectivity.
            Shape: [N_token, N_token, n_bond_features]
            
        token_pair_gen_mask (MaskType): Mask for token pairs to generate/evaluate.
            Shape: [N_token, N_token]
            
        token_bond_gen_mask (MaskType): Mask for bond type generation/evaluation.
            Shape: [N_token, N_token]
            
        # Reference atomic features
        ref_pos (FeatureType): Reference atomic positions from structure.
            Shape: [N_atom, 3]
            
        ref_mask (MaskType): Mask for valid reference atoms (exists in structure).
            Shape: [N_atom]
            
        ref_element (FeatureType): Atomic element encoding.
            Shape: [N_atom, n_elements] or [N_atom]
            
        ref_charge (FeatureType): Atomic partial charge.
            Shape: [N_atom]
            
        ref_atom_name_chars (OneHotType): One-hot encoded atom name characters.
            Shape: [N_atom, 4, n_char_types]
            
        ref_space_uid (IndexType): Spatial unit ID for grouping atoms in same frame.
            Used for residue-level symmetric atom permutation.
            Shape: [N_atom]
            
        has_frame (MaskType): Mask indicating atoms with valid coordinate frames.
            Shape: [N_atom]
            
        frame_atom_index (IndexType): Indices of atoms defining coordinate frames.
            For proteins: [N, CA, C] atoms; for nucleotides: [C1', C3', C4']
            Shape: [N_frame, 3]
            
        # Identity and mapping features
        atom_to_token_idx (IndexType): Mapping from atom index to token index.
            Shape: [N_atom]
            
        atom_to_tokatom_idx (IndexType): Mapping from atom to token-atom index.
            Index within atoms of the same token.
            Shape: [N_atom]
            
        is_protein (MaskType): Binary mask indicating protein atoms.
            Shape: [N_atom]
            
        is_ligand (MaskType): Binary mask indicating ligand atoms.
            Shape: [N_atom]
            
        is_dna (MaskType): Binary mask indicating DNA atoms.
            Shape: [N_atom]
            
        is_rna (MaskType): Binary mask indicating RNA atoms.
            Shape: [N_atom]
            
        # Resolution
        resolution (FeatureType): Structure resolution in Angstroms.
            Used for filtering confidence loss. Shape: scalar or [1]
            
        # Chain permutation features
        mol_id (IndexType): Molecule ID for each atom.
            Identifies individual molecules (multiple copies of same entity).
            Shape: [N_atom]
            
        mol_atom_index (IndexType): Atom index within molecule.
            Shape: [N_atom]
            
        entity_mol_id (IndexType): Entity molecule ID for each atom.
            Shape: [N_atom]
            
        masked_asym_ids (List[int], optional): List of asymmetric unit IDs
            to consider for chain permutation. None means all chains.
            
        # Atom permutation features
        atom_perm_list (list[list[int]]): List of symmetric atom groups.
            Each group contains atom indices that can be permuted
            (e.g., symmetric atoms in aromatic rings, carboxyl groups).
            
        # Various masks for training and evaluation
        pae_rep_atom_mask (MaskType): Representative atom mask for PAE computation.
            Shape: [N_atom]
            
        modified_res_mask (MaskType): Mask for modified residues.
            Shape: [N_token]
            
        is_condition_atom (MaskType): Mask for conditioned (fixed) atoms.
            Atoms not generated by diffusion.
            Shape: [N_atom]
            
        distogram_rep_atom_mask (MaskType): Representative atom mask for distogram.
            Typically Cβ/Cα for proteins, C4'/C2' for nucleotides.
            Shape: [N_atom]
            
        plddt_m_rep_atom_mask (MaskType): Representative atom mask for pLDDT-m.
            Shape: [N_atom]
            
        bond_mask (MaskType): Mask for valid bonds.
            Shape: [N_atom, N_atom] or [N_token, N_token]
            
        # Constraint features
        constraint_feature (FeatureType, optional): Constraint features for token pairs.
            Currently supports distogram constraint, pocket constraint, etc.
            Shape: [N_token, N_token, n_constraint_features]
            
        # MSA features (optional)
        msa (IndexType, optional): Multiple sequence alignment.
            Token indices from MSA sequences.
            Shape: [N_msa, N_token]
            
        has_deletion (MaskType, optional): Mask indicating deletions in MSA.
            Shape: [N_msa, N_token]
            
        deletion_value (FeatureType, optional): Deletion values in MSA.
            Shape: [N_msa, N_token]
            
        profile (FeatureType, optional): Sequence profile from MSA.
            Position-specific scoring matrix.
            Shape: [N_token, 32]
            
        deletion_mean (FeatureType, optional): Mean deletion probability.
            Shape: [N_token]
            
        msa_token_mask (MaskType, optional): Mask for valid MSA tokens.
            Shape: [N_msa, N_token]
            
        prot_pair_num_alignments (torch.Tensor, optional): Number of paired
            protein alignments.
            
        prot_unpair_num_alignments (torch.Tensor, optional): Number of unpaired
            protein alignments.
            
        rna_pair_num_alignments (torch.Tensor, optional): Number of paired
            RNA alignments.
            
        rna_unpair_num_alignments (torch.Tensor, optional): Number of unpaired
            RNA alignments.
            
        # Template features (under development)
        template_restype (IndexType, optional): Residue types in template structures.
            Shape: [N_template, N_token]
            
        template_all_atom_mask (MaskType, optional): Atom masks in templates.
            Shape: [N_template, N_token, n_atom_per_token]
            
        template_all_atom_positions (FeatureType, optional): Atom positions in templates.
            Shape: [N_template, N_token, n_atom_per_token, 3]
            
        # Hotspot features
        is_hotspot_residue (FeatureType, optional): Binary mask for hotspot residues.
            Residues of particular interest (e.g., binding site, catalytic residues).
            Shape: [N_token]
            
        # Cyclic structure features
        is_cyclic_token (FeatureType, optional): Binary mask for cyclic peptide tokens.
            Shape: [N_token]
    
    Properties:
        num_asym: Number of asymmetric units (chains).
        num_token: Number of tokens.
        num_atom: Number of atoms.
        num_msa: Number of MSA sequences.
        default_num_msa: Default number of MSA sequences for padding (1).
        default_num_templ: Default number of templates for padding (4).
        default_num_pocket: Default number of pocket residues (30).
    """
    # Token-level features
    token_index: IndexType
    residue_index: IndexType
    asym_id: IndexType
    entity_id: IndexType
    sym_id: IndexType
    restype: OneHotType

    # Bond features
    token_bonds: FeatureType
    token_pair_gen_mask: MaskType
    token_bond_gen_mask: MaskType

    # Reference atomic features
    ref_pos: FeatureType
    ref_mask: MaskType
    ref_element: FeatureType
    ref_charge: FeatureType
    ref_atom_name_chars: OneHotType
    ref_space_uid: IndexType
    has_frame: MaskType
    frame_atom_index: IndexType

    # Identity and mapping features
    atom_to_token_idx: IndexType
    atom_to_tokatom_idx: IndexType
    is_protein: MaskType
    is_ligand: MaskType
    is_dna: MaskType
    is_rna: MaskType

    # Resolution
    resolution: FeatureType

    # Chain permutation features
    mol_id: IndexType
    mol_atom_index: IndexType
    entity_mol_id: IndexType
    masked_asym_ids: List[int] | None = None

    # Atom permutation features
    atom_perm_list: list[list[int]]

    # Mask features for training and evaluation
    pae_rep_atom_mask: MaskType
    modified_res_mask: MaskType
    is_condition_atom: MaskType
    distogram_rep_atom_mask: MaskType
    plddt_m_rep_atom_mask: MaskType
    bond_mask: MaskType

    # Constraint features
    constraint_feature: FeatureType | None = None

    # MSA features (optional)
    msa: IndexType | None = None
    has_deletion: MaskType | None = None
    deletion_value: FeatureType | None = None
    profile: FeatureType | None = None
    deletion_mean: FeatureType | None = None
    msa_token_mask: MaskType | None = None
    prot_pair_num_alignments: torch.Tensor | None = None
    prot_unpair_num_alignments: torch.Tensor | None = None
    rna_pair_num_alignments: torch.Tensor | None = None
    rna_unpair_num_alignments: torch.Tensor | None = None

    # Template features (under development)
    template_restype: IndexType | None = None
    template_all_atom_mask: MaskType | None = None
    template_all_atom_positions: FeatureType | None = None

    # Hotspot features
    is_hotspot_residue: FeatureType | None = None

    # Cyclic structure features
    is_cyclic_token: FeatureType | None = None
    
    @classmethod
    def from_feature_dict(cls, feature_dict: dict) -> "OFeatureData":
        """
        Create OFeatureData instance from a feature dictionary.
        
        This factory method filters the input dictionary to only include
        fields defined in OFeatureData and validates that all required
        fields are present.
        
        Args:
            feature_dict (dict): Dictionary containing feature data.
                Keys should match OFeatureData field names.
        
        Returns:
            OFeatureData: Validated feature data instance.
        
        Raises:
            AssertionError: If required fields are missing.
        """
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in feature_dict.items() if k in all_fields}
        
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, (
                f"Missing required fields in {cls.__name__}: {missing_fields}"
            )

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)
    
    @property
    def num_asym(self) -> int:
        """
        Get the number of asymmetric units (chains) in the structure.
        
        Returns:
            int: Number of unique chains.
        """
        return len(torch.unique(self.asym_id))
    
    @property
    def num_token(self) -> int:
        """
        Get the number of tokens in the sequence.
        
        Returns:
            int: Number of tokens (residues/nucleotides/ligand atoms).
        """
        return self.token_index.shape[0]
    
    @property
    def num_atom(self) -> int:
        """
        Get the number of atoms in the structure.
        
        Returns:
            int: Total number of atoms.
        """
        return self.atom_to_token_idx.shape[0]
    
    @property
    def num_msa(self) -> int:
        """
        Get the number of MSA sequences.
        
        Returns:
            int: Number of sequences in multiple sequence alignment.
        """
        return self.msa.shape[0]
    
    @property
    def default_num_msa(self) -> int:
        """
        Get the default number of MSA sequences for padding.
        
        Returns:
            int: Default value (1) for structures without MSA.
        """
        return 1

    @property
    def default_num_templ(self) -> int:
        """
        Get the default number of template structures for padding.
        
        Returns:
            int: Default value (4) for template feature dimensions.
        """
        return 4

    @property
    def default_num_pocket(self) -> int:
        """
        Get the default number of pocket residues.
        
        Returns:
            int: Default value (30) for pocket size in ligand binding.
        """
        return 30
    

@define(kw_only=True)
@register_license('odesign2025')
@auto_type_convert
class OLabelData(DictAccessMixin):
    """
    Ground truth label data structure for training and evaluation.
    
    This class contains all ground truth labels including atomic coordinates,
    bond types, and metadata needed for loss computation and evaluation metrics.
    
    Attributes:
        coordinate (FeatureType): Ground truth atomic coordinates.
            Reference positions from experimental structure.
            Shape: [N_atom, 3]
            
        coordinate_mask (MaskType): Mask indicating which atoms have valid coordinates.
            Used to filter missing or disordered atoms.
            Shape: [N_atom]
            
        token_bond_type_label (FeatureType, optional): Ground truth bond type labels.
            Encodes bond types for token pairs (no_bond, single, double, triple, aromatic).
            Shape: [N_token, N_token] or [N_token, N_token, n_bond_types]
            
        ligand_bond_mask (MaskType, optional): Mask for ligand bonds.
            Indicates which bonds should be considered for ligand bond loss.
            Shape: [N_atom, N_atom] or [N_token, N_token]
            
        entity_mol_id (IndexType, optional): Entity molecule ID for each atom.
            Used for chain permutation during training.
            Shape: [N_atom]
            
        mol_id (IndexType, optional): Molecule ID for each atom.
            Identifies individual molecule instances.
            Shape: [N_atom]
            
        mol_atom_index (IndexType, optional): Atom index within molecule.
            Maps atoms to their position within molecule for permutation.
            Shape: [N_atom]
            
        pae_rep_atom_mask (MaskType, optional): Representative atom mask for PAE.
            Identifies atoms used as representatives for predicted aligned error.
            Shape: [N_atom]
            
        eval_type (np.ndarray, optional): Evaluation type identifier.
            Specifies the type of evaluation (e.g., interface, binding, folding).
            
        cluster_id (np.ndarray, optional): Cluster ID for evaluation.
            Used to group similar structures in validation.
            
        chain_1_mask (MaskType, optional): Mask for chain 1 in interface evaluation.
            Used for protein-protein interface or ligand binding evaluation.
            Shape: [N_atom]
            
        chain_2_mask (MaskType, optional): Mask for chain 2 in interface evaluation.
            Used for protein-protein interface or ligand binding evaluation.
            Shape: [N_atom]
    """
    coordinate: FeatureType
    coordinate_mask: MaskType

    token_bond_type_label: FeatureType | None = None
    ligand_bond_mask: MaskType | None = None

    entity_mol_id: IndexType | None = None
    mol_id: IndexType | None = None
    mol_atom_index: IndexType | None = None

    pae_rep_atom_mask: MaskType | None = None

    eval_type: np.ndarray | None = None
    cluster_id: np.ndarray | None = None
    chain_1_mask: MaskType | None = None
    chain_2_mask: MaskType | None = None

    @classmethod
    def from_label_dict(cls, label_dict: dict) -> "OLabelData":
        """
        Create OLabelData instance from a label dictionary.
        
        This factory method filters the input dictionary to only include
        fields defined in OLabelData and validates that all required
        fields are present.
        
        Args:
            label_dict (dict): Dictionary containing label data.
                Keys should match OLabelData field names.
        
        Returns:
            OLabelData: Validated label data instance.
        
        Raises:
            AssertionError: If required fields are missing.
        """
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in label_dict.items() if k in all_fields}
        
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, (
                f"Missing required fields in {cls.__name__}: {missing_fields}"
            )

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)
    