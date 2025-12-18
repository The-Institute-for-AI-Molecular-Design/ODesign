"""
Model interface definitions for ODesign.

This module defines data structures for inputs and outputs of various model components:
- GroundTruth: Ground truth labels for training
- PairFormerInput/Output: Input/output for PairFormer module
- DiffusionInput/Output: Input/output for Diffusion module
- PermutationInput: Input for symmetric permutation
- ODesignOutput: Final model output
- LossInput: Input features for loss computation

Note:
    - For input classes, use @auto_type_convert to automatically convert types
    - For output classes, avoid @auto_type_convert for mixed precision training compatibility
"""

import torch
import numpy as np
from typing import List
from attr import define
from ._base import *
from .data_interface import OFeatureData, OLabelData
from src.utils.license_register import register_license

@define
@register_license('odesign2025')
@auto_type_convert
class GroundTruth(DictAccessMixin):
    """
    Ground truth labels for model training and evaluation.
    
    This class contains all ground truth information including atomic coordinates,
    bond information, masks, and metadata needed for loss computation.
    
    Attributes:
        coordinate (FeatureType): Ground truth atomic coordinates.
            Shape: [N_atom, 3]
        coordinate_mask (MaskType): Mask indicating which atoms have valid coordinates.
            Shape: [N_atom]
            
        token_bond_type_label (FeatureType, optional): Bond type labels for token pairs.
            Shape: [N_token, N_token]
        ligand_bond_mask (MaskType, optional): Mask for ligand bonds.
            Shape: [N_atom, N_atom] or [N_token, N_token]
            
        entity_mol_id (IndexType, optional): Entity molecule ID for each atom.
            Shape: [N_atom]
        mol_id (IndexType, optional): Molecule ID for each atom.
            Shape: [N_atom]
        mol_atom_index (IndexType, optional): Atom index within molecule.
            Shape: [N_atom]
            
        pae_rep_atom_mask (MaskType, optional): Representative atom mask for PAE computation.
            Shape: [N_atom]
            
        eval_type (np.ndarray, optional): Evaluation type identifier.
        cluster_id (np.ndarray, optional): Cluster ID for evaluation.
        chain_1_mask (MaskType, optional): Mask for chain 1 in interface evaluation.
            Shape: [N_atom]
        chain_2_mask (MaskType, optional): Mask for chain 2 in interface evaluation.
            Shape: [N_atom]
            
        lddt_mask (MaskType, optional): Mask for lDDT computation (distance within threshold).
            Shape: [N_atom, N_atom]
        distance_mask (MaskType, optional): Mask for valid distance pairs.
            Shape: [N_atom, N_atom]
        distance (FeatureType, optional): Precomputed pairwise distances.
            Shape: [N_atom, N_atom]
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

    lddt_mask: MaskType | None = None
    distance_mask: MaskType | None = None
    distance: FeatureType | None = None

    @classmethod
    def from_label_data(cls, input_data: OLabelData) -> "GroundTruth":
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in input_data.items() if k in all_fields}
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, f"Missing required fields in {cls.__name__}: {missing_fields}"

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)
    

@define
@auto_type_convert
@register_license('odesign2025')
class PairFormerInput(DictAccessMixin):
    """
    Input features for PairFormer module (Algorithm 17 in AlphaFold3).
    
    This class contains all input features needed for the PairFormer trunk,
    including residue-level features, MSA features, atomic features, and
    positional encoding information.
    
    Attributes:
        residue_index (IndexType): Residue index in the sequence.
            Shape: [N_token]
        restype (OneHotType): One-hot encoded residue type.
            Shape: [N_token, n_residue_types]
        token_bonds (FeatureType): Bond features between tokens.
            Shape: [N_token, N_token, n_bond_features]
        profile (FeatureType): Sequence profile from MSA (e.g., position-specific scoring).
            Shape: [N_token, 32]
        deletion_mean (FeatureType): Mean deletion probability from MSA.
            Shape: [N_token]
        is_hotspot_residue (FeatureType): Binary flag indicating hotspot residues.
            Shape: [N_token]
            
        msa (FeatureType): Multiple sequence alignment features.
            Shape: [N_msa, N_token, n_msa_features]
        has_deletion (MaskType): Mask indicating presence of deletions in MSA.
            Shape: [N_msa, N_token]
        deletion_value (FeatureType): Deletion values in MSA.
            Shape: [N_msa, N_token]
        msa_token_mask (MaskType): Mask for valid MSA tokens.
            Shape: [N_msa, N_token]
            
        ref_pos (FeatureType): Reference atomic positions (from structure).
            Shape: [N_atom, 3]
        ref_space_uid (IndexType): Spatial unit ID for grouping atoms in same reference frame.
            Shape: [N_atom]
            
        ref_element (FeatureType): Atomic element encoding.
            Shape: [N_atom, n_elements] (one-hot or embedding)
        ref_charge (FeatureType): Atomic partial charge.
            Shape: [N_atom]
        ref_mask (MaskType): Mask indicating valid reference atoms.
            Shape: [N_atom]
        ref_atom_name_chars (OneHotType): One-hot encoded atom name characters.
            Shape: [N_atom, 4, n_char_types]
            
        atom_to_token_idx (IndexType): Mapping from atom index to token index.
            Shape: [N_atom]
            
        asym_id (IndexType): Asymmetric unit ID (chain identifier).
            Shape: [N_token]
        entity_id (IndexType): Entity ID (molecule type identifier).
            Shape: [N_token]
        sym_id (IndexType): Symmetry operation ID.
            Shape: [N_token]
        token_index (IndexType): Token index within residue.
            Shape: [N_token]
            
        constraint_feature (FeatureType): Constraint features for token pairs
            (e.g., from pocket, contact, substructure constraints).
            Shape: [N_token, N_token, n_constraint_features]
        token_bond_gen_mask (MaskType): Mask for bond type generation/evaluation.
            Shape: [N_token, N_token]
            
        is_cyclic_token (FlagType, optional): Flag indicating cyclic peptide tokens.
            Shape: [N_token]
    """
    # Residue-level features
    residue_index: IndexType
    restype: OneHotType  
    token_bonds: FeatureType
    profile: FeatureType
    deletion_mean: FeatureType
    is_hotspot_residue: FeatureType

    # MSA features
    msa: FeatureType
    has_deletion: MaskType
    deletion_value: FeatureType
    msa_token_mask: MaskType

    # Atomic position features
    ref_pos: FeatureType 
    ref_space_uid: IndexType

    # Atomic property features
    ref_element: FeatureType
    ref_charge: FeatureType 
    ref_mask: MaskType 
    ref_atom_name_chars: OneHotType

    # Mapping information
    atom_to_token_idx: IndexType

    # Positional encoding indices
    asym_id: IndexType
    entity_id: IndexType
    sym_id: IndexType
    token_index: IndexType

    # Pairwise features
    constraint_feature: FeatureType
    token_bond_gen_mask: MaskType

    # Optional fields
    is_cyclic_token: FlagType | None = None

    @classmethod
    def from_feature_data(cls, input_data: OFeatureData) -> "PairFormerInput":
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in input_data.items() if k in all_fields}
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, f"Missing required fields in {cls.__name__}: {missing_fields}"

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)


@define
@auto_type_convert
@register_license('odesign2025')
class PairFormerOutput(DictAccessMixin):
    """
    Output embeddings from PairFormer trunk module.
    
    This class contains the single (per-token) and pair (per-token-pair) embeddings
    produced by the PairFormer trunk, which are used by downstream modules like
    the Diffusion module and Pairwise heads.
    
    Attributes:
        s_inputs (EmbeddingType): Input single embeddings before PairFormer processing.
            Concatenation of per-token features from InputFeatureEmbedder.
            Shape: [N_token, c_s_inputs]
            where c_s_inputs = c_token + sum(INPUTFEATS_DIMS) ≈ 449
            
        s (EmbeddingType): Single embeddings after PairFormer processing.
            Processed through PairFormer stack (MSA + Pairformer blocks).
            Shape: [N_token, c_s]
            where c_s = 384 (default)
            
        z (PairwiseEmbeddingType): Pair embeddings after PairFormer processing.
            Captures pairwise relationships between tokens.
            Shape: [N_token, N_token, c_z]
            where c_z = 128 (default)
    
    Methods:
        get_embedding(): Returns tuple of (s_inputs, s, z) for downstream use.
    """
    s_inputs: EmbeddingType
    s: EmbeddingType
    z: PairwiseEmbeddingType

    def get_embedding(self) -> tuple[EmbeddingType, EmbeddingType, PairwiseEmbeddingType]:
        """
        Get all embedding outputs as a tuple.
        
        Returns:
            tuple: (s_inputs, s, z) embeddings for downstream modules.
        """
        return self.s_inputs, self.s, self.z


@define
@auto_type_convert
@register_license('odesign2025')
class PairwiseOutput(DictAccessMixin):
    """
    Output from Pairwise prediction heads.
    
    This class contains predictions for token-level pairwise properties including
    distance distributions (distogram) and bond types.
    
    Attributes:
        distogram (LogitsType): Distance distribution logits for token pairs.
            Predicts binned distances between representative atoms of token pairs.
            Shape: [N_token, N_token, n_distance_bins]
            where n_distance_bins = 64 (default, covering 2.3-21.7 Å)
            
        token_bond_type_logits (LogitsType): Bond type prediction logits.
            Predicts chemical bond types between token pairs.
            Shape: [N_token, N_token, n_bond_types]
            where n_bond_types includes: no_bond, single, double, triple, aromatic, etc.
    """
    distogram: LogitsType
    token_bond_type_logits: LogitsType


@define
@auto_type_convert
@register_license('odesign2025')
class DiffusionInput(DictAccessMixin):
    """
    Input features for Diffusion module (Algorithm 20 in AlphaFold3).
    
    This class contains the atomic-level features needed for the diffusion-based
    structure generation module, including reference positions, atomic properties,
    and conditioning information.
    
    Attributes:
        ref_pos (FeatureType): Reference atomic positions (ground truth or initial guess).
            Shape: [N_atom, 3]
            
        ref_space_uid (IndexType): Spatial unit ID for grouping atoms in same reference frame.
            Used for residue-level symmetric atom permutation.
            Shape: [N_atom]
            
        ref_element (FeatureType): Atomic element encoding.
            Shape: [N_atom, n_elements] or [N_atom]
            
        ref_mask (MaskType): Mask indicating valid atoms (exists in structure).
            Shape: [N_atom]
            
        ref_atom_name_chars (OneHotType): One-hot encoded atom name characters.
            Used in AtomAttentionEncoder for atom identification.
            Shape: [N_atom, 4, n_char_types]
            
        ref_charge (FeatureType): Atomic partial charge.
            Shape: [N_atom]
            
        atom_to_token_idx (IndexType): Mapping from atom index to token index.
            Used for aggregating atom features to token level.
            Shape: [N_atom]
            
        residue_index (IndexType): Residue index in sequence.
            Shape: [N_token]
            
        asym_id (IndexType): Asymmetric unit ID (chain identifier).
            Shape: [N_token]
            
        entity_id (IndexType): Entity ID (molecule type identifier).
            Shape: [N_token]
            
        sym_id (IndexType): Symmetry operation ID.
            Shape: [N_token]
            
        token_index (IndexType): Token index within residue.
            Shape: [N_token]
            
        is_condition_atom (MaskType): Mask indicating which atoms are conditioned
            (fixed during diffusion, not generated).
            Shape: [N_atom]
            
        is_cyclic_token (FlagType, optional): Flag for cyclic peptide tokens.
            Shape: [N_token]
            
        cyclic_mode (str, optional): Mode for handling cyclic structures.
            Options: "full", "partial". Defaults to "full".
            
        cycle_bonds (list, optional): List of bonds forming cycles.
            For cyclic peptide constraint handling.
    """
    ref_pos: FeatureType 
    ref_space_uid: IndexType

    ref_element: FeatureType
    ref_mask: MaskType 
    ref_atom_name_chars: OneHotType
    ref_charge: FeatureType

    atom_to_token_idx: IndexType

    residue_index: IndexType
    asym_id: IndexType
    entity_id: IndexType
    sym_id: IndexType
    token_index: IndexType

    is_condition_atom: MaskType

    is_cyclic_token: FlagType | None = None
    cyclic_mode: str = "full"
    cycle_bonds: list = []
    
    @classmethod
    def from_feature_data(cls, input_data: OFeatureData) -> "DiffusionInput":
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in input_data.items() if k in all_fields}
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, f"Missing required fields in {cls.__name__}: {missing_fields}"

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)
    

@define
@auto_type_convert
@register_license('odesign2025')
class DiffusionOutput(DictAccessMixin):
    """
    Output from Diffusion module.
    
    This class contains the denoised atomic coordinates predicted by the
    diffusion module, along with optional noise level information.
    
    Attributes:
        x_denoised (FeatureType): Denoised atomic coordinates.
            In training: Output of denoising network F_θ at given noise level.
            In inference: Final denoised coordinates after full diffusion sampling.
            Shape: [N_sample, N_atom, 3] (training/rollout) or [N_atom, 3] (inference)
            
        sigma (torch.Tensor, optional): Noise level(s) used for denoising.
            In training: Sampled noise level from log-normal distribution.
            In inference: Sequence of noise levels from sampling schedule.
            Shape: [N_sample] or scalar
    """
    x_denoised: FeatureType
    sigma: torch.Tensor | None = None


@define
@auto_type_convert
@register_license('odesign2025')
class PermutationInput(DictAccessMixin):
    """
    Input features for symmetric permutation operations.
    
    This class contains the information needed to perform chain permutation
    and atom permutation to resolve symmetry ambiguities in structures.
    Used in training to align ground truth to predictions.
    
    Attributes:
        entity_mol_id (IndexType): Entity molecule ID for each atom.
            Used to group atoms by molecule for permutation.
            Shape: [N_atom]
            
        mol_id (IndexType): Molecule ID for each atom.
            Identifies individual molecules (multiple copies of same entity).
            Shape: [N_atom]
            
        mol_atom_index (IndexType): Atom index within molecule.
            Maps atoms to their position within molecule for permutation.
            Shape: [N_atom]
            
        pae_rep_atom_mask (MaskType): Representative atom mask for PAE computation.
            Identifies atoms used as representatives for token-level metrics.
            Shape: [N_atom]
            
        is_ligand (MaskType): Mask indicating ligand atoms.
            Used for ligand-specific permutation handling.
            Shape: [N_atom]
            
        masked_asym_ids (List[int]): List of asymmetric unit IDs to consider
            for permutation. Used to select which chains can be permuted.
            
        ref_space_uid (IndexType): Spatial unit ID for residue-level grouping.
            Used for symmetric atom permutation within residues.
            Shape: [N_atom]
            
        atom_perm_list (List[int]): List of atom permutation groups.
            Defines which atoms can be permuted (e.g., symmetric atoms in
            aromatic rings, carboxyl groups, etc.).
    """
    entity_mol_id: IndexType
    mol_id: IndexType
    mol_atom_index: IndexType

    pae_rep_atom_mask: MaskType
    is_ligand: MaskType
    masked_asym_ids: List[int]

    ref_space_uid: IndexType
    atom_perm_list: List[int]

    @classmethod
    def from_feature_data(cls, input_data: OFeatureData) -> "PermutationInput":
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in input_data.items() if k in all_fields}
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, f"Missing required fields in {cls.__name__}: {missing_fields}"

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)
    

@define
@auto_type_convert
@register_license('odesign2025')
class ODesignOutput(DictAccessMixin):
    """
    Complete output from ODesign model.
    
    This class aggregates outputs from all model components including
    pairwise predictions (distogram, bonds) and structural predictions
    (atomic coordinates from diffusion).
    
    Attributes:
        token_bond_type_logits (LogitsType, optional): Predicted bond type logits.
            Output from PairwiseHead for bond type classification.
            Shape: [N_token, N_token, n_bond_types]
            
        distogram (LogitsType, optional): Distance distribution logits.
            Output from DistogramHead for distance prediction.
            Shape: [N_token, N_token, n_distance_bins]
            
        token_bond_gen_mask (MaskType, optional): Mask for bond generation evaluation.
            Indicates which token pairs should be evaluated for bond prediction.
            Shape: [N_token, N_token]
            
        coordinate (FeatureType, optional): Predicted atomic coordinates.
            Output from Diffusion module.
            Shape: [N_sample, N_atom, 3] (training/rollout) or [N_atom, 3] (inference)
            
        noise_level (torch.Tensor, optional): Noise levels used in diffusion.
            Shape: [N_sample] (training) or [N_diffusion_steps] (inference)
            
        distance (FeatureType, optional): Precomputed pairwise distances from coordinates.
            Used for efficient loss computation when not using sparse mode.
            Shape: [N_sample, N_atom, N_atom] or [N_atom, N_atom]
    
    Note:
        Fields are optional as different components may be enabled/disabled
        based on training configuration (e.g., distogram may be disabled
        in some training stages).
    """
    # Pairwise predictions
    token_bond_type_logits: LogitsType | None = None
    distogram: LogitsType | None = None

    # Evaluation masks
    token_bond_gen_mask: MaskType | None = None

    # Structural predictions
    coordinate: FeatureType | None = None
    noise_level: torch.Tensor | None = None
    distance: FeatureType | None = None


@define
@auto_type_convert
@register_license('odesign2025')
class LossInput(DictAccessMixin):
    """
    Input features for loss computation.
    
    This class contains molecular type information and metadata needed
    for computing various loss components with appropriate weighting and
    filtering.
    
    Attributes:
        is_rna (MaskType): Binary mask indicating RNA atoms.
            Used for RNA-specific loss weighting in MSE loss.
            Shape: [N_atom]
            
        is_dna (MaskType): Binary mask indicating DNA atoms.
            Used for DNA-specific loss weighting in MSE loss.
            Shape: [N_atom]
            
        is_ligand (MaskType): Binary mask indicating ligand atoms.
            Used for ligand-specific loss weighting and bond loss computation.
            Shape: [N_atom]
            
        is_condition_atom (MaskType): Mask indicating conditioned (fixed) atoms.
            Used to determine alignment strategy and filter generation atoms.
            Shape: [N_atom]
            
        resolution (FeatureType): Structure resolution in Angstroms.
            Used to filter confidence loss computation (only for structures
            with resolution in valid range).
            Shape: scalar or [1]
            
        distogram_rep_atom_mask (MaskType, optional): Representative atom mask
            for distogram loss computation. Identifies atoms used as token
            representatives (Cβ/Cα for proteins, C4'/C2' for nucleotides).
            Shape: [N_atom]
    
    Note:
        MSE loss uses different weights for different molecule types:
        - RNA: weight_rna (default 5.0)
        - DNA: weight_dna (default 5.0)
        - Ligand: weight_ligand (default 10.0)
        - Others: weight 1.0
    """
    is_rna: MaskType  
    is_dna: MaskType  
    is_ligand: MaskType
    is_condition_atom: MaskType
    resolution: FeatureType 
    distogram_rep_atom_mask: MaskType | None

    @classmethod
    def from_feature_data(cls, input_data: OFeatureData) -> "PermutationInput":
        all_fields = {field.name for field in attr.fields(cls)}
        filtered_dict = {k: v for k, v in input_data.items() if k in all_fields}
        def _check_required_fields(filtered_dict):
            required_fields = {
                field.name for field in attr.fields(cls) 
                if field.default == attr.NOTHING
            }
            missing_fields = required_fields - set(filtered_dict.keys())
            assert not missing_fields, f"Missing required fields in {cls.__name__}: {missing_fields}"

        _check_required_fields(filtered_dict)
        return cls(**filtered_dict)