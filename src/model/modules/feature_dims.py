# This file is used to define the extra feature dimensions of different modules.

INPUTFEATS_DIMS = {
    "restype": 35, 
    "profile": 32, 
    "deletion_mean": 1, 
    "is_hotspot_residue": 1
} # used by src.model.modules.embedders.InputFeatureEmbedder

RELPOSENC_DIMS = {
    "asym_id": 1,
    "residue_index": 1,
    "entity_id": 1,
    "sym_id": 1,
    "token_index": 1,
} # used by src.model.modules.embedders.RelativePositionEncoding

MSAFEATS_DIMS = {
    "msa": 32,
    "has_deletion": 1,
    "deletion_value": 1,
} # used by src.model.modules.pairformer.MSAModule

ATOMATTNENC_DIMS = {
    "ref_mask": 1,
    "ref_element": 129,
    "ref_atom_name_chars": 4 * 64,
} # used by src.model.modules.transformer.AtomAttentionEncoder