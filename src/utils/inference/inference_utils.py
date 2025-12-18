import random
from copy import deepcopy
import numpy as np
import logging
from typing import Optional
from collections import defaultdict
from biotite.structure import AtomArray
from biotite.structure.io import pdb, pdbx
import biotite.structure as struc

import src.utils.data.ccd as ccd
from src.utils.data.filter import Filter
from src.utils.data.misc import int_to_letters, get_data_shape_dict
from src.utils.data.data_pipeline import DataPipeline
from src.utils.data.parser import AddAtomArrayAnnot, MMCIFParser
from src.utils.data.tokenizer import AtomArrayTokenizer, TokenArray
from src.utils.data.featurizer import Featurizer
from src.utils.data.msa_featurizer import InferenceMSAFeaturizer
from src.utils.data.constants import NA_BACKBONE_ATOM_NAMES, mmcif_restype_1to3
from src.api.data_interface import (
    OFeatureData,
    OLabelData,
)

logger = logging.getLogger(__name__)


def empty_like_atomarray(arr):
    # make empty atomarray with all the annotation as arr
    empty = AtomArray(0)
    for category in arr.get_annotation_categories():
        dtype = arr.get_annotation(category).dtype
        empty.set_annotation(category, np.array([], dtype=dtype))
    return empty

def get_backbone_atom_array(backbone_type: str):

    if backbone_type == 'proteinChain':
        backbone = ccd.get_component_atom_array('GLY')
        backbone.hetero[:] = False

    elif backbone_type == 'dnaChain':
        backbone = ccd.get_component_atom_array('DC')
        backbone = backbone[np.isin(backbone.atom_name, NA_BACKBONE_ATOM_NAMES)]
        backbone.atom_name = np.char.replace(backbone.atom_name, 'N1', 'N')
        backbone.hetero[:] = False

    elif backbone_type == 'rnaChain':
        backbone = ccd.get_component_atom_array('C')
        backbone = backbone[np.isin(backbone.atom_name, NA_BACKBONE_ATOM_NAMES)]
        backbone.atom_name = np.char.replace(backbone.atom_name, 'N1', 'N')
        backbone.hetero[:] = False

    elif backbone_type == 'ligand':
        backbone = AtomArray(1)
        backbone.coord[0] = np.zeros((3,), dtype=np.float32)
        backbone.chain_id[0] = ""
        backbone.res_id[0] = 0
        backbone.ins_code[0] = ""
        backbone.res_name[0] = "-L"
        backbone.hetero[0] = True
        backbone.atom_name[0] = "-"
        backbone.element[0] = "C"
        backbone.set_annotation('charge', np.array([0], dtype=np.int_))

    backbone.set_annotation('condition_token_mask', np.zeros(len(backbone), dtype=np.bool_))
    backbone.set_annotation('is_hotspot_residue', np.zeros(len(backbone), dtype=np.bool_))

    return backbone

def add_ligand_fake_bond(atom_array: AtomArray) -> AtomArray:

    # add fake bond for design ligand
    if any(atom_array.res_name == '-L'):
        bonds = atom_array.bonds
        if bonds is None:
            bonds = struc.BondList(atom_array.array_length())
        r_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
            r = atom_array[r_s:r_e]
            if r.res_name[0] == '-L':
                for b_s, b_e in zip(range(r_s, r_e - 1), range(r_s + 1, r_e)):
                    bonds.add_bond(b_s, b_e, bond_type=struc.BondType.SINGLE)
        atom_array.bonds = bonds
        return atom_array
    else:
        return atom_array

def sample_length(segments: str, length: int) -> list:

    min_len = []
    max_len = []
    length_left = length
    segments = segments.split(',')
    num_segments = len(segments)
    for segment in segments:
        if '/' in segment:
            ref_c_id, ref_r_range = segment.split('/')
            s, e = ref_r_range.split('-')
            s, e = int(s), int(e)
            seg_len = e - s + 1
            min_len.append(-1)
            max_len.append(-1)
            length_left -= seg_len
        else:
            s, e = segment.split('-')
            s, e = int(s), int(e)
            min_len.append(s)
            max_len.append(e)
            length_left -= s
    
    n = len(segments)
    sample_l = deepcopy(min_len)
    for _ in range(length_left):
        available_indices = [
            i for i in range(n) if (sample_l[i] < max_len[i])
        ]
        if not available_indices:
            raise ValueError("Cannot sample lengths within the given constraints.")
        chosen_index = random.choice(available_indices)
        sample_l[chosen_index] += 1
    
    reshaped_segments = []
    for l, seg in zip(sample_l, segments):
        if l == -1:
            reshaped_segments.append(seg)
        else:
            reshaped_segments.append(f"{l}-{l}")

    return reshaped_segments

def build_chain_atom_array(chain_sequence: str, chain_type: str, im: str, chain_length: int | None, ref_atom_array: AtomArray | None = None) -> tuple[AtomArray, str]:
    
    if im == "antibody" or ref_atom_array is None:
        chain_array = None
        res_id = 1
        assert chain_sequence.count('-') == len(chain_length.split(',')), "Length of antibody chain does not match"
        length = chain_length.split(',')
        idx = 0
        for aa in chain_sequence:
            # Is Framework
            if aa != "-":
                ccdCode = mmcif_restype_1to3[aa]
                aa_array = ccd.get_component_atom_array(ccdCode)
                aa_array.hetero[:] = False
                aa_array.res_id[:] = res_id
                res_id += 1
                aa_array.set_annotation('condition_token_mask', np.ones(len(aa_array), dtype=np.bool_))
                aa_array.set_annotation('is_hotspot_residue', np.zeros(len(aa_array), dtype=np.bool_))
                if chain_array is None:
                    chain_array = aa_array
                else:
                    chain_array += aa_array
            # Is CDR to design
            else:
                l_seg = length[idx]
                l_s, l_e = l_seg.split('-')
                l_s, l_e = int(l_s), int(l_e)
                l = random.randint(l_s, l_e + 1)
                for _ in range(l):
                    backbone_array = get_backbone_atom_array(chain_type)
                    backbone_array.res_id[:] = res_id
                    res_id += 1
                    chain_array += backbone_array
                idx += 1
        chain_array.set_annotation('is_antigen_chain', np.zeros(len(chain_array), dtype=np.bool_))
    # antigen
    else:
        chain_array = empty_like_atomarray(ref_atom_array)
        segments = chain_sequence.split(',')
        for segment in segments:
            ref_c_id, ref_r_range = segment.split('/')
            s, e = ref_r_range.split('-')
            s, e = int(s), int(e)
            chain_array += ref_atom_array[
                (ref_atom_array.chain_id == ref_c_id) &
                np.isin(ref_atom_array.res_id, range(s, e+1))
            ]
        chain_array.set_annotation('is_antigen_chain', np.ones(len(chain_array), dtype=np.bool_))

    # reindex residue index
    r_starts = struc.get_residue_starts(chain_array, add_exclusive_stop=True)
    for r_id, (r_s, r_e) in enumerate(zip(r_starts[:-1], r_starts[1:])):
        chain_array.res_id[r_s:r_e] = r_id + 1
    if chain_type != 'ligand':
        chain_sequence = struc.to_sequence(chain_array)
        chain_array = ccd.add_inter_residue_bonds(chain_array)
    else:
        chain_sequence = 'LIGAND'
        chain_array = add_ligand_fake_bond(chain_array)
    
    return chain_array, str(chain_sequence[0][0])


class SampleDictToFeatures:
    def __init__(
        self,
        single_sample_dict,
        data_condition,
        use_msa,
    ):
        self.single_sample_dict = single_sample_dict
        self.motif_scaffolding = single_sample_dict.get('motif_scaffolding', False)
        self.center_method = single_sample_dict.get('center_method', '')
        self.data_condition = data_condition
        self.use_msa = use_msa

    def get_entity_poly_type(self, entity_identity) -> dict[str, str]:
        """
        Get the entity type for each entity.

        Allowed Value for "_entity_poly.type":
        · cyclic-pseudo-peptide
        · other
        · peptide nucleic acid
        · polydeoxyribonucleotide
        · polydeoxyribonucleotide/polyribonucleotide hybrid
        · polypeptide(D)
        · polypeptide(L)
        · polyribonucleotide

        Returns:
            dict[str, str]: a dict of polymer entity id to entity type.
        """
        entity_type_mapping_dict = {
            "proteinChain": "polypeptide(L)",
            "dnaChain": "polydeoxyribonucleotide",
            "rnaChain": "polyribonucleotide",
        }
        entity_poly_type = {}
        for _, value in entity_identity.items():
            if entity_type := entity_type_mapping_dict.get(value['chain_type']):
                entity_poly_type[str(value['entity_id'])] = entity_type 
        return entity_poly_type
    
    def build_full_atom_array(self) -> AtomArray:
        """
        By assembling the AtomArray of each entity, a complete AtomArray is created.

        Returns:
            AtomArray: Biotite Atom array.
        """
        ref_file = self.single_sample_dict['antigen']
        if ref_file.endswith('.cif'):
            cif_file = pdbx.CIFFile.read(ref_file)
            ref_atom_array = pdbx.get_structure(
                cif_file,
                include_bonds=True,
                model=1,
                extra_fields=["charge"],
            )
        elif ref_file.endswith('.pdb'):
            pdb_file = pdb.PDBFile.read(ref_file)
            ref_atom_array = pdb.get_structure(
                pdb_file,
                include_bonds=True,
                model=1,
                extra_fields=["charge"],
            )
        else:
            logger.warning('No ref_file was used. Please check if using free backbone design')
            ref_atom_array = get_backbone_atom_array('ligand')
        
        # UNK check
        r_starts = struc.get_residue_starts(ref_atom_array, add_exclusive_stop=True)
        for r_s, r_e in zip(r_starts[:-1], r_starts[1:]):
            r = ref_atom_array[r_s:r_e]
            if (r.res_name[0] == 'UNK') & (~self.motif_scaffolding):
                assert np.isin(r.atom_name, ['N', 'CA', 'C', 'O', 'CB']).all(), "UNK is not standard"
            elif (r.res_name[0] == 'UNK') & (self.motif_scaffolding):
                assert np.isin(r.atom_name, ['N', 'CA', 'C', 'O']).all(), "UNK only contains N, CA, C, O in motif scaffolding"
        
        # filter
        filter_functions = [
            Filter.remove_water,
            Filter.remove_hydrogens,
            MMCIFParser.fix_arginine,
            Filter.remove_element_X,
        ]
        for func in filter_functions:
            ref_atom_array = func(ref_atom_array)

        ref_atom_array.set_annotation('condition_token_mask', np.ones(len(ref_atom_array), dtype=np.bool_))
        ref_atom_array.set_annotation('is_hotspot_residue', np.zeros(len(ref_atom_array), dtype=np.bool_))
        if condition_atom := self.single_sample_dict.get('condition_atom', {}):
            for c_r, atom_names in condition_atom.items():
                c_id, r_id = c_r.split('/')
                ref_atom_array.condition_token_mask[
                    (ref_atom_array.chain_id == c_id) &
                    (ref_atom_array.res_id == int(r_id)) &
                    (~np.isin(ref_atom_array.atom_name, atom_names))
                ] = False
        if hotspot := self.single_sample_dict.get('hotspot', ''):
            for c_r in hotspot.split(','):
                c_id, r_id = c_r.split('/')
                ref_atom_array.is_hotspot_residue[
                    (ref_atom_array.chain_id == c_id) &
                    (ref_atom_array.res_id == int(r_id))
                ] = True

        atom_array = None
        sequence4msa = []
        entity_identity = defaultdict(dict)
        entity_id = 1
        for c_id, ch in enumerate(self.single_sample_dict['chains']):
            chain_type, sequence, im, length = ch['chain_type'], ch['sequence'], ch["im"], ch.get("length", None)
            chain_atom_array, chain_sequence = build_chain_atom_array(
                chain_sequence=sequence,
                chain_type=chain_type,
                im=im,
                chain_length=length,
                ref_atom_array=ref_atom_array
            )
            asym_id_str = int_to_letters(c_id + 1)
            chain_id = [asym_id_str] * len(chain_atom_array)
            chain_atom_array.set_annotation("label_asym_id", chain_id)
            chain_atom_array.set_annotation("auth_asym_id", chain_id)
            chain_atom_array.set_annotation("chain_id", chain_id)
            chain_atom_array.set_annotation("label_seq_id", chain_atom_array.res_id)

            if chain_sequence in entity_identity.keys():
                chain_atom_array.set_annotation("label_entity_id", [str(entity_identity[chain_sequence]['entity_id'])] * len(chain_atom_array))
                entity_identity[chain_sequence]['count'] += 1
                chain_atom_array.set_annotation("copy_id", [entity_identity[chain_sequence]['count']] * len(chain_atom_array))
            else:
                chain_atom_array.set_annotation("label_entity_id", [str(entity_id)] * len(chain_atom_array))
                chain_atom_array.set_annotation("copy_id", [1] * len(chain_atom_array))
                entity_identity[chain_sequence] = {
                    'chain_type': chain_type,
                    'entity_id': entity_id,
                    'count': 1
                }
                entity_id += 1

            if chain_type == 'proteinChain' and 'msa' in ch.keys():
                sequence4msa.append({
                    chain_type: {
                        "sequence": chain_sequence,
                        "msa": ch['msa'],
                        "chain_type": chain_type
                    }
                })
            if atom_array is None:
                atom_array = chain_atom_array
            else:
                atom_array += chain_atom_array

        atom_array.set_annotation('ref_pos', atom_array.coord)
        atom_array.set_annotation('ref_mask', np.ones(len(atom_array), dtype=np.int_))
        atom_array.set_annotation('alt_atom_id', atom_array.atom_name)
        atom_array.set_annotation('pdbx_component_atom_id', atom_array.atom_name)
        atom_array.set_annotation('leaving_atom_flag', np.zeros(len(atom_array), dtype=np.bool_))
        self.entity_poly_type = self.get_entity_poly_type(entity_identity)

        # center method problem
        if self.center_method == 'hotspot_center':
            assert sum(atom_array.is_hotspot_residue) > 0, 'No hotspot is provided. Please check if hotspot has been given'
            center = np.mean(atom_array.coord[(atom_array.is_hotspot_residue)&(atom_array.condition_token_mask)], axis=0)
            atom_array.coord = atom_array.coord - center
            logger.info('Use hotspot center in inference')
        elif self.center_method == 'global_center':
            center =  np.mean(ref_atom_array.coord, axis=0)
            atom_array.coord = atom_array.coord - center
            logger.info('Use global center in inference')
        elif self.center_method == 'usr_provide_center':
            assert 'usr_provide_center' in self.single_sample_dict.keys(), 'Center method is usr_provide_center, but no center is provided in the input json'
            center = np.array(self.single_sample_dict['usr_provide_center'])
            atom_array.coord = atom_array.coord - center
            logger.info(f'Use usr provide center {center} in inference')
        else:
            logger.info('No center method is used in inference')

        # for msa search
        self.sequence4msa = sequence4msa
        
        return atom_array

    @staticmethod
    def get_a_bond_atom(
        atom_array: AtomArray,
        entity_id: int,
        position: int,
        atom_name: str,
        copy_id: int = None,
    ) -> np.ndarray:
        """
        Get the atom index of a bond atom.

        Args:
            atom_array (AtomArray): Biotite Atom array.
            entity_id (int): Entity id.
            position (int): Residue index of the atom.
            atom_name (str): Atom name.
            copy_id (copy_id): A asym chain id in N copies of an entity.

        Returns:
            np.ndarray: Array of indices for specified atoms on each asym chain.
        """
        entity_mask = atom_array.label_entity_id == str(entity_id)
        position_mask = atom_array.res_id == int(position)
        atom_name_mask = atom_array.atom_name == str(atom_name)

        if copy_id is not None:
            copy_mask = atom_array.copy_id == int(copy_id)
            mask = entity_mask & position_mask & atom_name_mask & copy_mask
        else:
            mask = entity_mask & position_mask & atom_name_mask
        atom_indices = np.where(mask)[0]
        return atom_indices

    @staticmethod
    def add_atom_array_attributes(
        atom_array: AtomArray, entity_poly_type: dict[str, str], motif_scaffolding: bool
    ) -> AtomArray:
        """
        Add attributes to the Biotite AtomArray.

        Args:
            atom_array (AtomArray): Biotite Atom array.
            entity_poly_type (dict[str, str]): a dict of polymer entity id to entity type.

        Returns:
            AtomArray: Biotite Atom array with attributes added.
        """
        atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array, motif_scaffolding)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
        atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
        atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(
            atom_array, check_final_equiv=False
        )
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)
        atom_array.set_annotation("is_resolved", np.ones(len(atom_array), dtype=np.bool_))

        return atom_array

    @staticmethod
    def mse_to_met(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI chapter 2.1
        MSE residues are converted to MET residues.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object after converted MSE to MET.
        """
        mse = atom_array.res_name == "MSE"
        se = mse & (atom_array.atom_name == "SE")
        atom_array.atom_name[se] = "SD"
        atom_array.element[se] = "S"
        atom_array.res_name[mse] = "MET"
        atom_array.hetero[mse] = False
        return atom_array

    def get_atom_array(self) -> AtomArray:
        """
        Create a Biotite AtomArray and add attributes from the input dict.

        Returns:
            AtomArray: Biotite Atom array.
        """
        atom_array = self.build_full_atom_array()
        atom_array = self.mse_to_met(atom_array)
        atom_array = self.add_atom_array_attributes(atom_array, self.entity_poly_type, self.motif_scaffolding)
        return atom_array

    def get_feature_and_label(self) -> tuple[OFeatureData, OLabelData, AtomArray, TokenArray]:
        """
        Generates a feature dictionary from the input sample dictionary.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A TokenArray object.
        """
        atom_array = self.get_atom_array()

        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        featurizer = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            data_condition=self.data_condition,
            inference_mode=True,
        )
        features_dict = featurizer.get_all_input_features()
        feature_data = OFeatureData.from_feature_dict(features_dict)

        default_feat_shape_dict, _ = get_data_shape_dict(
            num_token=feature_data.num_token,
            num_atom=feature_data.num_atom,
            num_msa=feature_data.default_num_msa,
            num_templ=feature_data.default_num_templ,
            num_pocket=feature_data.default_num_pocket,
        )

        # prepare MSA features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=self.sequence4msa,
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
            if self.use_msa
            else {}
        )
        featurizer.msa_features = msa_features
        msa_features = featurizer.get_msa_features(
            features_dict=features_dict,
            feat_shape=default_feat_shape_dict
        )
        feature_data.update(msa_features)

        # prepare template features
        template_features = featurizer.get_template_features(
            feat_shape=default_feat_shape_dict
        )
        feature_data.update(template_features)

        # prepare hotspot features
        hotspot_features = featurizer.get_hotspot_features()
        feature_data.update(hotspot_features)

        # need label to fix condition atoms
        labels_dict = featurizer.get_labels()
        label_data = OLabelData.from_label_dict(labels_dict)

        return feature_data, label_data, atom_array, token_array