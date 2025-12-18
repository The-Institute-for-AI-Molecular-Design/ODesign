import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  
from ..src.datasets.featurizer import UniTokenizer, Featurize
import os.path as osp
import gzip
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple, Union, Sequence
from evaluation_tools.design_interface import MInterface
# from model_interface import MInterface
from sklearn.metrics import f1_score
import math
from .inference_utils import (   
    _parse_structure_path,          
    build_cif_id_maps,              
    extract_chain_seq_and_backbone, 
)
from collections import Counter
import re
import torch.nn.functional as F
from collections import defaultdict, OrderedDict

def id_to_char_global(i: int, kind: str, tok) -> str:

    if 0 <= i < 20:
        return tok.alphabet_protein[i]
    if 20 <= i < 25:
        base = tok.nuc5[i-20]  # A,C,G,T,U
        if kind == 'dna' and base == 'U': base = 'T'
        if kind == 'rna' and base == 'T': base = 'U'
        return base
    return 'X'

def diagnose_vocab_distribution(probs_cpu: torch.Tensor, kind: str, tok) -> None:
    """
    probs_cpu: [L, V] on CPU after softmax
    kind: 'dna' | 'rna' | 'protein'
    """
    L, V = probs_cpu.shape
    if kind == 'dna':
        allow = torch.tensor(tok.dna_ids)   # [20,21,22,23]
    elif kind == 'rna':
        allow = torch.tensor(tok.rna_ids)   # [20,21,22,24]
    else:
        allow = torch.arange(0, min(20, V)) 

    allow = allow[allow < V]
    if allow.numel() == 0:
        allow = torch.arange(0, V)

    mass_allowed = probs_cpu[:, allow].sum(dim=1)           # [L]
    mass_protein = probs_cpu[:, :min(20, V)].sum(dim=1)     # [L]
    top1 = probs_cpu.max(dim=-1).values                     # [L]
    ent = -(probs_cpu.clamp_min(1e-12).log() * probs_cpu).sum(dim=-1) / math.log(2.0)  # bits

    argmax_ids = probs_cpu.argmax(dim=-1)                   # [L]
    outside_frac = float((~torch.isin(argmax_ids, allow)).float().mean().item())

    cnt = Counter(int(i) for i in argmax_ids.tolist())
    allow_hist = {id_to_char_global(int(i), kind, tok): cnt.get(int(i), 0) for i in allow.tolist()}

    mass_per_token = probs_cpu.sum(dim=0) / L
    mass_dict = {id_to_char_global(i, kind, tok): float(mass_per_token[i].item()) for i in range(min(V, 25))}

    print("\n[Diag/Vocab]")
    print(f"- Allowed mass (mean/min): {float(mass_allowed.mean()):.4f} / {float(mass_allowed.min()):.4f}")
    print(f"- Protein mass leak (mean): {float(mass_protein.mean()):.4f}")
    print(f"- Argmax outside allowed: {outside_frac*100:.2f}%")
    print(f"- Top1 prob mean±std: {float(top1.mean()):.4f} ± {float(top1.std()):.4f}")
    print(f"- Entropy bits mean±std: {float(ent.mean()):.3f} ± {float(ent.std()):.3f}")
    print(f"- Argmax histogram in allowed: {allow_hist}")


def gzip_open(filename, *args, **kwargs):
    if args and "t" in args[0]:
        args = (args[0].replace("t", ""), ) + args[1:]
    if isinstance(filename, str):
        return gzip.open(filename, *args, **kwargs)
    else:
        return gzip.GzipFile(filename, *args, **kwargs)


def save_fasta(seqs, names, pred_fasta_path='pred.fasta'):
    with open(pred_fasta_path, 'w') as f:
        for seq, name in zip(seqs, names):
            f.write(f">{name}\n{seq}\n")

def reload_model(data_name, checkpoint_path, configs, device: str | None = None):
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    dev = torch.device(device)

    model = MInterface.load_from_checkpoint(
        dataset=data_name,
        checkpoint_path=checkpoint_path,
        configs=configs,
        map_location=dev,
    ).to(dev).eval()

    try:
        any_param = next(model.parameters())
        print(f"[reload_model] device={dev} first_param_on={any_param.device}")
    except StopIteration:
        print(f"[reload_model] device={dev} (no params?)")
    return model, dev

# ---------------- mmCIF ----------------

try:
    from Bio.PDB import StructureBuilder, Structure, Model, Chain, Residue, Atom
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.mmcifio import MMCIFIO
except Exception as _e:
    MMCIFParser = None
    MMCIFIO = None

# ------------------ basic mapping ------------------
from biotite.structure import AtomArray

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_AA1_TO_AA3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


NA3_TO_1 = {
    # DNA
    "DA": "A", "DT": "T", "DG": "G", "DC": "C",
    "DAD": "A", "DTY": "T", "DGU": "G", "DCY": "C",
    # RNA 
    "A": "A", "U": "U", "G": "G", "C": "C",
    "RA": "A", "RU": "U", "RG": "G", "RC": "C",
    # 
    "ADE": "A", "GUA": "G", "CYT": "C", "URA": "U",
}


def _to_numpy_coords(coord_tensor: torch.Tensor) -> np.ndarray:
    if isinstance(coord_tensor, torch.Tensor):
        return coord_tensor.detach().cpu().numpy().astype(np.float32)
    return np.asarray(coord_tensor, dtype=np.float32)


def _na1_to_compid(base: str, kind: str) -> str:
    b = base.upper()
    if kind == 'rna':
        if b in ('A','U','G','C'):
            return b
        return None
    else:
        mapping = {'A':'DA', 'T':'DT', 'G':'DG', 'C':'DC'}
        return mapping.get(b, None)


def _ensure_biopython():
    if MMCIFParser is None or MMCIFIO is None:
        raise RuntimeError("Biopython (Bio.PDB) not available. Please install biopython.")

def _load_structure_from_cif(cif_path: str):
    _ensure_biopython()
    parser = MMCIFParser(QUIET=True)
    struct_id = os.path.basename(cif_path)
    return parser.get_structure(struct_id, cif_path)

def _copy_single_chain(structure, chain_id: str):

    from Bio.PDB import Structure
    new_s = Structure.Structure("subset")
    model0 = list(structure.get_models())[0]

    from Bio.PDB import Model, Chain, Residue, Atom
    new_m = Model.Model(0)


    for ch in model0:
        if ch.id == chain_id:
            new_ch = Chain.Chain(chain_id)
            for res in ch:
                
                new_res = Residue.Residue(res.id, res.resname, res.segid)
                for atom in res:
                    new_atom = Atom.Atom(
                        atom.get_name(), atom.get_coord(), atom.get_bfactor(),
                        atom.get_occupancy(), atom.get_altloc(), atom.get_fullname(),
                        atom.get_serial_number(), element=atom.element
                    )
                    new_res.add(new_atom)
                new_ch.add(new_res)
            new_m.add(new_ch)
            break
    new_s.add(new_m)
    return new_s

def _mutate_chain_sequence_inplace(structure, chain_id: str, new_seq: str, kind: str) -> Tuple[int,int]:
    new_seq = (new_seq or "").upper()
    if not new_seq:
        return (0, 0)

    model0 = list(structure.get_models())[0]
    chain = None
    for ch in model0:
        if ch.id == chain_id:
            chain = ch
            break
    if chain is None:
        return (0, 0)


    candidates = []
    for res in chain:
        hetflag, resseq, icode = res.id
        if hetflag.strip() != "":
            continue
        candidates.append(res)

    L = min(len(candidates), len(new_seq))
    applied = 0
    for i in range(L):
        res = candidates[i]
        target = new_seq[i]
        if kind == 'protein':
            aa3 = _AA1_TO_AA3.get(target, None)
            if aa3:
                res.resname = aa3
                applied += 1
        else:
            comp = _na1_to_compid(target, kind)
            if comp:
                res.resname = comp
                applied += 1
    return (applied, L)


def save_single_chain_cif(cif_src_path: str, chain_auth_id: str, out_path: str) -> None:

    _ensure_biopython()
    struct = _load_structure_from_cif(cif_src_path)
    sub_s = _copy_single_chain(struct, chain_auth_id)  
    io = MMCIFIO()
    io.set_structure(sub_s)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.save(out_path)

def save_sequence_replaced_cif(cif_src_path: str,
                               chain_auth_id: str,
                               new_seq: str,
                               kind: str,
                               out_path: str) -> Tuple[int,int]:
    _ensure_biopython()

    full_struct = _load_structure_from_cif(cif_src_path)
    sub_struct = _copy_single_chain(full_struct, chain_auth_id)


    applied, aligned = _mutate_chain_sequence_inplace(
        sub_struct, chain_auth_id, new_seq, kind
    )


    io = MMCIFIO()
    io.set_structure(sub_struct)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    io.save(out_path)
    return (applied, aligned)


_CHAIN_AUTH_RE = re.compile(r"_auth([A-Za-z0-9]+)")

def parse_auth_from_title(title: str, fallback: Optional[str] = None) -> Optional[str]:
    m = _CHAIN_AUTH_RE.search(title or "")
    if m:
        return m.group(1)
    return fallback

def _decide_kind(one_seq: str, prefer: str) -> str:

    one = (one_seq or "").upper()
    if 'T' in one: return 'dna'
    if 'U' in one: return 'rna'
    return prefer



# ---------------- Inference & Metrics ----------------
def allowed_indices(kind: str, V: int) -> torch.Tensor:
    kind = str(kind).lower()
    if kind == 'dna':
        cand = torch.tensor([20, 21, 22, 23], dtype=torch.long)  # A,C,G,T
    elif kind == 'rna':
        cand = torch.tensor([20, 21, 22, 24], dtype=torch.long)  # A,C,G,U
    elif kind == 'protein':
        cand = torch.arange(0, min(20, V), dtype=torch.long)
    else:
        cand = torch.arange(0, V, dtype=torch.long)
    return cand[cand < V]

def beam_search_logp(logp_mat, allow_idx, beam_w, len_norm=True, alpha=0.6,
                     diversity_groups=1, gamma=0.0):
    L_, V_ = logp_mat.shape
    K_ = allow_idx.numel()
    G = max(1, int(diversity_groups))
    group_size = (beam_w + G - 1) // G
    groups = [[([], 0.0)] for _ in range(G)]
    for t in range(L_):
        chosen_per_group = []
        for g in range(G):
            beams = groups[g]
            row = logp_mat[t, allow_idx].clone()  # [K_]
            if gamma > 0 and len(chosen_per_group) > 0:
                used = torch.zeros(K_)
                for prev in chosen_per_group:
                    used[prev] += 1
                row = row - gamma * used
            cand = []
            for seq, s in beams:
                for k in range(K_):
                    cand.append((seq + [int(allow_idx[k])], s + float(row[k])))
            cand.sort(key=lambda x: x[1], reverse=True)
            groups[g] = cand[:group_size]
            top_tokens = []
            for ids, _ in groups[g]:
                pos = (allow_idx == ids[-1]).nonzero(as_tuple=False)
                top_tokens.append(int(pos[0, 0]) if pos.numel() > 0 else 0)
            chosen_per_group.append(top_tokens)
    beams_all = [item for grp in groups for item in grp]
    if len_norm:
        beams_all = [(ids, s / max(1, len(ids))) for ids, s in beams_all]
    elif alpha > 0:
        def lp_len(m): return ((5 + m) ** alpha) / ((5 + 1) ** alpha)
        beams_all = [(ids, s / lp_len(len(ids))) for ids, s in beams_all]
    beams_all.sort(key=lambda x: x[1], reverse=True)
    return beams_all[:beam_w]

def labels_4class(seq_or_ids, kind: str, tok) -> List[int]:
    kind = str(kind).lower()
    if isinstance(seq_or_ids, (list, tuple)) and len(seq_or_ids) > 0 and isinstance(seq_or_ids[0], int):
        ids = list(seq_or_ids)
    else:
        ids = tok.encode(str(seq_or_ids), kind=kind)
    if kind == 'protein':
        return [t if 0 <= t < 20 else 19 for t in ids]
    elif kind in ('dna', 'rna'):
        A, C, G = tok.nuc_to_id['A'], tok.nuc_to_id['C'], tok.nuc_to_id['G']
        if kind == 'dna':
            T = tok.nuc_to_id['T']
            id2cls = {A:0, C:1, G:2, T:3}
        else:
            U = tok.nuc_to_id['U']
            id2cls = {A:0, C:1, G:2, U:3}
        return [id2cls.get(t, 0) for t in ids]
    else:
        raise ValueError(f"Unknown kind: {kind}")

def _move_to(x, device):
    if isinstance(x, dict):
        return {k: _move_to(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_move_to(v, device) for v in x)
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x

# ==== helpers for ligand ====

def ligand_allowed_indices(tok, V: int,
                           allow_rare: bool = True,
                           allow_unk: bool = False,
                           allow_mask: bool = False) -> torch.Tensor:
    allowed = []
    inv = getattr(tok, "id_to_token", None)
    if inv is not None:
        for i, token in enumerate(inv):
            if i >= V: break
            if token == getattr(tok, "MASK_TOKEN", "*") and not allow_mask:
                continue
            if token == getattr(tok, "UNK_TOKEN", "UNK") and not allow_unk:
                continue
            if token == getattr(tok, "RARE_TOKEN", "RARE") and not allow_rare:
                continue
            allowed.append(i)
    else:
        allowed = list(range(min(V, getattr(tok, "vocab_size", V))))
    if not allowed:
        allowed = list(range(min(V, getattr(tok, "vocab_size", V))))
    return torch.tensor(allowed, dtype=torch.long)

def decode_ligand_ids(ids, tok, sep: str = " ") -> str:
    inv = getattr(tok, "id_to_token", None)
    if inv is None:
        return sep.join(str(i) for i in ids)
    out = []
    for i in ids:
        if 0 <= i < len(inv):
            out.append(inv[i])
        else:
            out.append(getattr(tok, "UNK_TOKEN", "UNK"))
    return sep.join(out)


def parse_invfold(
    atom_array: AtomArray,
    pred_output,
    design_modality: str,           # 'protein' / 'rna' / 'dna' / 'ligand'
    sample_name: str,
) -> List[List[Dict[str, Any]]]:

    prefer = design_modality.strip().lower()
    assert prefer in ("protein", "rna", "dna", "ligand")

    coord_tensor_all = pred_output.coordinate
    assert coord_tensor_all.ndim == 3, f"expect coordinate [N_struct, N_atom, 3], got {coord_tensor_all.shape}"
    n_struct, n_atom = coord_tensor_all.shape[0], coord_tensor_all.shape[1]

    atom_names = np.asarray(atom_array.atom_name).astype(str)
    chain_ids_atom = np.asarray(atom_array.chain_id).astype(str)
    res_ids_atom = np.asarray(atom_array.res_id)

    if hasattr(atom_array, "cano_seq_resname"):
        res3 = np.asarray(atom_array.cano_seq_resname).astype(str)
    else:
        res3 = np.asarray(atom_array.res_name).astype(str)

    if hasattr(atom_array, "condition_token_mask"):
        cond_mask_atom = np.asarray(atom_array.condition_token_mask).astype(bool)
    else:
        cond_mask_atom = np.zeros(n_atom, dtype=bool)

    hetero_atom = None
    elem_atom = None
    if prefer == "ligand":
        if hasattr(atom_array, "hetero"):
            hetero_atom = np.asarray(atom_array.hetero).astype(bool)
        else:
            hetero_atom = np.ones(n_atom, dtype=bool)
        if hasattr(atom_array, "element"):
            elem_atom = np.asarray(atom_array.element).astype(str)
        else:
            elem_atom = np.array([name[0] for name in atom_names], dtype=str)

    if n_atom != len(atom_array):
        raise ValueError(
            f"coordinate.shape[1]={n_atom} does not match {len(atom_array)} "
        )

    # 
    residue_atoms: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, (ch, rid) in enumerate(zip(chain_ids_atom, res_ids_atom)):
        residue_atoms[(ch, int(rid))].append(idx)

    # 
    sorted_keys = sorted(residue_atoms.keys(), key=lambda x: (x[0], x[1]))
    chains: "OrderedDict[str, list[tuple[str, int]]]" = OrderedDict()
    for ch, rid in sorted_keys:
        chains.setdefault(ch, []).append((ch, rid))

    samples_all: List[List[Dict[str, Any]]] = []

    # invfold 
    for struct_idx in range(n_struct):
        coord_tensor = coord_tensor_all[struct_idx]   # [N_atom, 3]
        coords_np = _to_numpy_coords(coord_tensor)

        def get_coord_for_atom(atom_idx_list: list[int], target_name: str) -> np.ndarray:
            t = target_name.upper()
            for idx in atom_idx_list:
                if atom_names[idx].upper() == t:
                    return coords_np[idx]
            return np.full((3,), np.nan, dtype=np.float32)

        def get_na_base_n_coord(atom_idx_list: list[int]) -> np.ndarray:

            for idx in atom_idx_list:
                if atom_names[idx].upper() == "N9":
                    return coords_np[idx]

            for idx in atom_idx_list:
                if atom_names[idx].upper() == "N1":
                    return coords_np[idx]

            for idx in atom_idx_list:
                if atom_names[idx].upper().startswith("N"):
                    return coords_np[idx]

            return np.full((3,), np.nan, dtype=np.float32)



        struct_samples: List[Dict[str, Any]] = []

        # ---------- ligand-only ----------
        if prefer == "ligand":
            for ch, res_keys in chains.items():
                for (ch_id, rid) in res_keys:
                    atom_idx_list = residue_atoms[(ch_id, rid)]

                    is_het_res = np.all(hetero_atom[atom_idx_list])
                    if not is_het_res:
                        continue

                    res_cond = cond_mask_atom[atom_idx_list]
                    if np.all(res_cond):
                        continue

                    elems = [elem_atom[i].upper() for i in atom_idx_list]
                    norm_elems = [e if e else "UNK" for e in elems]
                    lig_xyz = coords_np[atom_idx_list]  # (N_lig_atom, 3)
                    if lig_xyz.shape[0] == 0:
                        continue

                    L = lig_xyz.shape[0]
                    lig_seq_str = " ".join(norm_elems)

                    res_ids = np.arange(L, dtype=int)
                    chain_ids = np.array([str(ch_id)] * L, dtype=str)
                    res_atom_indices = [np.asarray([atom_idx_list[i]], dtype=int) for i in range(L)]

                    sample = {
                        "title": f"{sample_name}_s{struct_idx}_LIG_chain{ch_id}_res{int(rid)}",
                        "type": "ligand",
                        "seq": lig_seq_str,   
                        "ligand": {
                            "elements": norm_elems,
                            "coords": lig_xyz.tolist(),
                            "chain_mask": [[1.0] * L],
                            "chain_encoding": [[1.0] * L],
                        },
                        # res_ids / chain_ids / res_atom_indices
                        "res_ids": res_ids,
                        "chain_ids": chain_ids,
                        "res_atom_indices": res_atom_indices,
                        "design_mask": np.ones(L, dtype=bool),
                        "chain_id": str(ch_id),
                        "res_id": int(rid),
                        "ligand_seq": lig_seq_str,
                    }
                    struct_samples.append(sample)

            samples_all.append(struct_samples)
            continue  

        for ch, res_keys in chains.items():
            seq_chars: list[str] = []
            design_mask: list[bool] = []
            res_id_list: list[int] = []
            chain_id_list: list[str] = []

            if prefer == "protein":
                N_list, CA_list, C_list, O_list = [], [], [], []
                res_atom_indices_list: list[list[int]] = []   

                for (ch_id, rid) in res_keys:
                    atom_idx_list = residue_atoms[(ch_id, rid)]

                    res_cond = cond_mask_atom[atom_idx_list] 
                    if np.all(res_cond):
                        continue  

                    name3 = res3[atom_idx_list[0]].upper()
                    aa = AA3_TO_1.get(name3, "X")
                    seq_chars.append(aa)

                    N_list.append(get_coord_for_atom(atom_idx_list, "N"))
                    CA_list.append(get_coord_for_atom(atom_idx_list, "CA"))
                    C_list.append(get_coord_for_atom(atom_idx_list, "C"))
                    O_list.append(get_coord_for_atom(atom_idx_list, "O"))

                    design_mask.append(True)  
                    res_id_list.append(int(rid))
                    chain_id_list.append(str(ch_id))
                    res_atom_indices_list.append(atom_idx_list)  

                L = len(seq_chars)
                if L == 0:
                    continue

                seq_clean = "".join(seq_chars)
                N_arr  = np.stack(N_list,  axis=0).astype(np.float32)
                CA_arr = np.stack(CA_list, axis=0).astype(np.float32)
                C_arr  = np.stack(C_list,  axis=0).astype(np.float32)
                O_arr  = np.stack(O_list,  axis=0).astype(np.float32)

                sample = {
                    "title": f"{sample_name}_s{struct_idx}_chain{ch}",
                    "type":  "protein",
                    "seq":   seq_clean,
                    "N":     N_arr,
                    "CA":    CA_arr,
                    "C":     C_arr,
                    "O":     O_arr,
                    "chain_mask":     np.ones(L, dtype=np.float32),
                    "chain_encoding": np.ones(L, dtype=np.float32),
                    "design_mask":    np.asarray(design_mask, dtype=bool),
                    "res_ids":        np.asarray(res_id_list, dtype=int),
                    "chain_ids":      np.asarray(chain_id_list),
                    "res_atom_indices": [np.asarray(idxs, dtype=int) for idxs in res_atom_indices_list],
                }
                struct_samples.append(sample)

            else:
                P_list, O5_list, C5_list, C4_list, C3_list, O3_list, N_list_na = \
                    [], [], [], [], [], [], []
                res_atom_indices_list: list[list[int]] = []   

                for (ch_id, rid) in res_keys:
                    atom_idx_list = residue_atoms[(ch_id, rid)]

                    res_cond = cond_mask_atom[atom_idx_list]
                    if np.all(res_cond):
                        continue  

                    name3 = res3[atom_idx_list[0]].upper()
                    base = NA3_TO_1.get(name3, "N")  # A/T/G/C/U/N

                    if prefer == "dna":
                        ch1 = base.upper()
                        if ch1 == "U":
                            ch1 = "T"
                        if ch1 not in ("A", "T", "G", "C", "N"):
                            ch1 = "N"
                    else:  # rna
                        ch1 = base.upper()
                        if ch1 == "T":
                            ch1 = "U"
                        if ch1 not in ("A", "U", "G", "C", "N"):
                            ch1 = "N"

                    seq_chars.append(ch1)

                    P_list.append(  get_coord_for_atom(atom_idx_list, "P")   )
                    O5_list.append( get_coord_for_atom(atom_idx_list, "O5'") )
                    C5_list.append( get_coord_for_atom(atom_idx_list, "C5'") )
                    C4_list.append( get_coord_for_atom(atom_idx_list, "C4'") )
                    C3_list.append( get_coord_for_atom(atom_idx_list, "C3'") )
                    O3_list.append( get_coord_for_atom(atom_idx_list, "O3'") )
                    N_list_na.append( get_na_base_n_coord(atom_idx_list) )

                    design_mask.append(True)
                    res_id_list.append(int(rid))
                    chain_id_list.append(str(ch_id))
                    res_atom_indices_list.append(atom_idx_list)

                L = len(seq_chars)
                if L == 0:
                    continue  

                seq_clean = "".join(seq_chars)
                P_arr  = np.stack(P_list,  axis=0).astype(np.float32)
                O5_arr = np.stack(O5_list, axis=0).astype(np.float32)
                C5_arr = np.stack(C5_list, axis=0).astype(np.float32)
                C4_arr = np.stack(C4_list, axis=0).astype(np.float32)
                C3_arr = np.stack(C3_list, axis=0).astype(np.float32)
                O3_arr = np.stack(O3_list, axis=0).astype(np.float32)
                N_arr  = np.stack(N_list_na, axis=0).astype(np.float32)

                kind = "dna" if prefer == "dna" else "rna"
                sample = {
                    "title": f"{sample_name}_s{struct_idx}_{kind}_chain{ch}",
                    "type":  kind,
                    "seq":   seq_clean,
                    "P":     P_arr,
                    "O5":    O5_arr,
                    "C5":    C5_arr,
                    "C4":    C4_arr,
                    "C3":    C3_arr,
                    "O3":    O3_arr,
                    "N":     N_arr,
                    "chain_mask":     np.ones(L, dtype=np.float32),
                    "chain_encoding": np.ones(L, dtype=np.float32),
                    "design_mask":    np.asarray(design_mask, dtype=bool),
                    "res_ids":        np.asarray(res_id_list, dtype=int),
                    "chain_ids":      np.asarray(chain_id_list),
                    "res_atom_indices": [np.asarray(idxs, dtype=int) for idxs in res_atom_indices_list],
                }
                struct_samples.append(sample)

        samples_all.append(struct_samples)

    return samples_all



def inference(model, sample_input, design_modality,
              topk: int = 5, temp: float = 1.0, use_beam: bool = False,
              device=None):

    kind = str(design_modality).lower()
    # 1) featurizer/tokenizer
    if kind == 'ligand':
        from src.model.modules.invfold.src.datasets.featurizer_ligand import Featurize_Ligand, LigandTokenizer
        feat = Featurize_Ligand()
        tok = LigandTokenizer()
        batch = feat.featurize([sample_input['ligand']])
    else:
        from src.model.modules.invfold.src.datasets.featurizer import Featurize, UniTokenizer
        feat = Featurize(mixed=True, augmentor=None)
        tok = UniTokenizer()
        batch = feat.featurize([sample_input])

    # 2) featurize
    # batch = feat.featurize([sample_input]['ligand'])
    if batch is None:
        raise RuntimeError("featurize defective??")

    # 3) 设备
    if device is None:
        try:
            device = next(model.model.parameters()).device
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = _move_to(batch, device)

    # 4) forward
    # model.eval()
    model.model.eval()
    with torch.no_grad():
        out = model.model(batch)
        log_probs = out['log_probs']         # [N_all, V]
        logits    = out['logits']            # [N_all, V]
        probs_cpu_all = torch.softmax(logits / temp, dim=-1).detach().cpu()
        logp_cpu_all  = log_probs.detach().cpu()

    if kind == 'ligand':
        mask = batch.get('ligand_mask', None)
        if mask is None:
            mask = batch['chain_mask']
        mask = mask.detach().float().cpu().view(-1)  # [N_all]
        idx = (mask > 0.5).nonzero(as_tuple=False).view(-1)  # [L_lig]
        if idx.numel() == 0:
            return [], [], "", torch.empty(0, logits.shape[-1]), {
                'per_candidate': [], 'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
                'best_idx': 0, 'best': {'Accuracy': 0.0, 'Macro_F1': 0.0}
            }

        probs_cpu = probs_cpu_all.index_select(0, idx)  # [L_lig, V]
        logp_cpu  = logp_cpu_all.index_select(0, idx)   # [L_lig, V]
    else:
        probs_cpu = probs_cpu_all
        logp_cpu  = logp_cpu_all

    L, V = probs_cpu.shape
    if kind == 'ligand':
        allow = ligand_allowed_indices(tok, V, allow_rare=True, allow_unk=False, allow_mask=False)
    else:
        allow = allowed_indices(kind, V)
        if allow.numel() == 0:
            allow = torch.arange(0, V, dtype=torch.long)

    pred_seqs, scores = [], []
    if not use_beam:
        sub = probs_cpu[:, allow]                 # [L, |allow|]
        ids_rel = sub.argmax(dim=-1)              # [L]
        ids_glb = allow[ids_rel].tolist()         # [L]
        if kind == 'ligand':
            seq = decode_ligand_ids(ids_glb, tok, sep=" ")
        else:
            seq = tok.decode(ids_glb, kind=kind)
        lp  = float(logp_cpu[torch.arange(L), torch.tensor(ids_glb)].sum())
        pred_seqs = [seq]; scores = [lp]
    else:
        beam_width = max(1, int(topk))
        beams = beam_search_logp(
            logp_cpu, allow, beam_width,
            len_norm=True, alpha=0.6,
            diversity_groups=1, gamma=0.0
        )
        for ids_glb, logp_sum in beams:
            if kind == 'ligand':
                seq = decode_ligand_ids(ids_glb, tok, sep=" ")
            else:
                seq = tok.decode(ids_glb, kind=kind)
            pred_seqs.append(seq)
            scores.append(float(logp_sum))


    if kind == 'ligand':
       
        true_seq = sample_input.get('ligand_seq') \
                   or " ".join(sample_input.get('ligand', {}).get('elements', [])) \
                   or ""
        true_tokens = (true_seq or "").split()
        if pred_seqs:
            L_common = min(len(true_tokens), *(len(s.split()) for s in pred_seqs))
        else:
            L_common = 0
        if L_common == 0:
            metrics = {
                'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
                'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
                'best_idx': 0,
                'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
            }
            return pred_seqs, scores, true_seq, probs_cpu, metrics

        true_tokens = true_tokens[:L_common]
        per_cand = []
        for seq in pred_seqs:
            pred_tokens = seq.split()[:L_common]
            acc = sum(1 for a, b in zip(true_tokens, pred_tokens) if a == b) / float(L_common)
            labels = sorted(set(true_tokens) | set(pred_tokens))
            try:
                macro_f1 = float(f1_score(true_tokens, pred_tokens, labels=labels, average='macro'))
            except Exception:
                macro_f1 = 0.0
            per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
        top1 = per_cand[0]
        best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
                                                            per_cand[i]['Macro_F1'],
                                                            scores[i]))
        best = per_cand[best_idx]
        metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
        return pred_seqs, scores, true_seq, probs_cpu, metrics

    # protein / rna / dna
    true_seq = sample_input['seq'][0] if isinstance(sample_input.get('seq'), list) \
               else sample_input.get('seq', '')
    L_common = min(len(true_seq), *(len(s) for s in pred_seqs)) if pred_seqs else 0
    if L_common == 0:
        metrics = {
            'per_candidate': [{'Accuracy': 0.0, 'Macro_F1': 0.0} for _ in pred_seqs],
            'top1': {'Accuracy': 0.0, 'Macro_F1': 0.0},
            'best_idx': 0,
            'best': {'Accuracy': 0.0, 'Macro_F1': 0.0},
        }
        return pred_seqs, scores, true_seq, probs_cpu, metrics
    true_seq_c = true_seq[:L_common]
    per_cand = []
    for seq in pred_seqs:
        seq_c = seq[:L_common]
        acc = sum(1 for a, b in zip(true_seq_c, seq_c) if a == b) / float(L_common)
        y_true = labels_4class(true_seq_c, kind, tok)
        y_pred = labels_4class(seq_c, kind, tok)
        try:
            macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
        except Exception:
            macro_f1 = 0.0
        per_cand.append({'Accuracy': acc, 'Macro_F1': macro_f1})
    top1 = per_cand[0]
    best_idx = max(range(len(per_cand)), key=lambda i: (per_cand[i]['Accuracy'],
                                                        per_cand[i]['Macro_F1'],
                                                        scores[i]))
    best = per_cand[best_idx]
    metrics = {'per_candidate': per_cand, 'top1': top1, 'best_idx': best_idx, 'best': best}
    return pred_seqs, scores, true_seq, probs_cpu, metrics




