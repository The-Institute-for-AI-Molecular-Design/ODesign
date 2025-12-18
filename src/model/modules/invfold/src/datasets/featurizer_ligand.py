
# -*- coding: utf-8 -*-
# src/datasets/featurizer_ligand.py

import torch
import numpy as np
from typing import List, Sequence, Dict, Any

from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum

from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from ..modules.if_module import *  # rbf()


# ---------------- Tokenizer ----------------
class LigandTokenizer:

    BASE_ELEMS = [
        "C","O","N","P","S","Fe","Os","F","Mg","Cl","Br","W","B","Co","I","Mo",
        "As","Be","Mn","Cu","Ta","V","Al","Ir","Hg","Se","Ni","Ru","D","Pt",
        "Ca","Re","Zn","Si"
    ]
    def __init__(self) -> None:
        self.elem_to_id = {e: i for i, e in enumerate(self.BASE_ELEMS)}
        self.RARE_TOKEN = "RARE"; self.UNK_TOKEN = "UNK"; self.MASK_TOKEN = "*"
        self.rare_id = len(self.elem_to_id)       # 34
        self.unk_id  = self.rare_id + 1           # 35
        self.mask_id = self.unk_id + 1            # 36
        self.id_to_token = list(self.BASE_ELEMS) + [self.RARE_TOKEN, self.UNK_TOKEN, self.MASK_TOKEN]
        self.vocab_size = len(self.id_to_token)

    @staticmethod
    def _canon(sym: str):
        if sym is None: return None
        s = str(sym).strip()
        if not s: return None
        return s.upper() if len(s) == 1 else (s[0].upper() + s[1:].lower())

    def encode(self, elements: Sequence[str]) -> List[int]:
        out = []
        for e in elements:
            c = self._canon(e)
            if c is None: out.append(self.unk_id)
            elif c in self.elem_to_id: out.append(self.elem_to_id[c])
            else: out.append(self.rare_id)
        return out


# ---------------- Featurizer ----------------
class Featurize_Ligand:
    def __init__(self, knn_k: int = 48, virtual_frame_num: int = 3, min_atoms: int = 3):
        self.tokenizer = LigandTokenizer()
        self.knn_k = int(knn_k)
        self.virtual_frame_num = int(virtual_frame_num)
        self.min_atoms = int(min_atoms)

        self.A_TOTAL = 6
        self.MIXED_NODE_IN = 114
        self.MIXED_EDGE_IN = 272

    @staticmethod
    def _build_local_slots(coords: torch.Tensor) -> torch.Tensor:
        N = coords.shape[0]
        device = coords.device
        X6 = torch.full((N, 6, 3), float("nan"), dtype=coords.dtype, device=device)
        if N == 0:
            return X6

        dmat = torch.cdist(coords, coords, p=2)  # [N,N]
        dmat.fill_diagonal_(float("inf"))
        k2 = 2 if N >= 3 else 1
        knn2 = torch.topk(dmat, k=k2, largest=False).indices  # [N,k2]
        j1 = knn2[:, 0]
        j2 = knn2[:, 1] if knn2.shape[1] > 1 else j1  

        X6[:, 0] = coords[j1]
        X6[:, 1] = coords
        X6[:, 2] = coords[j2]
        return X6

    def _get_features_persample(self, item: Dict[str, Any]):
        # ---- 基本检查 ----
        coords_np = item.get("coords", None)
        if coords_np is None:
            return None
        coords = torch.as_tensor(coords_np, dtype=torch.float32)  # [N,3]
        if coords.ndim != 2 or coords.shape[1] != 3:
            return None
        N = int(coords.shape[0])
        if N < self.min_atoms:   
            return None

        elements = item.get("elements", None)
        if not isinstance(elements, list) or len(elements) != N:
            return None

        # ---- tokens / masks / encodings ----
        S_ids = torch.tensor(self.tokenizer.encode(elements), dtype=torch.long)  # [N]
        # mask
        cm_in = item.get("chain_mask", None)
        if isinstance(cm_in, (list, tuple)) and len(cm_in) >= 1:
            chain_mask = torch.from_numpy(np.asarray(cm_in[0], dtype=np.float32))
        else:
            chain_mask = torch.ones(N, dtype=torch.float32)
        # encoding
        ce_in = item.get("chain_encoding", None)
        if isinstance(ce_in, (list, tuple)) and len(ce_in) >= 1:
            chain_encoding = torch.from_numpy(np.asarray(ce_in[0], dtype=np.float32))
        else:
            chain_encoding = torch.ones(N, dtype=torch.float32)

        X6 = self._build_local_slots(coords)              
        X6_filled = X6.clone()
        X6_filled[torch.isnan(X6_filled)] = 0.0

        type_vec = torch.zeros(N, dtype=torch.long)      
        batch_id = torch.zeros(N, dtype=torch.long)

        center = X6_filled[:, 1, :]                       
        k_eff = min(max(self.knn_k, 1), max(N - 1, 1))
        edge_idx = knn_graph(center, k=k_eff, batch=batch_id, loop=False, flow='target_to_source')
        key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
        order = torch.argsort(key)
        edge_idx = edge_idx[:, order]                     # [2,E]
        E = edge_idx.shape[1]

        T = Rigid.make_transform_from_reference(
            X6_filled[:, 0].float(), X6_filled[:, 1].float(), X6_filled[:, 2].float()
        )
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])  # [E,1]

        K_g = self.virtual_frame_num
        X_c = T._trans                                    # [N,3]
        X_m = X_c.mean(dim=0, keepdim=True)               # [1,3]
        X_c = X_c - X_m
        cov = X_c.T @ X_c                                 # [3,3]
        U, Ssvd, V = torch.svd(cov)
        # right-handed 
        D = torch.eye(3, device=coords.device)
        if torch.det(U @ V.T) < 0:
            D[2, 2] = -1.0
        Rm = U @ D @ V.T                                  # [3,3]
        rot_g = [Rm] * K_g
        trans_g = [X_m] * K_g
        T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))  # [K_g]

        feat = get_interact_feats(T, T_ts, X6_filled.float(), edge_idx, batch_id)
        _V, _E = feat["_V"], feat["_E"]

        global_nodes = torch.arange(N, N + K_g, device=coords.device)       # [K_g]
        global_src = global_nodes.repeat_interleave(N)                       # [K_g*N] (node <- global)
        global_dst = torch.arange(N, device=coords.device).repeat(K_g)       # [K_g*N]
        edge_idx_g = torch.stack([global_dst, global_src], dim=0)            # [2, K_g*N]
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g.flip(0)], dim=1)      # [2, 2*K_g*N]
        E_g = edge_idx_g.shape[1]

        T_all = Rigid.cat([T, T_g], dim=0)            # [N + K_g]
        T_src = T_all[edge_idx_g[0], None]            # [E_g,1]
        T_dst = T_all[edge_idx_g[1], None]            # [E_g,1]
        T_gs  = T_dst.invert().compose(T_src)         # [E_g,1]

        dist_ts = T_ts._trans.norm(dim=-1)                # [E,1]
        dist_gs = T_gs._trans.norm(dim=-1)                # [E_g,1]
        rbf_ts = rbf(dist_ts, 0, 50, 16)[:, 0].reshape(E,   -1)  # [E,16]
        rbf_gs = rbf(dist_gs, 0, 50, 16)[:, 0].reshape(E_g, -1)  # [E_g,16]

        setattr(T_ts, "rbf", rbf_ts)
        setattr(T_gs, "rbf", rbf_gs)

        chain_features = torch.ones(E, dtype=torch.int32)  #
        S = S_ids.clone(); S_tgt = S_ids.clone()
        S_in = torch.full_like(S_ids, fill_value=self.tokenizer.mask_id)
        loss_mask = chain_mask

        mat9 = torch.zeros(3, 3, dtype=torch.long)
        edge_stats_detail = {
            "same_chain": int(E),
            "cross_same_type": 0,
            "cross_diff_type": 0,
            "total_edges": int(E),
            "same_frac": 1.0 if E > 0 else 0.0,
            "cross_same_type_frac": 0.0,
            "cross_diff_type_frac": 0.0,
            "type_pair_counts_3x3": mat9.cpu().tolist(),
            "type_legend": ["protein","rna","dna"],
        }

        _V_g = torch.arange(self.virtual_frame_num, dtype=torch.long)
        _E_g = torch.zeros((E_g, 128), dtype=torch.float32)

        out = {
            "type_vec": type_vec,  # [N] 0
            "T": T, "T_g": T_g, "T_ts": T_ts, "T_gs": T_gs,
            "rbf_ts": rbf_ts, "rbf_gs": rbf_gs,

            "X": X6_filled, "_V": _V, "_E": _E,
            "_V_g": _V_g, "_E_g": _E_g,
            "edge_idx": edge_idx, "edge_idx_g": edge_idx_g,

            "S": S, "S_tgt": S_tgt, "S_in": S_in,
            "loss_mask": loss_mask,

            "batch_id": batch_id,
            "batch_id_g": torch.zeros(self.virtual_frame_num, dtype=torch.long),
            "num_nodes": torch.tensor([N], dtype=torch.long),
            "mask": chain_mask, "chain_mask": chain_mask, "chain_encoding": chain_encoding,
            "K_g": self.virtual_frame_num,
            "chain_features": chain_features,
            "edge_stats_detail": edge_stats_detail,
        }
        return out

    def featurize(self, batch: List[dict]):
        samples = []
        for one in batch:
            feat = self._get_features_persample(one)
            if feat is not None:
                samples.append(feat)
        if not samples:
            return None
        return self.custom_collate_fn(samples)

    def custom_collate_fn(self, batch: List[dict]):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None

        K_g = int(batch[0]["K_g"])
        num_nodes_list = [int(b["num_nodes"][0]) for b in batch]
        B = len(batch)
        total_real = sum(num_nodes_list)

        prefix_real = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(num_nodes_list[:-1]), dim=0).tolist()),
            dtype=torch.long
        )
        base_virtual = total_real

        def remap_indices(local_idx: torch.Tensor, N_i: int, base_real_i: int, base_virt_i: int) -> torch.Tensor:
            is_virtual = (local_idx >= N_i)
            out = local_idx.clone()
            out[~is_virtual] += base_real_i
            out[is_virtual] = (local_idx[is_virtual] - N_i) + (base_virtual + base_virt_i)
            return out

        ret: Dict[str, torch.Tensor] = {}

        cat_keys = [
            "X", "_V", "_E",
            "S", "S_tgt", "S_in",
            "type_vec",
            "mask", "loss_mask", "chain_mask", "chain_encoding",
            "rbf_ts", "rbf_gs",
        ]
        for k in cat_keys:
            ret[k] = torch.cat([b[k] for b in batch], dim=0)

        ret["_V_g"] = torch.cat([b["_V_g"] for b in batch], dim=0).long()
        ret["_E_g"] = torch.cat([b["_E_g"] for b in batch], dim=0)

        for k in ["T", "T_g", "T_ts", "T_gs"]:
            T_cat = Rigid.cat([b[k] for b in batch], dim=0)
            ret[k + "_rot"]   = T_cat._rots._rot_mats    
            ret[k + "_trans"] = T_cat._trans             
            

        ret["num_nodes"] = torch.tensor(num_nodes_list, dtype=torch.long)
        ret["batch_id"]  = torch.cat([torch.full((num_nodes_list[i],), i, dtype=torch.long) for i in range(B)], dim=0)
        ret["batch_id_g"] = torch.cat([torch.full((K_g,), i, dtype=torch.long) for i in range(B)], dim=0)

        edge_parts = []
        for i, b in enumerate(batch):
            shift = prefix_real[i]
            edge_parts.append(b["edge_idx"] + shift)
        ret["edge_idx"] = torch.cat(edge_parts, dim=1)

        edge_g_parts = []
        for i, b in enumerate(batch):
            N_i = num_nodes_list[i]
            base_real_i = int(prefix_real[i].item())
            base_virt_i = i * K_g
            src_local = b["edge_idx_g"][0]
            dst_local = b["edge_idx_g"][1]
            src_global = remap_indices(src_local, N_i, base_real_i, base_virt_i)
            dst_global = remap_indices(dst_local, N_i, base_real_i, base_virt_i)
            edge_g_parts.append(torch.stack([src_global, dst_global], dim=0))
        ret["edge_idx_g"] = torch.cat(edge_g_parts, dim=1)

        ret["K_g"] = K_g
        return ret



