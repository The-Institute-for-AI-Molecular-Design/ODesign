import torch
import numpy as np
from collections.abc import Sequence
from ..tools.affine_utils import Rigid, Rotation, get_interact_feats
from torch_geometric.nn.pool import knn_graph
from torch_scatter import scatter_sum
from ..modules.if_module import *
from typing import List, Sequence, Optional, Set, Tuple
import math

class UniTokenizer:
    def __init__(self, map_N_mode: str = "A"):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY'       # 0..19
        self.nuc5 = ['A','C','G','T','U']         # 20..24
        self.map_N_mode = map_N_mode

        self.nuc_to_id = {b: 20+i for i, b in enumerate(self.nuc5)}
        self.vocab_size = 25

        self.dna_ids = [self.nuc_to_id[x] for x in ['A','C','G','T']]   # [20,21,22,23]
        self.rna_ids = [self.nuc_to_id[x] for x in ['A','C','G','U']]   # [20,21,22,24]

    def _map_Ns(self, s: str) -> str:
        if 'N' not in s: return s
        if self.map_N_mode == 'A':
            return s.replace('N', 'A')
        elif self.map_N_mode == 'random':
            import random
            return ''.join(ch if ch!='N' else random.choice('ACGT') for ch in s)
        else:
            return s.replace('N','A')

    def _auto_kind(self, s: str) -> str:
        S = set(s)
        if S.issubset(set('AUGC')):   return 'rna'
        if S.issubset(set('ATGCN')):  return 'dna'
        return 'protein'

    def encode(self, seq: str, kind: Optional[str] = None) -> List[int]:
        s = seq.upper()
        if kind is None:
            kind = self._auto_kind(s)

        if kind == 'protein':
            return [self.alphabet_protein.index(ch) for ch in s]  # 0..19

        if kind == 'dna':
            s = s.replace('U', 'T')
            s = self._map_Ns(s)
        elif kind == 'rna':
            s = s.replace('T', 'U')
        else:
            raise ValueError(f"Unknown kind: {kind}")

        ids = []
        for ch in s:
            if ch not in 'ACGTU':
                ch = 'A'
            ids.append(self.nuc_to_id[ch])  # 20..24
        return ids

    def decode(self, ids: Sequence[int], kind: Optional[str] = None) -> str:
        out = []
        for t in ids:
            if 0 <= t < 20:
                out.append(self.alphabet_protein[t])
            elif 20 <= t < 25:
                base = self.nuc5[t-20]  # 'A','C','G','T','U'
                if kind == 'dna' and base == 'U': base = 'T'
                if kind == 'rna' and base == 'T': base = 'U'
                out.append(base)
            else:
                out.append('X')
        return ''.join(out)


class GeoNoiseAugmentor:

    def __init__(
        self,
        p: float = 0.5,                       # 
        types: Set[str] = frozenset({"rna", "dna"}),  # 
        residue_mask_ratio: float = 0.10,     # 
        rot_sigma_deg: float = 10.0,          # 
        trans_sigma: float = 0.20,            # 
        atom_mask_ratio: float = 0.10,        # 
        atom_sigma: float = 0.20,             # 
        atom_replace: bool = True,            # 
        protect_anchor_atoms: bool = True,    # 
        gap_p: float = 0.0,                   # 
        gap_len_max: int = 3,                 # 
        gap_close: bool = True,               # 
        bend_mean_deg: float = 60.0,          # 
        bend_std_deg: float = 20.0,           # 
        target_dist_protein: float = 3.8,     # 
        target_dist_nuc: float = 6.0,         # 
        seed: Optional[int] = None,
    ):
        self.p = float(p)
        self.types = set(types)

        self.residue_mask_ratio = float(residue_mask_ratio)
        self.rot_sigma_deg = float(rot_sigma_deg)
        self.trans_sigma = float(trans_sigma)

        self.atom_mask_ratio = float(atom_mask_ratio)
        self.atom_sigma = float(atom_sigma)
        self.atom_replace = bool(atom_replace)
        self.protect_anchor_atoms = bool(protect_anchor_atoms)

        self.gap_p = float(gap_p)
        self.gap_len_max = int(gap_len_max)
        self.gap_close = bool(gap_close)
        self.bend_mean_rad = math.radians(bend_mean_deg)
        self.bend_std_rad = math.radians(bend_std_deg)
        self.target_dist_protein = float(target_dist_protein)
        self.target_dist_nuc = float(target_dist_nuc)

        self._g = torch.Generator()
        if seed is not None:
            self._g.manual_seed(int(seed))


    def set_epoch(self, epoch: int):
        base = 1469598103934665603  # FNV offset basis
        self._g.manual_seed(base ^ (epoch * 1099511628211))

    def enabled_for_kind(self, kind: str) -> bool:
        return kind in self.types

    def enable_now(self, device: torch.device) -> bool:
        return torch.rand((), generator=self._g, device=device) < self.p

    @staticmethod
    def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        ax = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
        x, y, z = ax.unbind(-1)
        zeros = torch.zeros_like(x)
        K = torch.stack([
            torch.stack([zeros, -z,    y], dim=-1),
            torch.stack([   z, zeros, -x], dim=-1),
            torch.stack([  -y,    x, zeros], dim=-1),
        ], dim=-2)  # [..., 3, 3]
        eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape[:-2] + (3, 3))
        a = angle.unsqueeze(-1).unsqueeze(-1)  # [..., 1, 1]
        sin_a = torch.sin(a); cos_a = torch.cos(a)
        R = eye + sin_a * K + (1.0 - cos_a) * (K @ K)
        return R

    def _apply_residue_se3_noise(self, X: torch.Tensor) -> torch.Tensor:
        N, A, _ = X.shape
        if N == 0:
            return X
        num_sel = max(1, int(round(self.residue_mask_ratio * N)))
        idx = torch.randperm(N, generator=self._g, device=X.device)[:num_sel]

        angle_sigma = math.radians(self.rot_sigma_deg)
        axis = torch.randn((num_sel, 3), generator=self._g, device=X.device)
        angle = torch.randn((num_sel,), generator=self._g, device=X.device) * angle_sigma
        R = self._axis_angle_to_matrix(axis, angle)               # [K, 3, 3]

        t = torch.randn((num_sel, 1, 3), generator=self._g, device=X.device) * self.trans_sigma  # [K,1,3]

        X_sel = X[idx]                                           # [K, A, 3]
        X_sel = torch.matmul(X_sel, R.transpose(-1, -2)) + t
        X = X.clone()
        X[idx] = X_sel
        return X

    def _apply_atom_randomization(self, X: torch.Tensor, anchors_idx: Tuple[int, int, int]) -> torch.Tensor:

        N, A, _ = X.shape
        if N == 0 or A == 0:
            return X
        X_out = X.clone()
        probs = torch.full((N, A), float(self.atom_mask_ratio), device=X.device)
        if self.protect_anchor_atoms:
            a0, a1, a2 = anchors_idx
            for a in (a0, a1, a2):
                if 0 <= a < A:
                    probs[:, a] = 0.0
        mask = torch.bernoulli(probs, generator=self._g).bool()
        if not mask.any():
            return X_out
        if self.atom_replace:
            noise = torch.randn((N, A, 3), generator=self._g, device=X.device) * self.atom_sigma
            X_out[mask] = noise[mask]
        else:
            noise = torch.randn((N, A, 3), generator=self._g, device=X.device) * self.atom_sigma
            X_out[mask] = X_out[mask] + noise[mask]
        return X_out

    def _pick_gap(self, N: int, device):
        if N < 6:
            return None
        k_max = min(self.gap_len_max, max(1, N - 2))
        k = int(torch.randint(1, k_max + 1, (1,), generator=self._g, device=device))
        start = int(torch.randint(1, N - k, (1,), generator=self._g, device=device))
        end = start + k
        if end >= N:
            end = N - 1
        return start, end

    def _close_suffix(self, X: torch.Tensor, pivot_idx: int, suffix_start: int,
                      center_atom_idx: int, angle_rad: float, target_dist: float) -> torch.Tensor:

        Y = X.clone()
        P = X[pivot_idx, center_atom_idx]  
        axis = torch.randn(3, generator=self._g, device=X.device)
        axis = axis / (axis.norm() + 1e-9)
        R = self._axis_angle_to_matrix(axis[None], torch.tensor([angle_rad], device=X.device, dtype=X.dtype))[0]

        Y_suffix = X[suffix_start:] - P
        Y_suffix = torch.matmul(Y_suffix, R.transpose(-1, -2)) + P
        Y[suffix_start:] = Y_suffix

        S_rot = Y[suffix_start, center_atom_idx]
        dir_vec = S_rot - P
        d = dir_vec.norm() + 1e-8
        dir_unit = dir_vec / d
        desired = P + dir_unit * target_dist
        t = desired - S_rot
        Y[suffix_start:] = Y[suffix_start:] + t
        return Y

    def apply_gap_bend(self, X: torch.Tensor, kind: str, anchors_idx: Tuple[int, int, int]):

        if self.gap_p <= 0.0:
            return X, None
        if torch.rand((), generator=self._g, device=X.device) >= self.gap_p:
            return X, None

        N, A, _ = X.shape
        sel = self._pick_gap(N, X.device)
        if sel is None:
            return X, None
        start, end = sel

        keep = torch.ones(N, dtype=torch.bool, device=X.device)
        keep[start:end] = False
        X2 = X[keep]  

        if self.gap_close and (start > 0) and (start < X2.shape[0]):
            angle = torch.clamp(
                torch.randn((), generator=self._g, device=X.device) * self.bend_std_rad + self.bend_mean_rad,
                min=math.radians(10.0),
                max=math.radians(170.0)
            ).item()
            target = self.target_dist_nuc if kind in ("rna", "dna") else self.target_dist_protein
            X2 = self._close_suffix(
                X2, pivot_idx=start-1, suffix_start=start,
                center_atom_idx=anchors_idx[1],  #
                angle_rad=angle, target_dist=target
            )
        return X2, keep

    def apply(self, X: torch.Tensor, kind: str, anchors_idx: Tuple[int, int, int]) -> torch.Tensor:
        if not self.enabled_for_kind(kind):
            return X
        if self.residue_mask_ratio > 0 and (self.rot_sigma_deg > 0 or self.trans_sigma > 0):
            X = self._apply_residue_se3_noise(X)
        if self.atom_mask_ratio > 0 and self.atom_sigma > 0:
            X = self._apply_atom_randomization(X, anchors_idx)
        return X



class Featurize:
    def __init__(self, mixed: bool = False, augmentor: Optional[GeoNoiseAugmentor] = None) -> None:
        self.tokenizer = UniTokenizer()
        self.mixed = mixed
        self.virtual_frame_num = 3
        self.augmentor = augmentor  


        self.atoms_lst_protein = ['N', 'CA', 'C', 'O']
        self.atoms_lst_nuc_rna = ["P", "C4", "N", "O5", 'C3',"O3", "C5"]
        self.atoms_lst_nuc_dna = ['O5', 'C5', 'C4', 'P', 'C3', 'O3', 'N']

        self.MIXED_NODE_IN = 114 + 19
        self.MIXED_EDGE_IN = 272 + 38

    @staticmethod
    def _pad_last_dim_to(x: torch.Tensor, target: int, value: float = 0.0) -> torch.Tensor:
        cur = x.size(-1)
        if cur == target: return x
        if cur < target:  return F.pad(x, (0, target - cur), value=value)
        return x[..., :target]

    @staticmethod
    def _pad_dim_to(x: torch.Tensor, dim: int, target: int, value: float = 0.0) -> torch.Tensor:
        if x.size(dim) == target:
            return x
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = x.permute(*perm)
        need = target - x.size(dim)
        if need < 0:
            x_perm = x_perm[..., :target]
        else:
            x_perm = F.pad(x_perm, (0, need), value=value)
        return x_perm.permute(*perm)

    def merge_coords(self, item):
        X = []
        if item['type'] == 'protein':
            atoms_lst = self.atoms_lst_protein
        elif item['type'] == 'dna':
            atoms_lst = self.atoms_lst_nuc_dna
        else:  # 'rna'
            atoms_lst = self.atoms_lst_nuc_rna
        for name in atoms_lst:
            if name in item:
                arr = item[name]
                if isinstance(arr, (list, tuple)):
                    arr = arr[0]
                X.append(torch.from_numpy(arr))
        return torch.cat(X, dim=0).permute(1, 0, 2)  # [L, A, 3]

    @staticmethod
    def _anchor_indices_for_kind(kind: str) -> Tuple[int, int, int]:
        return (0, 1, 2)

    def _get_features_persample(self, batch):
        # 规范 list 形状
        for key in batch:
            try:
                batch[key] = batch[key][None, ...]
            except Exception:
                batch[key] = batch[key]

        S = []
        for seq in batch['seq']:
            S.extend(self.tokenizer.encode(seq, batch['type']))
        S = torch.tensor(S)

        X = self.merge_coords(batch)                # [L, A, 3]

        chain_mask = torch.from_numpy(np.concatenate(batch['chain_mask'])).float()
        chain_encoding = torch.from_numpy(np.concatenate(batch['chain_encoding'])).float()

        X, S = X.unsqueeze(0), S.unsqueeze(0)      # [1,L,A,3], [1,L]
        mask = torch.isfinite(torch.sum(X, (2, 3))).float()  # [1,L]
        numbers = torch.sum(mask, dim=1).int()

        S_new = torch.zeros_like(S)
        X_new = torch.zeros_like(X) + torch.nan
        for i, n in enumerate(numbers):
            X_new[i, :n, ::] = X[i][mask[i] == 1]
            S_new[i, :n] = S[i][mask[i] == 1]
        X, S = X_new, S_new

        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X, (2, 3))).float()
        X[isnan] = 0.0

        mask_bool = (mask == 1)

        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], shape[1], -1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1, *shape[2:])
            return out

        batch_id_full = torch.arange(mask_bool.shape[0], device=mask_bool.device)[:, None].expand_as(mask_bool)

        chain_mask_compact = torch.masked_select(mask, mask_bool)
        chain_encoding_compact = chain_encoding[mask_bool[0].bool()].float()
        seq = node_mask_select(S)     # [N]
        X   = node_mask_select(X)     # [N, A, 3]

        if X.shape[0] < 2:
            return None

        batch_id = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)

        kind = batch['type']
        if self.augmentor is not None and self.augmentor.enabled_for_kind(kind):
            if self.augmentor.enable_now(device=X.device):
                anchors_idx = self._anchor_indices_for_kind(kind)
                with torch.no_grad():
                    X = self.augmentor.apply(X, kind=kind, anchors_idx=anchors_idx)
                    if hasattr(self.augmentor, "apply_gap_bend"):
                        X2, keep_mask = self.augmentor.apply_gap_bend(X, kind=kind, anchors_idx=anchors_idx)
                        if keep_mask is not None:
                            if X2.shape[0] < 2:
                                return None
                            X = X2
                            seq = seq[keep_mask]
                            chain_encoding_compact = chain_encoding_compact[keep_mask]
                            chain_mask_compact = chain_mask_compact[keep_mask]
                            batch_id = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        # -------------------------

        center = X[:, 1, :]  

        # KNN
        edge_idx = knn_graph(center, k=32, batch=batch_id, loop=False, flow='target_to_source')
        key = (edge_idx[1] * (edge_idx[0].max() + 1) + edge_idx[0]).long()
        order = torch.argsort(key)
        edge_idx = edge_idx[:, order]

        T = Rigid.make_transform_from_reference(X[:, 0].float(), X[:, 1].float(), X[:, 2].float())
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx, None].invert().compose(T[src_idx, None])

        num_global = self.virtual_frame_num
        X_c = T._trans
        X_m = X_c.mean(dim=0, keepdim=True)
        X_c = X_c - X_m
        U, Ssvd, V = torch.svd(X_c.T @ X_c)
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.zeros_like(V); D[[0, 1], [0, 1]] = 1; D[2, 2] = -1 * d + 1 * (~d)
        V = D @ V
        R = torch.matmul(U, V.permute(0, 1))

        rot_g = [R] * num_global
        trans_g = [X_m] * num_global

        feat = get_interact_feats(T, T_ts, X.float(), edge_idx, batch_id)
        _V, _E = feat['_V'], feat['_E']  # [N, d_v], [E, d_e]

        T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g, dim=0))
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        global_src = torch.cat([batch_id + k * num_nodes.shape[0] for k in range(num_global)]) + num_nodes
        global_dst = torch.arange(batch_id.shape[0], device=batch_id.device).repeat(num_global)
        edge_idx_g = torch.stack([global_dst, global_src])
        edge_idx_g_inv = torch.stack([global_src, global_dst])
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

        batch_id_g = torch.zeros(num_global, dtype=batch_id.dtype, device=batch_id.device)
        T_all = Rigid.cat([T, T_g], dim=0)
        idx, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx, None].invert().compose(T_all[idx, None])

        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(_E.shape[0], -1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:, 0].view(edge_idx_g.shape[1], -1)

        _V_g = torch.arange(num_global, device=batch_id.device)
        _E_g = torch.zeros([edge_idx_g.shape[1], 128], device=batch_id.device)

        mask_flat = chain_mask_compact
        chain_features = (chain_encoding_compact[edge_idx[0]] == chain_encoding_compact[edge_idx[1]]).int()

        kind2id = {'protein': 0, 'rna': 1, 'dna': 2}
        k = torch.tensor(kind2id.get(kind, 0), device=seq.device, dtype=torch.long)
        type_vec = k.expand(seq.shape[0])

        batch_out = {
            'type_vec': type_vec,
            'T': T, 'T_g': T_g, 'T_ts': T_ts, 'T_gs': T_gs,
            'rbf_ts': rbf_ts, 'rbf_gs': rbf_gs,
            'X': X,
            'chain_features': chain_features,
            '_V': _V, '_E': _E,
            '_V_g': _V_g, '_E_g': _E_g,
            'S': seq,
            'edge_idx': edge_idx, 'edge_idx_g': edge_idx_g,
            'batch_id': batch_id, 'batch_id_g': batch_id_g,
            'num_nodes': num_nodes,
            'mask': mask_flat,
            'chain_mask': mask_flat,
            'chain_encoding': chain_encoding_compact,
            'K_g': num_global
        }
        return batch_out

    def featurize(self, batch):
        samples = []
        for one in batch:
            feat = self._get_features_persample(one)
            if feat is not None:
                samples.append(feat)
        if not samples:
            return None
        return self.custom_collate_fn(samples)

    def custom_collate_fn(self, batch: List[dict]):
        batch = [one for one in batch if one is not None]

        num_nodes = torch.cat([one['num_nodes'] for one in batch])
        shift = num_nodes.cumsum(dim=0)
        shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)

        def shift_node_idx(idx, num_node, shift_real, shift_virtual):
            mask = idx >= num_node
            shift_combine = (~mask) * (shift_real) + (mask) * (shift_virtual)
            return idx + shift_combine

        ret = {}
        for key in batch[0].keys():
            if batch[0][key] is None:
                continue

            if key in ['T', 'T_g', 'T_ts', 'T_gs']:
                T = Rigid.cat([one[key] for one in batch], dim=0)
                ret[key + '_rot'] = T._rots._rot_mats
                ret[key + '_trans'] = T._trans

            elif key == 'edge_idx':
                ret[key] = torch.cat([one[key] + shift[idx] for idx, one in enumerate(batch)], dim=1)

            elif key == 'edge_idx_g':
                edge_idx_g = []
                for idx, one in enumerate(batch):
                    shift_virtual = shift[-1] + idx * one['K_g'] - num_nodes[idx]
                    src = shift_node_idx(one['edge_idx_g'][0], num_nodes[idx], shift[idx], shift_virtual)
                    dst = shift_node_idx(one['edge_idx_g'][1], num_nodes[idx], shift[idx], shift_virtual)
                    edge_idx_g.append(torch.stack([src, dst]))
                ret[key] = torch.cat(edge_idx_g, dim=1)

            elif key in ['batch_id', 'batch_id_g']:
                ret[key] = torch.cat([one[key] + idx for idx, one in enumerate(batch)])

            elif key == '_V':
                if self.mixed:
                    tensors = [self._pad_last_dim_to(one[key], self.MIXED_NODE_IN, value=0.0) for one in batch]
                    ret[key] = torch.cat(tensors, dim=0)
                else:
                    ret[key] = torch.cat([one[key] for one in batch], dim=0)

            elif key == '_E':
                if self.mixed:
                    tensors = [self._pad_last_dim_to(one[key], self.MIXED_EDGE_IN, value=0.0) for one in batch]
                    ret[key] = torch.cat(tensors, dim=0)
                else:
                    ret[key] = torch.cat([one[key] for one in batch], dim=0)

            elif key == 'X':
                if self.mixed:
                    max_atoms = max(one['X'].shape[1] for one in batch)
                    tensors = [self._pad_dim_to(one['X'], dim=1, target=max_atoms, value=float('nan')) for one in batch]
                    ret[key] = torch.cat(tensors, dim=0)
                else:
                    ret[key] = torch.cat([one['X'] for one in batch], dim=0)

            elif key in ['K_g']:
                pass

            else:
                ret[key] = torch.cat([one[key] for one in batch], dim=0)

        return ret
