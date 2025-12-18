import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax
import numpy as np


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def build_MLP(n_layers,dim_in, dim_hid, dim_out, dropout = 0.0, activation=nn.ReLU, normalize=True):
    if normalize:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.BatchNorm1d(dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    else:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        if normalize:
            layers.append(nn.BatchNorm1d(dim_hid))
        layers.append(nn.Dropout(dropout))
        layers.append(activation())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class GeoFeat(nn.Module):
    def __init__(self, geo_layer, num_hidden, virtual_atom_num, dropout=0.0):
        super(GeoFeat, self).__init__()
        self.__dict__.update(locals())
        self.virtual_atom = nn.Linear(num_hidden, virtual_atom_num*3)
        self.virtual_direct = nn.Linear(num_hidden, virtual_atom_num*3)
        # self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+32, num_hidden, num_hidden, dropout)
        self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+virtual_atom_num, num_hidden, num_hidden, dropout)
        self.MergeEG = nn.Linear(num_hidden+num_hidden, num_hidden)

    def forward(self, h_V, h_E, T_ts, edge_idx, h_E_0):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        num_edge = src_idx.shape[0]
        num_atom = h_V.shape[0]

        V_local = self.virtual_atom(h_V).view(num_atom,-1,3)
        V_edge = self.virtual_direct(h_E).view(num_edge,-1,3)
        Ks = torch.cat([V_edge,V_local[src_idx].view(num_edge,-1,3)], dim=1)
        Qt = T_ts.apply(Ks)
        Ks = Ks.view(num_edge,-1)
        Qt = Qt.reshape(num_edge,-1)
        V_edge = V_edge.reshape(num_edge,-1)
        quat_st = T_ts._rots._rot_mats[:, 0].reshape(num_edge, -1)


        RKs = torch.einsum('eij,enj->eni', T_ts._rots._rot_mats[:,0], V_local[src_idx].view(num_edge,-1,3))
        QRK = torch.einsum('enj,enj->en', V_local[dst_idx].view(num_edge,-1,3), RKs)

        H = torch.cat([Ks, Qt, quat_st, T_ts.rbf, QRK], dim=1)
        G_e = self.we_condition(H)
        h_E = self.MergeEG(torch.cat([h_E, G_e], dim=-1))
        return h_E


class PiFoldAttn(nn.Module):
    def __init__(self, attn_layer, num_hidden, num_V, num_E, dropout=0.0):
        super(PiFoldAttn, self).__init__()
        self.__dict__.update(locals())
        self.num_heads = 4
        self.W_V = nn.Sequential(nn.Linear(num_E, num_hidden),
                                nn.GELU())
                                
        self.Bias = nn.Sequential(
                                nn.Linear(2*num_V+num_E, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,self.num_heads))
        self.W_O = nn.Linear(num_hidden, num_V, bias=False)
        self.gate = nn.Linear(num_hidden, num_V)


    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        h_V_skip = h_V

        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        num_nodes = h_V.shape[0]
        
        w = self.Bias(torch.cat([h_V[src_idx], h_E, h_V[dst_idx]],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1,n_heads, d) 
        attend = scatter_softmax(attend_logits, index=src_idx, dim=0)
        h_V = scatter_sum(attend*V, src_idx, dim=0).view([num_nodes, -1])

        h_V_gate = F.sigmoid(self.gate(h_V))
        dh = self.W_O(h_V)*h_V_gate

        h_V = h_V_skip + dh
        return h_V


class UpdateNode(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden),
            nn.BatchNorm1d(num_hidden)
        )
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden))
    
    def forward(self, h_V, batch_id):
        dh = self.dense(h_V)
        h_V = h_V + dh

        uni = batch_id.unique()
        mat = (uni[:,None] == batch_id[None]).to(h_V.dtype)
        mat = mat/mat.sum(dim=1, keepdim=True)
        c_V = mat@h_V

        h_V = h_V * F.sigmoid(self.V_MLP_g(c_V))[batch_id]
        return h_V


class UpdateEdge(nn.Module):
    def __init__(self, edge_layer, num_hidden, dropout=0.1):
        super(UpdateEdge, self).__init__()
        self.W = build_MLP(edge_layer, num_hidden*3, num_hidden, num_hidden, dropout, activation=nn.GELU, normalize=False)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.pred_quat = nn.Linear(num_hidden,8)

    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_E = self.norm(h_E + self.W(h_EV))
        return h_E


class GeneralGNN(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer,
                 num_hidden, 
                 virtual_atom_num=32, 
                 dropout=0.1,
                 mask_rate=0.15):
        super(GeneralGNN, self).__init__()
        self.__dict__.update(locals())
        self.geofeat = GeoFeat(geo_layer, num_hidden, virtual_atom_num, dropout)
        self.attention = PiFoldAttn(attn_layer, num_hidden, num_hidden, num_hidden, dropout) 
        self.update_node = UpdateNode(num_hidden)
        self.update_edge = UpdateEdge(edge_layer, num_hidden, dropout)
        self.mask_token = nn.Embedding(2, num_hidden)
    
    def get_rand_idx(self, h_V, mask_rate):
        num_N = int(h_V.shape[0] * mask_rate)
        indices = torch.randperm(h_V.shape[0], device=h_V.device)
        selected_indices = indices[:num_N]
        return selected_indices
        
    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id, h_E_0):
        if self.training:
            selected_indices = self.get_rand_idx(h_V, self.mask_rate)
            h_V[selected_indices] = self.mask_token.weight[0]

            selected_indices = self.get_rand_idx(h_E, self.mask_rate)
            h_E[selected_indices] = self.mask_token.weight[1]
        
        h_E = self.geofeat(h_V, h_E, T_ts, edge_idx, h_E_0)
        h_V = self.attention(h_V, h_E, edge_idx)
        h_V = self.update_node(h_V, batch_id)
        h_E = self.update_edge( h_V, h_E, T_ts, edge_idx, batch_id )
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 encoder_layer,
                 hidden_dim, 
                 dropout=0,
                 mask_rate=0.15):
        super(StructureEncoder, self).__init__()
        self.__dict__.update(locals())
        self.encoder_layers = nn.ModuleList([GeneralGNN(geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 hidden_dim, 
                 dropout=dropout,
                 mask_rate=mask_rate) for i in range(encoder_layer)])
        self.s = nn.Linear(hidden_dim, 1)
    
    def merge_local_global(self, h_V, h_E, T_ts, T_gs, batch_id, edge_idx, h_V_g, h_E_g, batch_id_g, edge_idx_g):
        batch_id = torch.cat([batch_id, batch_id_g])
        h_V = torch.cat([h_V, h_V_g], dim = 0)
        h_E = torch.cat([h_E, h_E_g], dim = 0)
        edge_idx = torch.cat([edge_idx, edge_idx_g], dim = -1)
        rbf = torch.cat([T_ts.rbf, T_gs.rbf], dim = 0)
        T_ts = T_ts.cat([T_ts, T_gs], dim = 0)
        T_ts.rbf = rbf
        return h_V, h_E, T_ts, batch_id, edge_idx
    
    def decouple_local_global(self, h_V, h_E, batch_id, edge_idx, h_V_g, h_E_g, batch_id_g):
        num_node_g = batch_id_g.shape[0]
        num_edge_g = h_E_g.shape[0]
        batch_id, batch_id_g = batch_id[:-num_node_g], batch_id[-num_node_g]
        h_V, h_V_g = h_V[:-num_node_g], h_V[-num_node_g:]
        h_E, h_E_g = h_E[:-num_edge_g], h_E[-num_edge_g:]
        edge_idx, global_edge = edge_idx[:,:-num_edge_g], edge_idx[:,-num_edge_g:]
        return h_V, h_E, h_V_g, h_E_g

    def forward(self, h_S,
                    T, T_g, 
                    h_V, h_V_g,
                    h_E, h_E_g,
                    T_ts, T_gs, 
                    edge_idx, edge_idx_g,
                    batch_id, batch_id_g, h_E_0):
        h_V, h_E, T_ts, batch_id, edge_idx = self.merge_local_global(h_V, h_E, T_ts, T_gs, batch_id, edge_idx, h_V_g, h_E_g, batch_id_g, edge_idx_g)

        outputs = []
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, T_ts, edge_idx, batch_id, h_E_0)
            h_V_real = h_V[:-batch_id_g.shape[0]]
            outputs.append(h_V_real.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        S = F.sigmoid(self.s(outputs))
        output = torch.einsum('nkc, nkb -> nbc', outputs, S).squeeze(1)
        return output


# class MLPDecoder(nn.Module):
#     def __init__(self, hidden_dim, vocab=33):
#         super().__init__()
#         self.readout = nn.Linear(hidden_dim, vocab)
    
#     def forward(self, h_V):
#         logits = self.readout(h_V)
#         log_probs = F.log_softmax(logits, dim=-1)
#         return log_probs, logits
    
    
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=33, mlp_hidden=None, dropout=0.1):
        super().__init__()
        h = mlp_hidden or 2 * hidden_dim

        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, hidden_dim),
            nn.Dropout(dropout),
        )
        self.readout = nn.Linear(hidden_dim, vocab)

    def forward(self, h_V):
        h = h_V + self.ffn(h_V)
        logits = self.readout(h)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits



class SimpleARBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        # x: [B, L, d]
        h = self.ln1(x)
        h, _ = self.self_attn(
            h, h, h,
            attn_mask=attn_mask,               # [L, L] 
            key_padding_mask=key_padding_mask  # [B, L] 
        )
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleARDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab, n_layers=4, n_heads=8,
                 dropout=0.1, max_pos=8192, tie_weights=True, use_struct=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, hidden_dim)
        self.pos_emb   = nn.Embedding(max_pos, hidden_dim)
        self.bos_embed = nn.Parameter(torch.randn(hidden_dim))
        self.use_struct = use_struct
        if use_struct:
            self.struct_proj = nn.Linear(hidden_dim, hidden_dim)  # 把 H_enc 映射到解码维度

        self.blocks = nn.ModuleList([
            SimpleARBlock(hidden_dim, n_heads, d_ff=hidden_dim*4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.readout = nn.Linear(hidden_dim, vocab, bias=True)
        if tie_weights:
            self.readout.weight = self.token_emb.weight

    @staticmethod
    def _causal_mask(L, device):
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return mask  # [L, L]


    def forward(self,
                H_enc_pad,   
                y_pad,       
                valid_mask   
                ):
        B, L, d = H_enc_pad.shape
        device = H_enc_pad.device

        y_prev = torch.roll(y_pad, shifts=1, dims=1)  # [B, L]
        first_pos = torch.zeros((B, L), dtype=torch.bool, device=device)
        first_pos[:, 0] = True

        tok = self.token_emb(y_prev)                  # [B, L, d]
        tok[first_pos] = self.bos_embed           

        # 2) 位置 & 结构条件
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = tok + self.pos_emb(pos_ids)
        if self.use_struct:
            x = x + self.struct_proj(H_enc_pad)

        # 3) 掩码
        attn_mask = self._causal_mask(L, device)          # [L, L]
        key_padding_mask = ~valid_mask                    # [B, L] True=PAD

        # 4) 堆叠自注意力块
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # 5) 读出
        logits = self.readout(x)               # [B, L, V]
        log_probs = F.log_softmax(logits, -1)
        return log_probs, logits
    

class StructARBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, use_struct=True):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.use_struct = use_struct

        # causal self-attn
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # cross-attn to structure memory
        if use_struct:
            self.ln_q  = nn.LayerNorm(d_model)
            self.ln_kv = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.cross_gate = nn.Parameter(torch.zeros(d_model))

        # ffn
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask,
                struct=None, struct_padding_mask=None):

        h = self.self_attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            attn_mask=attn_mask,               # [L, L] True=mask
            key_padding_mask=key_padding_mask  # [B, L] True=PAD
        )[0]
        x = x + h

        if self.use_struct and struct is not None:
            q  = self.ln_q(x)
            kv = self.ln_kv(struct)
            h_cross = self.cross_attn(
                q, kv, kv,
                key_padding_mask=struct_padding_mask  # [B, L] True=PAD
            )[0]
            gate = torch.sigmoid(self.cross_gate).view(1, 1, -1)
            x = x + gate*h_cross ## No gate here

        # 3) FFN
        x = x + self.ffn(self.ln2(x))
        return x




class MLPDecoder_Ligand(nn.Module):
    def __init__(self, hidden_dim, vocab=33):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits