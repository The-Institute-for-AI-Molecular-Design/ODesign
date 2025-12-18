# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.if_module import *
from ..datasets.featurizer_ligand import LigandTokenizer
from ..tools.affine_utils import Rigid, Rotation
from ..modules.if_module import MLPDecoder_Ligand as MLPDecoder

class ODesign_Ligand_Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        geo_layer, attn_layer = args.geo_layer, args.attn_layer
        node_layer, edge_layer = args.node_layer, args.edge_layer
        encoder_layer, hidden_dim = args.encoder_layer, args.hidden_dim
        dropout, mask_rate = args.dropout, args.mask_rate
        
        self.tokenizer = LigandTokenizer()
        self.mask_id = self.tokenizer.mask_id
        V = self.tokenizer.vocab_size

        NODE_IN = 114
        EDGE_IN = 272
        self.node_embedding = build_MLP(2, NODE_IN, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, EDGE_IN, hidden_dim, hidden_dim)


        self.seq_embedding  = nn.Embedding(V, hidden_dim)   
        self.type_embedding = nn.Embedding(3, hidden_dim)   

        self.task_embedding = nn.Embedding(3, hidden_dim)
        self.edge_rel_emb = nn.Embedding(2, hidden_dim)   
        self.edge_pairtype_emb = nn.Embedding(9, hidden_dim)  


        self.virtual_embedding = nn.Embedding(30, hidden_dim)
        self.encoder = StructureEncoder(geo_layer, attn_layer, node_layer, edge_layer,
                                        encoder_layer, hidden_dim, dropout, mask_rate)
        self.decoder = MLPDecoder(hidden_dim, vocab=V)

        self._init_params()

    def _init_params(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch, num_global=3):

        X, h_V0, h_E0 = batch['X'], batch['_V'], batch['_E']
        edge_idx, batch_id = batch['edge_idx'], batch['batch_id']
        edge_idx_g, batch_id_g = batch['edge_idx_g'], batch['batch_id_g']

        # rigid body
        T   = Rigid(Rotation(batch['T_rot']),   batch['T_trans'])
        T_g = Rigid(Rotation(batch['T_g_rot']), batch['T_g_trans'])
        T_ts= Rigid(Rotation(batch['T_ts_rot']),batch['T_ts_trans'])
        T_gs= Rigid(Rotation(batch['T_gs_rot']),batch['T_gs_trans'])
        T_ts.rbf = batch['rbf_ts']
        T_gs.rbf = batch['rbf_gs']


        h_V = self.node_embedding(h_V0)
        h_E = self.edge_embedding(h_E0)
        h_V_g = self.virtual_embedding(batch['_V_g'])
        h_E_g = torch.zeros((edge_idx_g.shape[1], h_E.shape[1]), device=h_V.device, dtype=h_V.dtype)


        h_E_0_cat = torch.cat(
            [h_E0, torch.zeros((edge_idx_g.shape[1], h_E0.shape[1]), device=h_V.device, dtype=h_V.dtype)]
        )
        h_V = self.encoder(
            h_S=None,
            T=T, T_g=T_g,
            h_V=h_V, h_V_g=h_V_g,
            h_E=h_E, h_E_g=h_E_g,
            T_ts=T_ts, T_gs=T_gs,
            edge_idx=edge_idx, edge_idx_g=edge_idx_g,
            batch_id=batch_id, batch_id_g=batch_id_g,
            h_E_0=h_E_0_cat
        )


        logits = self.decoder.readout(h_V)          
        if 'vocab_logit_mask' in batch:
            logits = logits + batch['vocab_logit_mask'].to(logits.dtype)
        log_probs = F.log_softmax(logits, dim=-1)

        return {'log_probs': log_probs, 'logits': logits}


    def _get_features(self, batch):
        return batch