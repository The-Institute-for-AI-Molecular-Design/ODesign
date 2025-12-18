import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import os
# from model_interface import MInterface_base
from omegaconf import OmegaConf
import pytorch_lightning as pl

class MInterface(pl.LightningModule):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = torch.nn.NLLLoss(reduction='none')

    def forward(self, batch):
        batch = self.model._get_features(batch)
        results = self.model(batch)
        log_probs, mask = results['log_probs'], batch['mask']
        cmp = log_probs.argmax(dim=-1) == batch['S']
        recovery = (cmp * mask).sum() / (mask.sum())
        return recovery

    def load_model(self):
        params = self.hparams.configs
        params.update(self.hparams)

        if params.dataset == 'ligand':
            from src.model.modules.invfold.src.models.odesign_ligand import ODesign_Ligand_Model
            self.model = ODesign_Ligand_Model(params)
        else:
            from src.model.modules.invfold.src.models.odesign_model import ODesign_Model
            self.model = ODesign_Model(params)
        