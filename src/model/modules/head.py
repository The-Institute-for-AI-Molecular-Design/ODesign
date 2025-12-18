import torch
import torch.nn as nn

from src.utils.license_register import register_license
from src.model.modules.primitives import Linear
from src.utils.data.constants import BOND_TYPE
from src.api.model_interface import (
    PairFormerOutput,
    PairwiseOutput,
)

@register_license('bytedance2024')
class DistogramHead(nn.Module):
    """Implements Algorithm 1 [Line17] in AF3
    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2)
    """

    def __init__(self, c_z: int = 128, no_bins: int = 64) -> None:
        """
        Args:
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            no_bins (int, optional): Number of distogram bins. Defaults to 64.
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(
            in_features=self.c_z, out_features=self.no_bins, initializer="zeros"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # [*, N, N, C_z]
        """
        Args:
            z (torch.Tensor): pairformer output
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, no_bins]
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits

@register_license('odesign2025')
class BondTypeHead(nn.Module):
    """Implements a bond type classifier head.
    Computes bond type probability distribution.
    """

    def __init__(self, c_z: int = 1, c_hidden: int = 128, no_bond_types: int = len(BOND_TYPE)) -> None:
        """
        Args:
            c_in (int, optional): hidden dim [for pair embedding]. Defaults to 1.
            c_hidden (int, optional): hidden dim. Defaults to 128.
            no_bond_types (int, optional): Number of bond types. Defaults to len(BOND_TYPE).
        """
        super(BondTypeHead, self).__init__()

        self.c_in = c_z
        self.c_hidden = c_hidden
        self.no_bond_types = no_bond_types

        self.classifier = nn.Sequential(
            Linear(self.c_in, self.c_hidden),
            Linear(self.c_hidden, no_bond_types),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): pairformer output
                [*, N_token, N_token, c_z]

        Returns:
            torch.Tensor: bond type probability distribution
                [*, N_token, N_token, no_bond_types]
        """
        # [*, N_token, N_token, no_bond_types]
        logits = self.classifier(z.unsqueeze(0))
        return logits

@register_license('odesign2025')
class PairwiseHead(nn.Module):
    """Implements a pairwise head.
    Computes pairwise probability distribution.
    """

    def __init__(
        self, 
        c_z: int = 128,
        no_bond_types: int = len(BOND_TYPE), 
        no_bins: int = 64,
        bond_reconstruction = True
    ) -> PairwiseOutput:
        """
        Args:
            c_in (int, optional): hidden dim [for pair embedding]. Defaults to 1.
            c_hidden (int, optional): hidden dim. Defaults to 128.
            no_bond_types (int, optional): Number of bond types. Defaults to len(BOND_TYPE).
        """
        super(PairwiseHead, self).__init__()

        self.distogram_head = DistogramHead(c_z=c_z, no_bins=no_bins)
        self.bond_reconstruction = bond_reconstruction
        if bond_reconstruction:
            self.bond_type_head = BondTypeHead(c_z=c_z, no_bond_types=no_bond_types)

    def forward(self, input_embedding: PairFormerOutput) -> PairwiseOutput:
        """
        Args:
            input (PairFormerOutput): pairformer output
                [*, N_token, N_token, c_z]

        Returns:
            PairwiseOutput: bond type probability distribution
                [*, N_token, N_token, no_bond_types]
        """
        # [*, N_token, N_token, no_bond_types]
        distogram = self.distogram_head(input_embedding.z)
        if self.bond_reconstruction:
            token_bond_type_logits = self.bond_type_head(input_embedding.z)
        else:
            token_bond_type_logits = None
        return PairwiseOutput(distogram, token_bond_type_logits)