from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.license_register import register_license
from src.model.modules.primitives import LinearNoBias, Transition
from src.model.modules.transformer import AttentionPairBias
from src.model.modules.feature_dims import MSAFEATS_DIMS
from src.utils.model.misc import (
    pad_at_dim,
    sample_msa_feature_dict_random_without_replacement,
)
from src.utils.openfold_local.model.dropout import DropoutRowwise
from src.utils.openfold_local.model.outer_product_mean import (
    OuterProductMean,  # Alg 9 in AF3
)
from src.utils.openfold_local.model.primitives import LayerNorm
from src.utils.openfold_local.model.triangular_attention import TriangleAttention
from src.utils.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,  # Alg 13 in AF3
)
from src.utils.openfold_local.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,  # Alg 12 in AF3
)
from src.utils.openfold_local.utils.checkpointing import (
    checkpoint_blocks,
    get_checkpoint_fn,
)
from src.api.model_interface import PairFormerInput

@register_license('bytedance2024')
class PairformerBlock(nn.Module):
    """
    Implements Algorithm 17 [Line2-Line8] in AlphaFold3.
    
    A single Pairformer block that updates pair and single representations through:
        1. Triangle multiplicative updates (outgoing and incoming)
        2. Triangle attention (starting and ending)
        3. Pair transition
        4. Single attention with pair bias (optional)
        5. Single transition (optional)
    
    The c_hidden_mul configuration follows the OpenFold implementation.
    Reference: https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/evoformer.py#L123
    
    Attributes:
        n_heads: Number of attention heads for AttentionPairBias.
        tri_mul_out: Triangle multiplication outgoing module.
        tri_mul_in: Triangle multiplication incoming module.
        tri_att_start: Triangle attention starting module.
        tri_att_end: Triangle attention ending module.
        dropout_row: Rowwise dropout layer.
        pair_transition: Transition layer for pair features.
        c_s: Channel dimension for single representations.
        attention_pair_bias: Optional attention with pair bias module (if c_s > 0).
        single_transition: Optional transition layer for single features (if c_s > 0).
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        dropout: float = 0.25,
    ) -> None:
        """
        Initializes the PairformerBlock with specified configurations.
        
        Args:
            n_heads: Number of attention heads for AttentionPairBias. Defaults to 16.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            c_s: Channel dimension for single embedding. If 0, single processing is disabled.
                Defaults to 384.
            c_hidden_mul: Hidden dimension for TriangleMultiplication modules. Defaults to 128.
            c_hidden_pair_att: Hidden dimension for TriangleAttention modules. Defaults to 32.
            no_heads_pair: Number of attention heads for TriangleAttention. Defaults to 4.
            dropout: Dropout ratio for triangle updates. Defaults to 0.25.
        
        Returns:
            None
        """
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.pair_transition = Transition(c_in=c_z, n=4)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the PairformerBlock.
        
        Applies a series of transformations to update pair and single representations:
            1. Triangle multiplication (outgoing and incoming)
            2. Triangle attention (start and end nodes)
            3. Pair transition
            4. Single attention with pair bias (if c_s > 0)
            5. Single transition (if c_s > 0)
        
        The inplace_safe flag enables memory-efficient inplace operations during inference.

        Args:
            s: Single feature representation. Can be None if c_s = 0.
                Shape: [..., N_token, c_s]
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]
            pair_mask: Mask for valid token pairs.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Should be True during inference. Defaults to False.
            chunk_size: Chunk size for memory-efficient attention operations.
                Defaults to None (no chunking).

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: A tuple containing:
                - Updated single features [..., N_token, c_s] or None if c_s = 0
                - Updated pair features [..., N_token, N_token, c_z]
        """
        if inplace_safe:
            z = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True
            )
            z += self.tri_att_start(
                z,
                mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.tri_att_end(
                z,
                mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            z = z.transpose(-2, -3).contiguous()
            z += self.pair_transition(z)
        else:
            tmu_update = self.tri_mul_out(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            tmu_update = self.tri_mul_in(
                z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=False
            )
            z = z + self.dropout_row(tmu_update)
            del tmu_update
            z = z + self.dropout_row(
                self.tri_att_start(
                    z,
                    mask=pair_mask,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)
            z = z + self.dropout_row(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
                    use_memory_efficient_kernel=use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            )
            z = z.transpose(-2, -3)

            z = z + self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
            )
            s = s + self.single_transition(s)
        return s, z

@register_license('bytedance2024')
class PairformerStack(nn.Module):
    """
    Implements Algorithm 17 [PairformerStack] in AlphaFold3.
    
    A stack of PairformerBlock modules applied sequentially to refine pair and single
    representations. Supports gradient checkpointing for memory-efficient training.
    
    The stack processes inputs through multiple Pairformer blocks, each applying:
        - Triangle updates (multiplication and attention)
        - Pair and single transitions
        - Optional attention with pair bias
    
    Attributes:
        n_blocks: Number of PairformerBlock modules in the stack.
        n_heads: Number of attention heads for each block.
        blocks_per_ckpt: Number of blocks per gradient checkpoint. If None, no checkpointing.
        blocks: ModuleList containing all PairformerBlock instances.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Initializes the PairformerStack with specified configurations.
        
        Args:
            n_blocks: Number of PairformerBlock modules in the stack. Defaults to 48.
            n_heads: Number of attention heads for AttentionPairBias in each block.
                Defaults to 16.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            c_s: Channel dimension for single embedding. Defaults to 384.
            dropout: Dropout ratio applied in each block. Defaults to 0.25.
            blocks_per_ckpt: Number of Pairformer blocks in each activation checkpoint.
                A higher value uses fewer checkpoints, trading memory for speed.
                If None, no gradient checkpointing is performed. Defaults to None.
        
        Returns:
            None
        """
        super(PairformerStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = PairformerBlock(n_heads=n_heads, c_z=c_z, c_s=c_s, dropout=dropout)
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ) -> list:
        """
        Prepares blocks with partial application of common arguments.
        
        Creates a list of partially applied block functions with shared arguments,
        optionally adding cache clearing between blocks for large structures.
        
        Args:
            pair_mask: Mask for valid token pairs. Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
            use_lma: Whether to use low-memory attention.
            inplace_safe: Whether to use inplace operations for memory efficiency.
            chunk_size: Chunk size for memory-efficient operations.
            clear_cache_between_blocks: Whether to clear CUDA cache between blocks.
        
        Returns:
            list: List of partially applied block functions ready for execution.
        """
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all PairformerBlocks in the stack.
        
        Processes single and pair representations through multiple Pairformer blocks
        sequentially. Automatically enables cache clearing between blocks for large
        structures (> 2000 tokens) during inference.
        
        Args:
            s: Single feature representation.
                Shape: [..., N_token, c_s]
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]
            pair_mask: Mask for valid token pairs.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Updated single features [..., N_token, c_s]
                - Updated pair features [..., N_token, N_token, c_z]
        """
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return s, z


@register_license('bytedance2024')
class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AlphaFold3.
    
    Performs weighted averaging of MSA features using pair representations as attention weights.
    This module aggregates information across MSA sequences, weighted by pairwise relationships
    between tokens.
    
    The operation computes:
        1. Value projections from MSA features
        2. Attention weights from pair features
        3. Weighted average: wv = sum_j(w_ij * v_j)
        4. Gated output: o = g * wv
    
    Attributes:
        c_m: Channel dimension for MSA embedding.
        c: Hidden dimension for value projections.
        n_heads: Number of attention heads.
        c_z: Channel dimension for pair embedding.
        layernorm_m: Layer normalization for MSA features.
        linear_no_bias_mv: Linear projection for MSA values.
        layernorm_z: Layer normalization for pair features.
        linear_no_bias_z: Linear projection for pair-based attention weights.
        linear_no_bias_mg: Linear projection for gating values.
        softmax_w: Softmax for attention weights.
        linear_no_bias_out: Output projection.
    """

    def __init__(self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8) -> None:
        """
        Initializes the MSAPairWeightedAveraging module.

        Args:
            c_m: Channel dimension for MSA embedding. Defaults to 64.
            c: Hidden dimension for value projections and weighted averaging. Defaults to 32.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            n_heads: Number of attention heads for weighted averaging. Defaults to 8.
        
        Returns:
            None
        """
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_m = c_m
        self.c = c
        self.n_heads = n_heads
        self.c_z = c_z
        # Input projections
        self.layernorm_m = LayerNorm(self.c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.n_heads
        )
        self.linear_no_bias_mg = LinearNoBias(
            in_features=self.c_m,
            out_features=self.c * self.n_heads,
            initializer="zeros",
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads,
            out_features=self.c_m,
            initializer="zeros",
        )

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MSAPairWeightedAveraging.
        
        Computes weighted average of MSA features using pair-based attention weights:
            1. Project MSA to values (v) and gates (g)
            2. Compute attention weights (w) from pair features
            3. Compute weighted average: wv = sum_j(w_ij * v_j)
            4. Apply gating: o = g * wv
            5. Project to output
        
        Args:
            m: MSA feature embedding.
                Shape: [..., N_msa_sampled, N_token, c_m]
            z: Pair feature embedding used to compute attention weights.
                Shape: [..., N_token, N_token, c_z]
        
        Returns:
            torch.Tensor: Updated MSA embedding after weighted averaging.
                Shape: [..., N_msa_sampled, N_token, c_m]
        """
        # Input projections
        m = self.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
        v = self.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
        v = v.reshape(
            *v.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        b = self.linear_no_bias_z(
            self.layernorm_z(z)
        )  # [...,n_token, n_token, n_heads]
        g = torch.sigmoid(
            self.linear_no_bias_mg(m)
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        g = g.reshape(
            *g.shape[:-1], self.n_heads, self.c
        )  # [...,n_msa_sampled, n_token, n_heads, c]
        w = self.softmax_w(b)  # [...,n_token, n_token, n_heads]
        wv = torch.einsum(
            "...ijh,...mjhc->...mihc", w, v
        )  # [...,n_msa_sampled,n_token,n_heads,c]
        o = g * wv
        o = o.reshape(
            *o.shape[:-2], self.n_heads * self.c
        )  # [...,n_msa_sampled, n_token, n_heads * c]
        m = self.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
        if (not self.training) and m.shape[-3] > 5120:
            del v, b, g, w, wv, o
            torch.cuda.empty_cache()
        return m

@register_license('bytedance2024')
class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8 of AlphaFold3.
    
    Processes MSA features through:
        1. MSA pair-weighted averaging
        2. MSA transition
    
    Supports chunked processing for memory efficiency with large MSA sizes.
    
    Attributes:
        c: Hidden dimension for MSAPairWeightedAveraging.
        msa_pair_weighted_averaging: Module for weighted averaging of MSA features.
        dropout_row: Rowwise dropout layer.
        transition_m: Transition layer for MSA features.
        msa_chunk_size: Chunk size for processing MSA sequences.
        msa_max_size: Maximum MSA size for padding during training.
    """

    def __init__(
        self,
        c_m: int = 64,
        c: int = 8,
        dropout: float = 0.15,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
    ) -> None:
        """
        Initializes the MSAStack module.
        
        Args:
            c_m: Channel dimension for MSA embedding. Defaults to 64.
            c: Hidden dimension for MSAPairWeightedAveraging. Defaults to 8.
            dropout: Dropout ratio for training. Defaults to 0.15.
            msa_chunk_size: Chunk size for processing MSA sequences to reduce memory.
                Defaults to 2048.
            msa_max_size: Maximum MSA size for padding during training to maintain static graph.
                Defaults to 16384.
        
        Returns:
            None
        """
        super(MSAStack, self).__init__()
        self.c = c
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(c=self.c)
        self.dropout_row = DropoutRowwise(dropout)
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MSAStack.
        
        Applies MSA pair-weighted averaging and transition with chunked processing
        for memory efficiency. During training, pads MSA to fixed size for static graph.
        During inference, uses inplace operations for memory savings.
        
        Args:
            m: MSA feature embedding.
                Shape: [..., N_msa_sampled, N_token, c_m]
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]

        Returns:
            torch.Tensor: Updated MSA embedding.
                Shape: [..., N_msa_sampled, N_token, c_m]
        """
        chunk_size = self.msa_chunk_size
        if self.training:
            # Padded m to avoid static graph change in DDP training, which will raise
            # RuntimeError: Your training graph has changed in this iteration,
            # e.g., one parameter is unused in first iteration, but then got used in the second iteration.
            # this is not compatible with static_graph set to True
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            assert (m_new[: m.shape[-3], :, :] == m).all()
            msa_pair_weighted = self.chunk_forward(
                self.msa_pair_weighted_averaging, m_new, z, chunk_size
            )
            m = m + self.dropout_row(msa_pair_weighted[: m.shape[-3], :, :])
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            m_transition = self.chunk_forward(
                self.transition_m, m_new, None, chunk_size
            )
            m = m + m_transition[: m.shape[-3], :, :]
            if (not self.training) and (z.shape[-2] > 2000 or m.shape[-3] > 5120):
                del msa_pair_weighted, m_transition
                torch.cuda.empty_cache()
        else:
            m = self.inference_forward(m, z, chunk_size)
        return m

    def chunk_forward(
        self,
        module: nn.Module,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Processes MSA in chunks with gradient checkpointing for memory efficiency.
        
        Splits MSA along the sequence dimension and processes each chunk through the
        given module with gradient checkpointing. This reduces memory usage for large MSAs.
        
        Args:
            module: The module to apply to each chunk (e.g., msa_pair_weighted_averaging,
                transition_m).
            m: MSA feature embedding.
                Shape: [..., N_msa_sampled, N_token, c_m]
            z: Pair feature embedding (optional, only used if module requires it).
                Shape: [..., N_token, N_token, c_z]
            chunk_size: Size of each chunk along the MSA sequence dimension.
                Defaults to 2048.

        Returns:
            torch.Tensor: Updated MSA embedding after processing all chunks.
                Shape: [..., N_msa_sampled, N_token, c_m]
        """

        def fixed_length_chunk(m, chunk_length, dim=0):
            dim_size = m.size(dim)
            chunk_num = (dim_size + chunk_length - 1) // chunk_length
            chunks = []

            for i in range(chunk_num):
                start = i * chunk_length
                end = min(start + chunk_length, dim_size)
                chunk = m.narrow(dim, start, end - start)
                chunks.append(chunk)

            return chunks

        checkpoint_fn = get_checkpoint_fn()
        # Split the tensor `m` into chunks along the first dimension
        # m_chunks = torch.chunk(m, chunk_size, dim=0)
        m_chunks = fixed_length_chunk(m, chunk_size, dim=0)

        # Process each chunk with gradient checkpointing
        if z is not None:
            processed_chunks = [checkpoint_fn(module, chunk, z) for chunk in m_chunks]
        else:
            processed_chunks = [checkpoint_fn(module, chunk) for chunk in m_chunks]
        if (not self.training) and m.shape[-3] > 5120:
            del m_chunks
            torch.cuda.empty_cache()
        # Concatenate the processed chunks back together
        m = torch.cat(processed_chunks, dim=0)
        if (not self.training) and m.shape[-3] > 5120:
            del processed_chunks
            torch.cuda.empty_cache()
        return m

    def inference_forward(
        self, m: torch.Tensor, z: torch.Tensor, chunk_size: int = 2048
    ) -> torch.Tensor:
        """
        Memory-efficient inference forward pass using inplace operations.
        
        Processes MSA in chunks using inplace updates to minimize memory allocation.
        This is particularly beneficial for large MSAs during inference.
        
        Args:
            m: MSA feature embedding (modified inplace).
                Shape: [..., N_msa_sampled, N_token, c_m]
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]
            chunk_size: Size of each chunk along the MSA sequence dimension.
                Defaults to 2048.

        Returns:
            torch.Tensor: Updated MSA embedding (same object as input, modified inplace).
                Shape: [..., N_msa_sampled, N_token, c_m]
        """
        num_msa = m.shape[-3]
        no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
        for i in range(no_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_msa)
            # Use inplace to save memory
            m[start:end, :, :] += self.msa_pair_weighted_averaging(
                m[start:end, :, :], z
            )
            m[start:end, :, :] += self.transition_m(m[start:end, :, :])
        return m

@register_license('bytedance2024')
class MSABlock(nn.Module):
    """
    Base MSA Block implementing Line6-Line13 in Algorithm 8 of AlphaFold3.
    
    Combines MSA and pair processing through:
        1. Outer product mean for MSA-to-pair communication
        2. MSA stack (weighted averaging + transition)
        3. Pair stack (PairformerBlock)
    
    The last block skips MSA processing to avoid unnecessary computation since
    MSA features are not used after the final block.
    
    Attributes:
        c_m: Channel dimension for MSA embedding.
        c_z: Channel dimension for pair embedding.
        c_hidden: Hidden dimension for outer product mean.
        is_last_block: Whether this is the final block in MSAModule.
        outer_product_mean_msa: Module for MSA-to-pair communication.
        msa_stack: Optional MSA processing stack (None if is_last_block).
        pair_stack: PairformerBlock for pair feature processing.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
    ) -> None:
        """
        Initializes the MSABlock module.
        
        Args:
            c_m: Channel dimension for MSA embedding. Defaults to 64.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            c_hidden: Hidden dimension for outer product mean. Defaults to 32.
            is_last_block: If True, skips MSA processing (no msa_stack).
                Defaults to False.
            msa_dropout: Dropout ratio for MSA processing. Defaults to 0.15.
            pair_dropout: Dropout ratio for pair processing. Defaults to 0.25.
            msa_chunk_size: Chunk size for MSA processing. Defaults to 2048.
            msa_max_size: Maximum MSA size for padding. Defaults to 16384.
        
        Returns:
            None
        """
        super(MSABlock, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block
        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=self.c_hidden
        )
        if not self.is_last_block:
            # MSA stack
            self.msa_stack = MSAStack(
                c_m=self.c_m,
                dropout=msa_dropout,
                msa_chunk_size=msa_chunk_size,
                msa_max_size=msa_max_size,
            )
        # Pair stack
        self.pair_stack = PairformerBlock(c_z=c_z, c_s=0, dropout=pair_dropout)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass through MSABlock.
        
        Processes MSA and pair features through:
            1. Outer product mean (MSA -> pair communication)
            2. MSA stack (if not last block)
            3. Pair stack (PairformerBlock)
        
        For the last block, returns None for MSA to signal no further MSA processing needed.
        
        Args:
            m: MSA feature embedding.
                Shape: [..., N_msa_sampled, N_token, c_m]
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]
            pair_mask: Mask for valid token pairs.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[Optional[torch.Tensor], torch.Tensor]: A tuple containing:
                - Updated MSA features [..., N_msa_sampled, N_token, c_m] or None if last block
                - Updated pair features [..., N_token, N_token, c_z]
        """
        # Communication
        if (not self.training) and z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        z = z + self.outer_product_mean_msa(
            m, inplace_safe=inplace_safe, chunk_size=chunk_size
        )
        if (not self.training) and z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z)
        # Pair stack
        _, z = self.pair_stack(
            s=None,
            z=z,
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if (not self.training) and (z.shape[-2] > 2000 or m.shape[-3] > 5120):
            torch.cuda.empty_cache()
        if not self.is_last_block:
            return m, z
        else:
            return None, z  # to ensure that `m` will not be used.


@register_license('bytedance2024')
class MSAModule(nn.Module):
    """
    Implements Algorithm 8 [MSAModule] in AlphaFold3.
    
    The MSA module processes Multiple Sequence Alignment (MSA) features to enhance
    pair representations with evolutionary information. It consists of:
        1. MSA feature embedding
        2. Multiple MSA blocks (outer product mean + MSA/pair processing)
        3. Integration with single representations
    
    Supports configurable MSA sampling strategies and chunk sizes for memory efficiency.
    
    Attributes:
        n_blocks: Number of MSABlock modules.
        c_m: Channel dimension for MSA embedding.
        c_s_inputs: Channel dimension for input single features.
        blocks_per_ckpt: Number of blocks per gradient checkpoint.
        msa_chunk_size: Chunk size for MSA processing.
        msa_max_size: Maximum MSA size for padding during training.
        input_feature_dims: Dictionary of MSA feature dimensions.
        msa_configs: Configuration dictionary for MSA processing.
        linear_no_bias_m: Linear projection for MSA features.
        linear_no_bias_s: Linear projection for single features.
        blocks: ModuleList of MSABlock instances.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = 1,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        msa_configs: dict = None,
    ) -> None:
        """
        Initializes the MSAModule with specified configurations.

        Args:
            n_blocks: Number of MSABlock modules in the stack. Defaults to 4.
            c_m: Channel dimension for MSA embedding. Defaults to 64.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            c_s_inputs: Channel dimension for single embedding from InputFeatureEmbedder.
                Defaults to 449.
            msa_dropout: Dropout ratio for MSA processing. Defaults to 0.15.
            pair_dropout: Dropout ratio for pair processing. Defaults to 0.25.
            blocks_per_ckpt: Number of MSA blocks in each activation checkpoint.
                Higher values use fewer checkpoints, trading memory for speed.
                If None, no gradient checkpointing is performed. Defaults to 1.
            msa_chunk_size: Chunk size for processing MSA sequences. Defaults to 2048.
            msa_max_size: Maximum MSA size for padding during training. Defaults to 16384.
            msa_configs: Configuration dictionary containing:
                - "enable": Whether to use MSA embedding.
                - "strategy": MSA sampling strategy (e.g., "random").
                - "sample_cutoff": Dict with "train" and "test" cutoff values.
                - "min_size": Dict with "train" and "test" minimum MSA sizes.
        
        Returns:
            None
        """
        super(MSAModule, self).__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.blocks_per_ckpt = blocks_per_ckpt
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size
        self.input_feature_dims = MSAFEATS_DIMS

        self.msa_configs = {
            "enable": msa_configs.get("enable", False),
            "strategy": msa_configs.get("strategy", "random"),
        }
        if "sample_cutoff" in msa_configs:
            self.msa_configs["train_cutoff"] = msa_configs["sample_cutoff"].get(
                "train", 512
            )
            self.msa_configs["test_cutoff"] = msa_configs["sample_cutoff"].get(
                "test", 16384
            )
            # the default msa_max_size is 16384 if not specified
            self.msa_max_size = self.msa_configs["train_cutoff"]
        if "min_size" in msa_configs:
            self.msa_configs["train_lowerb"] = msa_configs["min_size"].get("train", 1)
            self.msa_configs["test_lowerb"] = msa_configs["min_size"].get("test", 1)

        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            block = MSABlock(
                c_m=self.c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                msa_chunk_size=self.msa_chunk_size,
                msa_max_size=self.msa_max_size,
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ) -> list:
        """
        Prepares MSA blocks with partial application of common arguments.
        
        Creates a list of partially applied block functions with shared arguments,
        optionally adding cache clearing between blocks for large structures.
        
        Args:
            pair_mask: Mask for valid token pairs. Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
            use_lma: Whether to use low-memory attention.
            inplace_safe: Whether to use inplace operations for memory efficiency.
            chunk_size: Chunk size for memory-efficient operations.
            clear_cache_between_blocks: Whether to clear CUDA cache between blocks.
        
        Returns:
            list: List of partially applied block functions ready for execution.
        """
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def one_hot_fp32(
        self, tensor: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Performs one-hot encoding with float32 output (unlike F.one_hot which outputs int64).
        
        This is memory-efficient for large MSAs as float32 uses half the memory of int64.
        
        Args:
            tensor: Input tensor with integer class indices.
                Shape: [..., N_msa_sampled, N_token]
            num_classes: Number of classes for one-hot encoding.
            dtype: Output data type. Defaults to torch.float32.

        Returns:
            torch.Tensor: One-hot encoded tensor.
                Shape: [..., N_msa_sampled, N_token, num_classes]
        """
        shape = tensor.shape
        one_hot_tensor = torch.zeros(
            *shape, num_classes, dtype=dtype, device=tensor.device
        )
        one_hot_tensor.scatter_(len(shape), tensor.unsqueeze(-1), 1)
        return one_hot_tensor

    @register_license('odesign2025')
    def forward(
        self,
        input_feature: PairFormerInput,
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the MSAModule.
        
        Processes MSA features through multiple MSA blocks to enhance pair representations
        with evolutionary information. The module:
            1. Samples MSA sequences based on configured strategy and cutoffs
            2. Encodes MSA features (sequence, deletion, profile)
            3. Combines with single representations
            4. Processes through MSA blocks
        
        If n_blocks < 1 or no MSA data is available, returns the input pair features unchanged.
        
        Args:
            input_feature: PairFormerInput object containing input features. Expected keys:
                - "msa": MSA sequence indices [..., N_msa, N_token]
                - "msa_token_mask": Optional mask for MSA tokens [..., N_token]
                - Other MSA features (deletion values, has_deletion, profile)
            z: Pair feature embedding.
                Shape: [..., N_token, N_token, c_z]
            s_inputs: Single feature embedding from InputFeatureEmbedder.
                Shape: [..., N_token, c_s_inputs]
            pair_mask: Mask for valid token pairs.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: Updated pair feature embedding after MSA processing.
                Shape: [..., N_token, N_token, c_z]
        """
        # If n_blocks < 1, return z
        if self.n_blocks < 1:
            return z

        if "msa" not in input_feature:
            return z
        # Check msa shape!
        # IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)
        if input_feature["msa"].dim() < 2:
            return z
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature,
            dim_dict={feat_name: -2 for feat_name in self.input_feature_dims},
            cutoff=(
                self.msa_configs["train_cutoff"]
                if self.training
                else self.msa_configs["test_cutoff"]
            ),
            lower_bound=(
                self.msa_configs["train_lowerb"]
                if self.training
                else self.msa_configs["test_lowerb"]
            ),
            strategy=self.msa_configs["strategy"],
        )
        # pylint: disable=E1102
        if not self.training and z.shape[-2] > 2000:
            # msa_feat["msa"] is torch.int64, we convert it
            # to torch.float32 for saving half of the CUDA memory
            msa_feat["msa"] = self.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self.input_feature_dims["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self.input_feature_dims["msa"],
            )

        if input_feature.msa_token_mask is not None:
            msa_feat["msa"][..., input_feature['msa_token_mask'], :] = 0
            
        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [
                msa_feat[name].reshape(*target_shape, d)
                for name, d in self.input_feature_dims.items()
            ],
            dim=-1,
        )  # [..., N_msa_sample, N_token, 32 + 1 + 1]
        # Msa_feat is very large, if N_MSA=16384 and N_token=4000,
        # msa_feat["msa"] consumes about 16G CUDA memory, so we
        # need to clear cache to avoid OOM
        if not self.training:
            del msa_feat
            torch.cuda.empty_cache()
        # Line2
        msa_sample = self.linear_no_bias_m(msa_sample)

        # Auto broadcast [...,n_msa_sampled, n_token, c_m]
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        msa_sample, z = checkpoint_blocks(
            blocks,
            args=(msa_sample, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        if z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return z


@register_license('odesign2025')
class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 [TemplateEmbedder] in AlphaFold3.
    
    The TemplateEmbedder module processes template structural information to enhance
    pair representations. For each template structure:
        1. Constructs template features (distogram, unit vectors, backbone masks, residue types)
        2. Processes through Pairformer blocks
        3. Accumulates and averages template embeddings
    
    Templates provide structural constraints from homologous structures in the PDB.
    
    Attributes:
        n_blocks: Number of Pairformer blocks for processing each template.
        c: Hidden dimension for template processing.
        c_z: Channel dimension for pair embedding.
        input_feature1: Dictionary of template distance/geometry features.
        input_feature2: Dictionary of template residue type features.
        distogram: Configuration for distance binning.
        inf: Large value for masking.
        linear_no_bias_z: Linear projection for pair features.
        layernorm_z: Layer normalization for pair features.
        linear_no_bias_a: Linear projection for template features.
        pairformer_stack: Stack of Pairformer blocks for template processing.
        layernorm_v: Layer normalization for processed template features.
        linear_no_bias_u: Output projection to pair embedding dimension.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Initializes the TemplateEmbedder module.
        
        Args:
            n_blocks: Number of Pairformer blocks for processing each template.
                Defaults to 2.
            c: Hidden dimension for template feature processing. Defaults to 64.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            dropout: Dropout ratio for PairformerStack. Defaults to 0.25.
                Note: This value is not specified in Algorithm 16, so we use the
                default Pairformer dropout ratio.
            blocks_per_ckpt: Number of Pairformer blocks in each activation checkpoint.
                Higher values use fewer checkpoints, trading memory for speed.
                If None, no gradient checkpointing is performed. Defaults to None.
        
        Returns:
            None
        """
        super(TemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "b_template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "b_template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)
        
    
    def forward(
        self,
        input_feature: PairFormerInput,
        z: torch.Tensor,
        pair_mask: torch.Tensor = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through TemplateEmbedder implementing Algorithm 16 in AlphaFold3.
        
        Processes template structures through the following steps:
            1. Compute pairwise template features (distogram, unit vectors, masks)
            2. Concatenate with residue type features
            3. For each template:
                a. Combine with pair features and process through Pairformer
                b. Accumulate processed features
            4. Average over all templates and project to output
        
        If n_blocks < 1, returns 0 (no template processing).
        
        Args:
            input_feature: PairFormerInput object containing template features. Expected keys:
                - "template_distogram": Template distance histogram [..., N_templates, N_token, N_token, 39]
                - "template_backbone_frame_mask": Backbone frame validity [..., N_templates, N_token]
                - "template_unit_vector": Unit vectors between residues [..., N_templates, N_token, N_token, 3]
                - "template_pseudo_beta_mask": Pseudo-beta validity [..., N_templates, N_token]
                - "template_restype": Template residue types [..., N_templates, N_token, 32]
                - "asym_id": Asymmetric unit IDs for masking same-chain pairs [..., N_token]
            z: Pair feature embedding from the model.
                Shape: [..., N_token, N_token, c_z]
            pair_mask: Mask for valid token pairs. Defaults to None.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: Template embedding to be added to pair features.
                Shape: [..., N_token, N_token, c_z]
        """
        if self.n_blocks < 1:
            return 0
        
        # Get the number of templates from the first template feature
        # Assuming template features have shape [..., N_templates, N_token, N_token, feat_dim]
        # or [..., N_templates, N_token, feat_dim]
        n_templates = input_feature["template_distogram"].shape[-4]
        
        # Algorithm 16, Line 1: b_template_backbone_frame_mask_ij = f_ti * f_tj
        b_backbone_frame = (
            input_feature["template_backbone_frame_mask"][..., :, None] * 
            input_feature["template_backbone_frame_mask"][..., None, :]
        )  # [..., N_templates, N_token, N_token]
        
        # Algorithm 16, Line 2: b_template_pseudo_beta_mask_ij = f_ti * f_tj
        b_pseudo_beta = (
            input_feature["template_pseudo_beta_mask"][..., :, None] * 
            input_feature["template_pseudo_beta_mask"][..., None, :]
        )  # [..., N_templates, N_token, N_token]
        
        # Algorithm 16, Line 3: Concatenate template features
        # template_distogram: [..., N_templates, N_token, N_token, 39]
        # b_backbone_frame: [..., N_templates, N_token, N_token, 1]
        # template_unit_vector: [..., N_templates, N_token, N_token, 3]
        # b_pseudo_beta: [..., N_templates, N_token, N_token, 1]
        a_tij = torch.cat([
            input_feature["template_distogram"],
            b_backbone_frame.unsqueeze(-1),
            input_feature["template_unit_vector"],
            b_pseudo_beta.unsqueeze(-1)
        ], dim=-1)  # [..., N_templates, N_token, N_token, 44]
        
        # Algorithm 16, Line 4: Mask by same asym_id (same chain)
        # asym_id has shape [..., N_token]
        same_chain_mask = (
            input_feature["asym_id"][..., :, None] == 
            input_feature["asym_id"][..., None, :]
        ).float()  # [..., N_token, N_token]
        # Expand to match template dimension
        same_chain_mask = same_chain_mask.unsqueeze(-3)  # [..., 1, N_token, N_token]
        a_tij = a_tij * same_chain_mask.unsqueeze(-1)
        
        # Algorithm 16, Line 5: Concatenate template restype features
        # template_restype: [..., N_templates, N_token, 32]
        template_restype_i = input_feature["template_restype"][..., :, None, :]  # [..., N_templates, N_token, 1, 32]
        template_restype_j = input_feature["template_restype"][..., None, :, :]  # [..., N_templates, 1, N_token, 32]
        # Broadcast to [..., N_templates, N_token, N_token, 32] for both
        template_restype_i = template_restype_i.expand(-1, -1, input_feature["template_restype"].shape[-2], -1)
        template_restype_j = template_restype_j.expand(-1, input_feature["template_restype"].shape[-2], -1, -1)
        
        a_tij = torch.cat([a_tij, template_restype_i, template_restype_j], dim=-1)
        # Final shape: [..., N_templates, N_token, N_token, 44 + 32 + 32 = 108]
        
        # Algorithm 16, Line 6: Initialize u_ij = 0
        batch_dims = z.shape[:-3]
        n_token = z.shape[-2]
        u_ij = torch.zeros((*batch_dims, n_token, n_token, self.c), 
                        device=z.device, dtype=z.dtype)
        
        # Algorithm 16, Lines 7-11: Process each template
        for t in range(n_templates):
            # Line 8: v_ij = LinearNoBias(LayerNorm(z_ij)) + LinearNoBias(a_tij)
            v_ij = self.linear_no_bias_z(self.layernorm_z(z)) + self.linear_no_bias_a(a_tij[..., t, :, :, :])
            
            # Line 9: Process through PairformerStack
            # PairformerStack expects (single, pair) but we only have pair, so pass None for single
            _, v_ij = self.pairformer_stack(
                s=None,
                z=v_ij,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            
            # Line 10: u_ij += LayerNorm(v_ij)
            u_ij = u_ij + self.layernorm_v(v_ij)
        
        # Algorithm 16, Line 12: Average over templates
        u_ij = u_ij / n_templates
        
        # Algorithm 16, Line 13: Apply ReLU and linear projection
        u_ij = self.linear_no_bias_u(F.relu(u_ij))
        
        # Return the template embedding
        return u_ij
    

@register_license('odesign2025')
class ConstraintTemplateEmbedder(nn.Module):
    """
    Constraint-based template embedder for incorporating distance constraints into pair features.
    
    This module processes distance constraints (e.g., from experimental data or user-specified
    constraints) by:
        1. Converting distances to distogram bins
        2. Processing through Pairformer blocks
        3. Projecting to pair embedding dimension
    
    Unlike TemplateEmbedder which processes multiple template structures, this module handles
    single constraint maps (e.g., distance restraints for structure prediction).
    
    Attributes:
        n_blocks: Number of Pairformer blocks for processing constraints.
        c: Hidden dimension for constraint processing.
        c_z: Channel dimension for pair embedding.
        max_bin: Maximum distance for binning.
        min_bin: Minimum distance for binning.
        no_bins: Number of distance bins.
        inf: Large value for masking.
        linear_no_bias_z: Linear projection for pair features.
        layernorm_z: Layer normalization for pair features.
        linear_no_bias_a: Linear projection for constraint distogram.
        pairformer_stack: Stack of Pairformer blocks for constraint processing.
        layernorm_v: Layer normalization for processed constraint features.
        linear_no_bias_u: Output projection to pair embedding dimension.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Initializes the ConstraintTemplateEmbedder module.
        
        Args:
            n_blocks: Number of Pairformer blocks for processing constraints.
                Defaults to 2.
            c: Hidden dimension for constraint feature processing. Defaults to 64.
            c_z: Channel dimension for pair embedding. Defaults to 128.
            dropout: Dropout ratio for PairformerStack. Defaults to 0.25.
                Note: This value is not specified in the original algorithm, so we use
                the default Pairformer dropout ratio.
            blocks_per_ckpt: Number of Pairformer blocks in each activation checkpoint.
                Higher values use fewer checkpoints, trading memory for speed.
                If None, no gradient checkpointing is performed. Defaults to None.
        
        Returns:
            None
        """
        super(ConstraintTemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.max_bin = 50.75
        self.min_bin = 3.25
        self.no_bins = 39
        self.inf = 100000.0
        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=self.no_bins,
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)
    
    def _check_input(self, input_data: PairFormerInput) -> None:
        """
        Validates that input data contains required constraint features.
        
        Args:
            input_data: PairFormerInput object to validate.
        
        Returns:
            None
        
        Raises:
            ValueError: If constraint_feature is missing or None.
        """
        if not hasattr(input_data, "constraint_feature"):
            raise ValueError("Input data must contain constraint_feature.")
        elif input_data.constraint_feature is None:
            raise ValueError("Input data must contain constraint_feature and it cannot be None.")

    def forward(
        self,
        input_data: PairFormerInput,
        z: torch.Tensor,
        pair_mask: torch.Tensor = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through ConstraintTemplateEmbedder.
        
        Processes distance constraints to enhance pair representations:
            1. Bins constraint distances into distogram
            2. Combines with pair features and processes through Pairformer
            3. Projects to output dimension
        
        This enables incorporation of experimental constraints (e.g., from NMR, cross-linking,
        SAXS) or user-specified distance restraints into structure prediction.
        
        Args:
            input_data: PairFormerInput object containing:
                - constraint_feature: Distance constraints [..., N_token, N_token]
            z: Pair feature embedding from the model.
                Shape: [..., N_token, N_token, c_z]
            pair_mask: Mask for valid token pairs. Defaults to None.
                Shape: [..., N_token, N_token]
            use_memory_efficient_kernel: Whether to use memory-efficient attention kernel.
                Defaults to False.
            use_deepspeed_evo_attention: Whether to use DeepSpeed EVO attention optimization.
                Defaults to False.
            use_lma: Whether to use low-memory attention. Defaults to False.
            inplace_safe: Whether to use inplace operations for memory efficiency.
                Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: Constraint embedding to be added to pair features.
                Shape: [..., N_token, N_token, c_z]
        """
        # Assign distance to bins
        boundaries = torch.linspace(
            start=self.min_bin,
            end=self.max_bin,
            steps=self.no_bins - 1,
            device=z.device,
        )
        true_bins = torch.sum(
            input_data.constraint_feature > boundaries, dim=-1
        )  # range in [0, no_bins-1], shape = [..., N_token, N_token]
        distogram = F.one_hot(true_bins, self.no_bins).to(z.dtype)  #  [..., N_token, N_token, no_bins]
        # Line 8: v_ij = LinearNoBias(LayerNorm(z_ij)) + LinearNoBias(a_tij)
        v_ij = self.linear_no_bias_z(self.layernorm_z(z)) + self.linear_no_bias_a(distogram)
        # Line 9: Process through PairformerStack
        # PairformerStack expects (single, pair) but we only have pair, so pass None for single
        _, v_ij = self.pairformer_stack(
            s=None,
            z=v_ij,
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        # Line 10: u_ij += LayerNorm(v_ij)
        u_ij = self.layernorm_v(v_ij)
        # Algorithm 16, Line 13: Apply ReLU and linear projection
        u_ij = self.linear_no_bias_u(F.relu(u_ij))
        # Return the template embedding
        return u_ij
