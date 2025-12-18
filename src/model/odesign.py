# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ODesign: Main model architecture for protein structure prediction and generation.

This module implements the core ODesign model architecture, which is based on AlphaFold3's
design and adapted for both structure prediction and conditional generation tasks.

Main Components:
    ODesign: Main model class implementing the full forward pass including:
        - Input feature embedding
        - MSA (Multiple Sequence Alignment) processing
        - Pairformer module for residue and pair representations
        - Diffusion module for coordinate generation
        - Pairwise prediction heads (distogram, bond types)
        - Multi-cycle refinement
        - Symmetric permutation handling

Key Features:
    - Multi-cycle iterative refinement
    - Multiple model seed sampling for ensemble predictions
    - Training and inference modes with different noise schedulers
    - Support for constraint-based generation (distogram constraints)
    - Memory-efficient kernels and chunking for large structures
    - DeepSpeed EVO attention optimization
    - Automatic mixed precision (AMP) support
    - Symmetric permutation for handling molecular symmetries

Architecture Flow:
    1. Input Embedding: Convert input features to token and pair representations
    2. MSA Module: Process multiple sequence alignments
    3. Pairformer Stack: Iterative refinement of token and pair representations
    4. Pairwise Head: Predict distogram and bond types
    5. Diffusion Module: Generate 3D coordinates through denoising
    6. Permutation: Handle molecular symmetries

Training vs Inference:
    - Training: Single cycle randomly selected, gradient enabled only on last cycle
    - Inference: Fixed number of cycles, multiple model seeds for ensemble
"""

import copy
import random
import logging
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.license_register import register_license
from src.api.data_interface import OFeatureData, OLabelData
from src.api.model_interface import (
    PairFormerInput,
    PairFormerOutput,
    DiffusionInput,
    PermutationInput,
    LossInput,
    GroundTruth,
    PairwiseOutput,
    ODesignOutput
)

from src.model.modules.generator import (
    sample_diffusion,
    sample_diffusion_training,
)
from src.model.modules.schedulers import (
    TrainingNoiseEDMScheduler,
    InferenceNoiseEDMScheduler
)

from src.utils.openfold_local.model.primitives import LayerNorm
from src.utils.permutation.permutation import SymmetricPermutation
from src.utils.model.torch_utils import (
    autocasting_disable_decorator,
    cat_dict
)

from .modules.diffusion import DiffusionModule
from .modules.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from .modules.head import PairwiseHead
from .modules.pairformer import MSAModule, PairformerStack
from .modules.primitives import LinearNoBias
from .modules.pairformer import ConstraintTemplateEmbedder

logger = logging.getLogger(__name__)

@register_license('odesign2025')
class ODesign(nn.Module):
    """
    ODesign main model implementing AlphaFold3-inspired architecture.
    
    This class implements the complete pipeline for protein structure prediction and
    generation, including input embedding, MSA processing, pairformer iterations,
    diffusion-based coordinate generation, and pairwise predictions.
    
    The model operates in three modes:
        - train: Training mode with gradient computation and random cycle selection
        - inference: Inference mode with multiple model seeds for ensemble predictions
        - eval: Evaluation mode with symmetry-aware permutation matching
    
    Architecture Components:
        1. Input Feature Embedder: Embeds input features (sequence, positions, etc.)
        2. Relative Position Encoding: Encodes token-token relative positions
        3. MSA Module: Processes multiple sequence alignments
        4. Constraint Embedder (optional): Embeds constraint distograms for conditional generation
        5. Pairformer Stack: Iterative refinement of single and pair representations
        6. Diffusion Module: Denoising network for coordinate generation
        7. Pairwise Head: Predicts distogram and token bond types
    
    Attributes:
        configs: Model configuration object
        N_cycle: Number of refinement cycles
        N_model_seed: Number of model seeds for inference ensemble
        train_noise_schedulers: Noise schedulers for training
        inference_noise_schedulers: Noise schedulers for inference
        diffusion_batch_size: Batch size for diffusion sampling
        input_embedder: Input feature embedding module
        relative_position_encoding: Relative position encoding module
        msa_module: MSA processing module
        constraint_distogram_embedder: Constraint embedding module (optional)
        pairformer_stack: Pairformer refinement module
        diffusion_module: Diffusion denoising module
        pairwise_head: Pairwise prediction head
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_inputs: Input feature channel dimension
    
    Multi-Cycle Refinement:
        The model iteratively refines representations through N_cycle iterations:
        - Each cycle processes: z_init -> MSA -> Pairformer -> updated (s, z)
        - Gradient is only computed on the last cycle during training
        - Previous cycle outputs are used to initialize the next cycle
    
    Diffusion Process:
        - Training: Uses ground truth coordinates to generate noisy samples for denoising
        - Inference: Starts from random noise and iteratively denoises to predict structure
    """

    def __init__(self, configs) -> None:
        """
        Initialize the ODesign model.

        Args:
            configs: Configuration object containing all model hyperparameters including:
                - model.N_cycle: Number of refinement cycles
                - model.N_model_seed: Number of model seeds for inference
                - model.train_noise_schedulers: Training noise scheduler configs
                - model.inference_noise_schedulers: Inference noise scheduler configs
                - model.input_embedder: Input embedder configuration
                - model.relative_position_encoding: Relative position encoding config
                - model.msa_module: MSA module configuration
                - model.constraint_distogram_embedder: Constraint embedder config (optional)
                - model.pairformer: Pairformer stack configuration
                - model.diffusion_module: Diffusion module configuration
                - model.pairwise_head: Pairwise head configuration
                - model.c_s, c_z, c_s_inputs: Channel dimensions
                - diffusion_batch_size: Diffusion sampling batch size
                - data_condition: Data conditioning types (e.g., ['constraint_distogram'])

        Returns:
            None
        """
        super(ODesign, self).__init__()
        self.configs = configs
        
        # Some constants
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed

        # Diffusion scheduler
        self.train_noise_schedulers = {
            "coordinate" : TrainingNoiseEDMScheduler(**configs.model.train_noise_schedulers.coordinate)
        }
        self.inference_noise_schedulers = {
            "coordinate": InferenceNoiseEDMScheduler(**configs.model.inference_noise_schedulers.coordinate)
        }
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )

        if set(self.configs.data_condition) & set(['constraint_distogram']):
            self.constraint_distogram_embedder = ConstraintTemplateEmbedder(**configs.model.constraint_distogram_embedder)
            
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)        
        self.pairwise_head = PairwiseHead(**configs.model.pairwise_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.model.c_s,
            configs.model.c_z,
            configs.model.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_data: PairFormerInput,
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
    ) -> PairFormerOutput:
        """
        Forward pass from input features to pairformer output representations.
        
        This method processes input features through multiple cycles of refinement:
        1. Input Embedding: Embed input features to s_inputs
        2. Initialization: Create initial single (s) and pair (z) representations
        3. Multi-cycle Refinement: Iteratively refine s and z through:
            - Constraint embedding (optional)
            - MSA module processing
            - Pairformer stack processing
        
        The multi-cycle strategy allows the model to iteratively refine its predictions,
        with gradient computation only on the last cycle during training for efficiency.

        Args:
            input_data: PairFormerInput object containing:
                - residue_index: Residue indices [N_token]
                - token_bonds: Token bond connectivity [N_token, N_token]
                - And other input features (see PairFormerInput for complete list)
            N_cycle: Number of refinement cycles to perform (typically 1-4)
            inplace_safe: Whether it is safe to use inplace operations for memory efficiency.
                Set to True during inference when gradients are not needed. Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Used to reduce memory
                consumption for large structures. Defaults to None (no chunking).

        Returns:
            PairFormerOutput: Object containing:
                - s_inputs: Initial embedded single representations [N_token, c_s_inputs]
                - s: Refined single representations [N_token, c_s]
                - z: Refined pair representations [N_token, N_token, c_z]
                
        Note:
            - For small structures (N_token <= 16), DeepSpeed EVO attention is disabled
            - Gradient is only enabled on the last cycle during training
            - Inplace operations are used when safe to reduce memory usage
        """
        N_token = input_data.residue_index.shape[-1]
        if N_token <= 16:
            # Deepspeed_evo_attention do not support token <= 16
            deepspeed_evo_attention_condition_satisfy = False
        else:
            deepspeed_evo_attention_condition_satisfy = True

        s_inputs = self.input_embedder(
            input_data, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]

        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_data)
            z_init += self.linear_no_bias_token_bond(
                input_data.token_bonds.unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(input_data)
            z_init = z_init + self.linear_no_bias_token_bond(
                input_data.token_bonds.unsqueeze(dim=-1)
            )
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if set(self.configs.data_condition) & set(['constraint_distogram']):
                        z += self.constraint_distogram_embedder(
                            input_data,
                            z,
                            pair_mask=None,
                            use_memory_efficient_kernel=self.configs.model.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.model.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.model.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_data,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.model.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.model.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.model.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if set(self.configs.data_condition) & set(['constraint_distogram']):
                        z = z + self.constraint_distogram_embedder(
                            input_data,
                            z,
                            pair_mask=None,
                            use_memory_efficient_kernel=self.configs.model.use_memory_efficient_kernel,
                            use_deepspeed_evo_attention=self.configs.model.use_deepspeed_evo_attention
                            and deepspeed_evo_attention_condition_satisfy,
                            use_lma=self.configs.model.use_lma,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_data,
                        z,
                        s_inputs,
                        pair_mask=None,
                        use_memory_efficient_kernel=self.configs.model.use_memory_efficient_kernel,
                        use_deepspeed_evo_attention=self.configs.model.use_deepspeed_evo_attention
                        and deepspeed_evo_attention_condition_satisfy,
                        use_lma=self.configs.model.use_lma,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    use_memory_efficient_kernel=self.configs.model.use_memory_efficient_kernel,
                    use_deepspeed_evo_attention=self.configs.model.use_deepspeed_evo_attention
                    and deepspeed_evo_attention_condition_satisfy,
                    use_lma=self.configs.model.use_lma,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return PairFormerOutput(s_inputs=s_inputs, s=s, z=z)

    def main_inference_loop(
        self,
        pairformer_input: PairFormerInput,
        diffusion_input: DiffusionInput,
        permutation_input: PermutationInput,
        ground_truth: GroundTruth,
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[ODesignOutput, dict[str, Any], dict[str, Any]]:
        """
        Main inference loop with multiple model seeds for ensemble predictions.
        
        This method runs the inference loop multiple times with different model seeds
        and aggregates the results. Each model seed produces independent predictions
        that can be used for confidence estimation or ensemble averaging.
        
        Inference Pipeline:
            1. For each model seed:
                a. Run pairformer to get representations (s, z)
                b. Predict distogram and bond types from pairwise head
                c. Sample coordinates from diffusion model
                d. Apply symmetric permutation (if provided)
            2. Concatenate results from all model seeds

        Args:
            pairformer_input: PairFormerInput object containing input features for pairformer
            diffusion_input: DiffusionInput object containing features for diffusion module
            permutation_input: PermutationInput object for handling molecular symmetries
            ground_truth: GroundTruth object containing ground truth labels (optional for inference)
            N_cycle: Number of refinement cycles in pairformer
            mode: Mode of operation ('inference' or 'eval')
            inplace_safe: Whether to use inplace operations for memory efficiency. 
                Defaults to True for inference.
            chunk_size: Chunk size for memory-efficient attention computation. 
                Defaults to 4.
            N_model_seed: Number of independent model seeds for ensemble predictions. 
                Defaults to 1.
            symmetric_permutation: SymmetricPermutation object for handling molecular 
                symmetries in evaluation mode. Defaults to None.

        Returns:
            ODesignOutput: Object containing:
                - coordinate: Predicted 3D coordinates [N_model_seed, N_sample, N_atom, 3]
                - token_bond_type_logits: Bond type logits [N_model_seed, N_token, N_token, N_bond_types]
                - token_bond_gen_mask: Bond generation mask [N_model_seed, N_token, N_token]
                - distogram: Distance distribution logits [N_model_seed, N_token, N_token, N_bins]
                
        Note:
            - Multiple model seeds provide uncertainty estimates and ensemble predictions
            - Memory is managed by clearing cache for large structures (N_token > 2000)
            - Symmetric permutation is applied in eval mode to match ground truth symmetries
        """
        model_outputs = []
        for _ in range(N_model_seed):
            model_output = self._main_inference_loop(
                pairformer_input=pairformer_input,
                diffusion_input=diffusion_input,
                permutation_input=permutation_input,
                ground_truth=ground_truth,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
            )
            model_outputs.append(model_output)
        
        return ODesignOutput(
            coordinate=cat_dict(model_outputs, "coordinate"),
            token_bond_type_logits=cat_dict(model_outputs, "token_bond_type_logits"),
            token_bond_gen_mask=cat_dict(model_outputs, "token_bond_gen_mask"),
        )

    def _main_inference_loop(
        self,
        pairformer_input: PairFormerInput,
        diffusion_input: DiffusionInput,
        permutation_input: PermutationInput,
        ground_truth: GroundTruth,
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Single model seed inference loop for structure prediction.
        
        This internal method performs one complete inference pass:
        1. Pairformer Module: Generate single and pair representations
        2. Pairwise Head: Predict distogram and bond types
        3. Diffusion Module: Sample 3D coordinates through denoising
        4. Permutation (optional): Match predicted symmetries to ground truth
        
        Memory Optimization:
            - Deletes MSA and template features after pairformer to save memory
            - Clears CUDA cache for large structures (N_token > 2000)
            - Uses inplace operations when safe

        Args:
            pairformer_input: PairFormerInput object with input features
            diffusion_input: DiffusionInput object for coordinate generation
            permutation_input: PermutationInput object for symmetry handling
            ground_truth: GroundTruth object with ground truth labels (optional)
            N_cycle: Number of pairformer refinement cycles
            mode: Operation mode ('inference' or 'eval')
            inplace_safe: Whether to use inplace operations. Defaults to True.
            chunk_size: Chunk size for memory-efficient operations. Defaults to 4.
            symmetric_permutation: SymmetricPermutation object for matching
                predicted symmetries to ground truth. Used in eval mode. Defaults to None.

        Returns:
            ODesignOutput: Object containing model predictions:
                - coordinate: Predicted 3D coordinates [N_sample, N_atom, 3]
                - distogram: Distance distribution logits [N_token, N_token, N_bins]
                - token_bond_type_logits: Bond type predictions [N_token, N_token, N_bond_types]
                - token_bond_gen_mask: Valid bond mask [N_token, N_token]
                
        Note:
            - This is called by main_inference_loop for each model seed
            - AMP (Automatic Mixed Precision) is disabled for certain operations
            - Symmetric permutation ensures predicted coordinates match ground truth symmetries
        """
        N_token = pairformer_input.residue_index.shape[-1]

        model_output = ODesignOutput()

        pairformer_output = self.get_pairformer_output(
            input_data=pairformer_input,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        pairwise_output: PairwiseOutput = autocasting_disable_decorator(
            self.configs.model.skip_amp.sample_diffusion_training
        )(self.pairwise_head)(
            pairformer_output
        )
        model_output.update(
            {
                "distogram": pairwise_output.distogram,
                "token_bond_type_logits": pairwise_output.token_bond_type_logits,
                "token_bond_gen_mask": pairformer_input.token_bond_gen_mask
            }
        )

        if mode == "inference":
            keys_to_delete = []
            for key in pairformer_input.keys():
                if "template_" in key or key in [
                    "msa",
                    "has_deletion",
                    "deletion_value",
                    "profile",
                    "deletion_mean",
                    "token_bonds",
                ]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                delattr(pairformer_input, key)
            torch.cuda.empty_cache()

        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        diffusion_output = autocasting_disable_decorator(
            self.configs.model.skip_amp.sample_diffusion
        )(sample_diffusion)(
            denoise_net=self.diffusion_module,
            input_data=diffusion_input,
            input_embedding=pairformer_output,
            ground_truth=ground_truth,
            N_sample=self.configs.model.sample_diffusion.N_sample,
            N_step=self.configs.model.sample_diffusion.N_step,
            attn_chunk_size=self.configs.model.sample_diffusion.attn_chunk_size,
            diffusion_chunk_size=self.configs.model.sample_diffusion.diffusion_chunk_size,
            noise_schedulers=self.inference_noise_schedulers,
            inplace_safe=inplace_safe,
        )

        model_output.update(
            {
                "coordinate": diffusion_output.x_denoised,
            }
        )

        if mode == "inference" and N_token > 2000:
            torch.cuda.empty_cache()

        # Permutation: when label is given, permute coordinates and other heads
        if symmetric_permutation is not None:
            model_output, _ = symmetric_permutation.permute_inference_output(
                input_data=permutation_input,
                model_output=model_output,
                ground_truth=ground_truth,
            )

        return model_output

    def main_train_loop(
        self,
        pairformer_input: PairFormerInput,
        diffusion_input: DiffusionInput,
        permutation_input: PermutationInput,
        ground_truth: GroundTruth,
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[ODesignOutput, GroundTruth, dict[str, Any]]:
        """
        Main training loop with diffusion-based coordinate denoising.
        
        Training Pipeline:
            1. Pairformer Module: 
                - Generate single (s) and pair (z) representations
                - Multi-cycle refinement with gradient only on last cycle
            
            2. Pairwise Module:
                - Predict distogram from pair representations
                - Predict token bond types
            
            3. Denoising Module:
                - Add noise to ground truth coordinates
                - Train diffusion model to denoise coordinates
                - Support conditional dropout for robustness
            
            4. Permutation Module:
                - Permute predicted coordinates to match ground truth
                - Handle symmetric atoms and chains
                - Compute optimal permutation for loss calculation
        
        Key Training Features:
            - Random conditioning dropout with configurable rate
            - Symmetric permutation matching for proper loss computation
            - Efficient gradient computation (only last cycle)
            - Support for multiple diffusion samples per batch

        Args:
            pairformer_input: PairFormerInput object containing input features
            diffusion_input: DiffusionInput object for diffusion module
            permutation_input: PermutationInput object for symmetry handling
            ground_truth: GroundTruth object with ground truth coordinates and labels
            N_cycle: Number of pairformer refinement cycles
            symmetric_permutation: SymmetricPermutation object for handling molecular
                symmetries. Required for training.
            inplace_safe: Whether to use inplace operations. Should be False during
                training to preserve computation graph. Defaults to False.
            chunk_size: Chunk size for memory-efficient operations. Defaults to None
                (no chunking during training).

        Returns:
            tuple: Tuple containing three elements:
                - model_output: ODesignOutput object with predictions:
                    - coordinate: Denoised coordinates [N_sample, N_atom, 3]
                    - noise_level: Noise levels used for training [N_sample]
                    - distogram: Predicted distance distributions [N_token, N_token, N_bins]
                    - token_bond_type_logits: Bond type predictions [N_token, N_token, N_bond_types]
                    - token_bond_gen_mask: Valid bond mask [N_token, N_token]
                - ground_truth: GroundTruth object (may be permuted to match predictions)
                - log_dict: Dictionary containing logging information:
                    - Permutation statistics
                    - Any other training metrics
                    
        Note:
            - Ground truth coordinates are used to generate noisy training samples
            - Conditioning dropout helps model learn unconditional generation
            - Symmetric permutation is crucial for correct loss computation
            - AMP is disabled for diffusion module for numerical stability
        """

        log_dict = {}
        model_output = ODesignOutput()

        # Pairformer Module: Encode information to residue level embedding and pairwise embedding
        pairformer_output = self.get_pairformer_output(
            input_data=pairformer_input,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        # End of Pairformer Module

        # Pairwise Module: Decode pairwise embedding to distogram and bond type logits
        pairwise_output: PairwiseOutput = autocasting_disable_decorator(
            self.configs.model.skip_amp.sample_diffusion_training
        )(self.pairwise_head)(
            pairformer_output
        )
        model_output.update(
            {
                "distogram": pairwise_output.distogram,
                "token_bond_type_logits": pairwise_output.token_bond_type_logits,
                "token_bond_gen_mask": pairformer_input.token_bond_gen_mask
            }
        )
        # End of Pairwise Module

        # Denoising Module: Use ground truth coords to generate noisy samples and perform denoising
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        diffusion_output = autocasting_disable_decorator(
            self.configs.model.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_schedulers=self.train_noise_schedulers,
            denoise_net=self.diffusion_module,
            input_data=diffusion_input,
            input_embedding=pairformer_output,
            ground_truth=ground_truth,
            N_sample=self.diffusion_batch_size,
            diffusion_chunk_size=self.configs.model.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
        )
        model_output.update(
            {
                "coordinate": diffusion_output.x_denoised,
                "noise_level": diffusion_output.sigma,
            }
        )
        # End of Denoising Module

        # Permutation Module: Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        model_output, perm_log_dict, _, _ = (
            symmetric_permutation.permute_diffusion_sample_to_match_label(
                permutation_input, model_output, ground_truth, stage="train"
            )
        )
        log_dict.update(perm_log_dict)
        # End of Permutation Module

        return model_output, ground_truth, log_dict

    @register_license('odesign2025')
    def forward(
        self,
        feature_data: OFeatureData,
        label_full_data: OLabelData,
        label_data: OLabelData,
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the ODesign model supporting multiple operation modes.
        
        This is the main entry point for the model, handling three distinct modes:
        
        1. Training Mode ('train'):
            - Randomly selects number of cycles (1 to N_cycle)
            - Computes gradients for backpropagation
            - Uses ground truth coordinates for diffusion training
            - Requires symmetric_permutation for loss computation
            - Single model seed
        
        2. Inference Mode ('inference'):
            - Uses fixed N_cycle cycles
            - Multiple model seeds for ensemble predictions
            - Starts from random noise for coordinate generation
            - No symmetric permutation needed
            - Memory-optimized (deletes unused features, clears cache)
        
        3. Evaluation Mode ('eval'):
            - Uses fixed N_cycle cycles
            - Multiple model seeds for ensemble predictions
            - Applies symmetric permutation to match ground truth
            - Uses full (uncropped) labels for evaluation
            - Suitable for benchmarking and validation
        
        Mode-Specific Behavior:
            - train: Requires labels, uses random cycles, single seed
            - inference: No labels needed, fixed cycles, multiple seeds
            - eval: Requires labels, fixed cycles, multiple seeds, with permutation

        Args:
            feature_data: OFeatureData object containing input features:
                - Sequence information (residue_index, token_index, etc.)
                - Coordinates and masks (ref_pos, ref_mask, etc.)
                - MSA features (msa, msa_mask, etc.)
                - Template features (template_*, etc.)
                - Bonding information (token_bonds, etc.)
            label_full_data: OLabelData object with full (uncropped) labels:
                - Full complex coordinates before cropping
                - Used for multi-chain permutation in evaluation
            label_data: OLabelData object with cropped labels:
                - Ground truth coordinates [N_atom, 3]
                - Ground truth masks and features
                - Used for training and evaluation
            mode: Operation mode, one of:
                - 'train': Training with gradient computation
                - 'inference': Inference without ground truth
                - 'eval': Evaluation with ground truth matching
                Defaults to 'inference'.
            current_step: Current training step number. Used to randomly sample
                N_cycle in training mode. Required for training. Defaults to None.
            symmetric_permutation: SymmetricPermutation object for handling molecular
                symmetries. Required for training and eval modes. Defaults to None.

        Returns:
            tuple: Tuple containing three elements:
                - model_output: ODesignOutput object with predictions:
                    - coordinate: 3D coordinates [N_model_seed, N_sample, N_atom, 3]
                    - distogram: Distance distributions [N_model_seed, N_token, N_token, N_bins]
                    - token_bond_type_logits: Bond predictions [N_model_seed, N_token, N_token, N_types]
                    - token_bond_gen_mask: Valid bonds [N_model_seed, N_token, N_token]
                    - noise_level: Noise levels (training only) [N_model_seed, N_sample]
                - ground_truth: GroundTruth object (may be permuted in training)
                - loss_input: LossInput object with features needed for loss computation
                
        Raises:
            AssertionError: If mode-specific requirements are not met:
                - Training mode requires self.training to be True
                - Training mode requires label_data
                - Training mode requires symmetric_permutation
                
        Note:
            - Inplace operations are used in inference/eval for memory efficiency
            - Chunking is enabled in inference/eval for large structures
            - Memory is actively managed (cache clearing) for inference
            - Random seed for cycle selection is based on current_step in training
        """
        assert mode in ["train", "inference", "eval"]

        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.model.chunk_size if inplace_safe else None

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            if mode == 'train':
                assert self.training
            assert label_data is not None
            assert symmetric_permutation is not None

            def prepare_training_inputs(feature_data, label_data):
                """
                Prepare input objects for training mode.
                
                Extracts and organizes features into specialized input objects for
                different model components.

                Args:
                    feature_data: OFeatureData with all input features
                    label_data: OLabelData with ground truth labels

                Returns:
                    tuple: (pairformer_input, diffusion_input, permutation_input, 
                           loss_input, ground_truth)
                """
                pairformer_input = PairFormerInput.from_feature_data(feature_data)
                diffusion_input = DiffusionInput.from_feature_data(feature_data)
                permutation_input = PermutationInput.from_feature_data(feature_data)
                loss_input = LossInput.from_feature_data(feature_data)
                ground_truth = GroundTruth.from_label_data(label_data)
                return pairformer_input, diffusion_input, permutation_input, loss_input, ground_truth

            (
                pairformer_input, 
                diffusion_input, 
                permutation_input, 
                loss_input, 
                ground_truth
            ) = prepare_training_inputs(feature_data, label_data)

            model_output, ground_truth, _ = self.main_train_loop(
                pairformer_input=pairformer_input,
                diffusion_input=diffusion_input,
                permutation_input=permutation_input,
                ground_truth=ground_truth,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        elif mode == "inference":
            def prepare_inference_inputs(feature_data, label_data):
                """
                Prepare input objects for inference mode.
                
                Similar to training preparation but sets permutation_input and loss_input
                to None since they are not needed for pure inference.

                Args:
                    feature_data: OFeatureData with all input features
                    label_data: OLabelData with ground truth labels (may be None)

                Returns:
                    tuple: (pairformer_input, diffusion_input, None, None, ground_truth)
                """
                pairformer_input = PairFormerInput.from_feature_data(feature_data)
                diffusion_input = DiffusionInput.from_feature_data(feature_data)
                permutation_input = None
                loss_input = None
                ground_truth = GroundTruth.from_label_data(label_data)
                return pairformer_input, diffusion_input, permutation_input, loss_input, ground_truth

            (
                pairformer_input, 
                diffusion_input, 
                permutation_input, 
                loss_input, 
                ground_truth
            ) = prepare_inference_inputs(feature_data, label_data)

            model_output = self.main_inference_loop(
                pairformer_input=pairformer_input,
                diffusion_input=diffusion_input,
                permutation_input=permutation_input,
                ground_truth=ground_truth,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
            )
        elif mode == "eval":
            if label_data is not None:
                assert (
                    label_data["coordinate"].size()
                    == label_full_data["coordinate"].size()
                ), print(f'{label_data["coordinate"].size()} != {label_full_data["coordinate"].size()}')
                label_data.update(label_full_data)

            def prepare_eval_inputs(feature_data, label_data):
                """
                Prepare input objects for evaluation mode.
                
                Similar to training preparation. Includes permutation_input and loss_input
                for symmetry-aware evaluation metrics.

                Args:
                    feature_data: OFeatureData with all input features
                    label_data: OLabelData with ground truth labels (merged with full labels)

                Returns:
                    tuple: (pairformer_input, diffusion_input, permutation_input, 
                           loss_input, ground_truth)
                """
                pairformer_input = PairFormerInput.from_feature_data(feature_data)
                diffusion_input = DiffusionInput.from_feature_data(feature_data)
                permutation_input = PermutationInput.from_feature_data(feature_data)
                loss_input = LossInput.from_feature_data(feature_data)
                ground_truth = GroundTruth.from_label_data(label_data)
                return pairformer_input, diffusion_input, permutation_input, loss_input, ground_truth

            (
                pairformer_input, 
                diffusion_input, 
                permutation_input, 
                loss_input, 
                ground_truth
            ) = prepare_eval_inputs(feature_data, label_data)

            model_output = self.main_inference_loop(
                pairformer_input=pairformer_input,
                diffusion_input=diffusion_input,
                permutation_input=permutation_input,
                ground_truth=ground_truth,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=symmetric_permutation,
            )

        return model_output, ground_truth, loss_input
