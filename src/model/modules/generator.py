
"""
Diffusion generator module for coordinate sampling and training.

This module implements diffusion-based structure generation algorithms for both
inference and training, following the AlphaFold3 approach with conditional generation
support.

Main Functions:
    sample_diffusion: Implements Algorithm 18 from AF3 for inference-time sampling
        - Iterative denoising from noise to structure
        - Supports conditional generation with fixed atom positions
        - Memory-efficient chunking for multiple samples
        - Random augmentation for SE(3) equivariance
    
    sample_diffusion_training: Training-time diffusion sampling
        - Adds noise to ground truth coordinates
        - Trains denoising network to predict clean coordinates
        - Supports conditional dropout for robust generation
        - Handles symmetric permutation for proper loss computation

Key Features:
    - Conditional Generation: Can fix certain atom positions while generating others
    - SE(3) Equivariance: Random rotation/translation augmentation during sampling
    - Memory Efficiency: Chunking support for large structures and multiple samples
    - Predictor-Corrector Sampling: Improved sampling quality through correction steps
    - Noise Scheduling: Separate schedulers for training and inference

Diffusion Process:
    1. Training: x_clean -> add_noise -> x_noisy -> denoise_net -> x_denoised
    2. Inference: random_noise -> iterative_denoising -> x_final
    
Conditional Generation:
    - Specified atoms can be fixed to known positions (e.g., ligand binding sites)
    - Conditioning mask determines which atoms are fixed
    - Fixed atoms are preserved throughout the sampling process
"""

from typing import Any, Callable, Optional

import torch

from src.utils.license_register import register_license
from src.utils.model.misc import (
    centre_random_augmentation,
    reverse_centre_random_augmentation,
    check_condition_atom_coords,
)

from src.api.model_interface import (
    DiffusionInput,
    PairFormerOutput,
    DiffusionOutput,
    GroundTruth
)
from src.model.modules.schedulers import (
    TrainingNoiseScheduler,
    InferenceNoiseScheduler,
    InferenceNoiseEDMScheduler
)
        

@register_license('odesign2025')
def sample_diffusion(
    denoise_net: torch.nn.Module,
    input_data: DiffusionInput,
    input_embedding: PairFormerOutput,
    ground_truth: GroundTruth,
    noise_schedulers: dict[str, InferenceNoiseScheduler],
    N_sample: int = 1,
    N_step: int = 200,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> DiffusionOutput:
    """
    Sample structures from diffusion model using iterative denoising (Algorithm 18 in AF3).
    
    This function performs inference-time structure generation through iterative denoising,
    starting from random noise and gradually refining to predict 3D coordinates. It supports
    conditional generation where certain atom positions can be fixed.
    
    Sampling Process:
        1. Initialize noise: Sample from Gaussian or use conditional positions
        2. For each denoising step (t from T to 0):
            a. Apply random SE(3) augmentation (rotation/translation)
            b. Add corrector noise at current noise level
            c. Denoise with neural network
            d. Update coordinates using predictor-corrector formula
            e. Reverse augmentation to original frame
            f. Enforce conditioning constraints (fix specified atoms)
        3. Return final denoised coordinates
    
    Conditional Generation:
        - Atoms marked in input_data.is_condition_atom are fixed to ground_truth positions
        - Useful for tasks like: ligand docking, partial structure completion, motif scaffolding
        - Conditioning is enforced at every denoising step
    
    Memory Management:
        - Supports chunking multiple samples to reduce memory usage
        - Each chunk is processed independently through all denoising steps

    Args:
        denoise_net: Denoising neural network (DiffusionModule) that predicts clean
            coordinates from noisy coordinates. Must implement forward pass taking:
            - x_noisy: Noisy coordinates [N_sample, N_atom, 3]
            - t_hat_noise_level: Noise level [N_sample]
            - input_data: DiffusionInput object
            - input_embedding: PairFormerOutput object
        input_data: DiffusionInput object containing:
            - atom_to_token_idx: Atom to token mapping [N_atom]
            - is_condition_atom: Conditioning mask [N_atom], True for fixed atoms
            - Other atom-level features required by denoise_net
        input_embedding: PairFormerOutput object containing:
            - s_inputs: Initial single representations [N_token, c_s_inputs]
            - s: Refined single representations [N_token, c_s]
            - z: Pair representations [N_token, N_token, c_z]
        ground_truth: GroundTruth object containing:
            - coordinate: Ground truth coordinates [N_atom, 3], used for conditioning
            - coordinate_mask: Valid coordinate mask [N_atom]
        noise_schedulers: Dictionary mapping scheduler names to InferenceNoiseScheduler
            objects. Must contain 'coordinate' scheduler for coordinate denoising.
        N_sample: Number of independent samples to generate. Multiple samples provide
            diversity and uncertainty estimation. Defaults to 1.
        N_step: Number of denoising steps. More steps generally improve quality but
            increase computation. Typical values: 50-200. Defaults to 200.
        diffusion_chunk_size: Chunk size for processing multiple samples. If None,
            all samples are processed together. Use chunking to reduce memory usage
            for large N_sample. Defaults to None.
        inplace_safe: Whether to use inplace operations for memory efficiency. Should
            be True during inference. Defaults to False.
        attn_chunk_size: Chunk size for attention operations in denoise_net. Used to
            reduce memory for large structures. Defaults to None (no chunking).

    Returns:
        DiffusionOutput: Object containing:
            - x_denoised: Generated 3D coordinates [N_sample, N_atom, 3]
                Multiple samples provide ensemble predictions
                Conditioned atoms exactly match ground_truth positions
            - sigma: None (not used in inference)
    
    Note:
        - SE(3) augmentation (random rotation/translation) is applied at each step
        - Conditioning atoms are checked to ensure they match ground truth exactly
        - Memory scales with N_sample * N_atom, use chunking for large structures
        - Noise schedule is automatically set based on N_step
        
    Example:
        For conditional generation with 5 samples and 100 denoising steps:
        >>> output = sample_diffusion(
        ...     denoise_net=diffusion_module,
        ...     input_data=diffusion_input,  # with is_condition_atom mask
        ...     input_embedding=pairformer_output,
        ...     ground_truth=ground_truth,  # contains fixed atom positions
        ...     noise_schedulers=inference_schedulers,
        ...     N_sample=5,
        ...     N_step=100,
        ...     diffusion_chunk_size=2,  # Process 2 samples at a time
        ... )
        >>> print(output.x_denoised.shape)  # [5, N_atom, 3]
    """
    N_atom = input_data.atom_to_token_idx.size(-1)
    device = input_embedding.s_inputs.device
    dtype = input_embedding.s_inputs.dtype

    coord_noise_scheduler: InferenceNoiseEDMScheduler = noise_schedulers['coordinate']
    coord_noise_scheduler.set_noise_schedule(
        N_step=N_step, device=device, dtype=dtype
    )

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        """
        Internal function to process a chunk of samples through the full denoising loop.
        
        This handles the iterative denoising for a subset of samples, processing all
        N_step denoising iterations for chunk_n_sample independent samples.

        Args:
            chunk_n_sample: Number of samples in this chunk
            inplace_safe: Whether to use inplace operations

        Returns:
            torch.Tensor: Denoised coordinates [chunk_n_sample, N_atom, 3]
        """
        # Initialize conditioning mask: atoms that should be fixed to ground truth
        # Combines valid coordinates mask with user-specified conditioning mask
        # Shape: [chunk_n_sample, N_atom]
        condition_mask = torch.logical_and(
            ground_truth.coordinate_mask, input_data.is_condition_atom
        ).unsqueeze(dim=0).expand(chunk_n_sample, -1)
        
        # Expand ground truth coordinates for all samples in chunk
        # Shape: [chunk_n_sample, N_atom, 3]
        x_gt = (
            ground_truth.coordinate
            .unsqueeze(dim=0)
            .expand(chunk_n_sample, -1, -1)
        )

        # Initialize noise: Sample from Gaussian for non-conditioned atoms,
        # use ground truth for conditioned atoms
        # Shape: [chunk_n_sample, N_atom, 3]
        x_l = coord_noise_scheduler.sample_init_noise_with_condition(
            size=(chunk_n_sample, N_atom, 3),
            x_gt=x_gt,
            condition_mask=condition_mask,
            device=device,
            dtype=dtype,
        )

        # Iterative denoising loop: gradually reduce noise from high to low
        for step_idx in range(N_step):
            # Apply random SE(3) augmentation (rotation + translation)
            # This makes the model SE(3) equivariant
            x_l, trans, rot, x_center = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1, dtype=dtype)
            )
            x_l = x_l.squeeze(dim=-3)
            x_l_augment = x_l.clone()  # Save for conditioning enforcement
            
            # Get noise level for current denoising step
            # Typically decreases from high (T) to low (0)
            t_hat = coord_noise_scheduler.get_noise_level(step_idx, chunk_n_sample)

            # Predictor-Corrector step: Add small corrector noise at current level
            # This improves sampling quality
            x_noisy = coord_noise_scheduler.add_noise_with_condition(
                x_l=x_l,
                condition_mask=condition_mask,
            )

            # Denoise using neural network
            # Network predicts clean coordinates from noisy input
            x_update = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_data=input_data,
                input_embedding=input_embedding,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            # Update coordinates using predictor-corrector formula
            # Conditioned atoms are kept fixed to ground truth
            x_l = coord_noise_scheduler.update_with_condition(
                x_noisy=x_noisy,
                x_update=x_update,
                x_gt=x_l_augment,
                condition_mask=condition_mask,
            )          

            # Reverse SE(3) augmentation to return to original frame
            x_l = reverse_centre_random_augmentation(x_l, trans, rot, x_center)

            # Verification: Ensure conditioned atoms match ground truth exactly
            check_condition_atom_coords(x_l, ground_truth.coordinate, condition_mask)
            
        return x_l


    # Process samples: either all at once or in chunks for memory efficiency
    if diffusion_chunk_size is None:
        # Process all samples together
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        # Process samples in chunks to reduce peak memory usage
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            # Calculate chunk size (last chunk may be smaller)
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            # Process this chunk through all denoising steps
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        # Concatenate all chunks along sample dimension
        x_l = torch.cat(x_l, -3)  # [N_sample, N_atom, 3]
    
    # Return denoised coordinates wrapped in DiffusionOutput
    return DiffusionOutput(x_l)


@register_license('odesign2025')
def sample_diffusion_training(
    noise_schedulers: dict[str, TrainingNoiseScheduler],
    denoise_net: torch.nn.Module,
    input_data: DiffusionInput,
    input_embedding: PairFormerOutput,
    ground_truth: GroundTruth,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
) -> DiffusionOutput:
    """
    Training-time diffusion sampling with noise addition and denoising.
    
    This function implements the training procedure for diffusion models as described
    in AlphaFold3 Appendix (page 23). It adds noise to ground truth coordinates and
    trains the denoising network to predict the clean structure, supporting conditional
    generation where certain atom positions can be fixed.
    
    Training Process:
        1. Apply random SE(3) augmentation to ground truth coordinates
        2. Sample random noise levels for each training sample
        3. Add Gaussian noise to coordinates at sampled noise levels
        4. Predict clean coordinates using denoising network
        5. Compute loss between predicted and ground truth coordinates
        6. Reverse augmentation to original frame
        7. Enforce conditioning constraints
    
    Conditional Training:
        - Atoms marked in input_data.is_condition_atom are NOT noised
        - These atoms remain at ground truth positions throughout training
        - Helps model learn to generate structures conditioned on partial information
        - Essential for tasks like ligand design, protein completion, etc.
    
    Conditioning Dropout:
        - use_conditioning=False randomly drops conditioning information
        - Forces model to learn unconditional generation
        - Improves robustness and generation quality
        - Typically applied with probability 0.1-0.2
    
    SE(3) Augmentation:
        - Random rotation and translation applied to coordinates
        - Makes model invariant to global reference frame
        - Improves generalization
    
    Memory Management:
        - Supports chunking when N_sample is large
        - Each chunk is processed independently through the network

    Args:
        noise_schedulers: Dictionary mapping scheduler names to TrainingNoiseScheduler
            objects. Must contain 'coordinate' scheduler. Training schedulers typically
            use log-normal or EDM-style noise distributions.
        denoise_net: Denoising neural network (DiffusionModule) that predicts clean
            coordinates from noisy coordinates. Must implement forward pass taking:
            - x_noisy: Noisy coordinates [N_sample, N_atom, 3]
            - t_hat_noise_level: Noise levels [N_sample]
            - input_data: DiffusionInput object
            - input_embedding: PairFormerOutput object
            - use_conditioning: Whether to use conditioning information
        input_data: DiffusionInput object containing:
            - atom_to_token_idx: Atom to token mapping [N_atom]
            - is_condition_atom: Conditioning mask [N_atom], True for atoms to keep fixed
            - Other atom-level features required by denoise_net
        input_embedding: PairFormerOutput object containing:
            - s_inputs: Initial single representations [N_token, c_s_inputs]
            - s: Refined single representations [N_token, c_s]
            - z: Pair representations [N_token, N_token, c_z]
        ground_truth: GroundTruth object containing:
            - coordinate: Ground truth clean coordinates [N_atom, 3]
            - coordinate_mask: Valid coordinate mask [N_atom]
                Only masked atoms contribute to loss
        N_sample: Number of noisy samples to generate per training example.
            Multiple samples per structure improve training stability and sample efficiency.
            Typical values: 1-4. Defaults to 1.
        diffusion_chunk_size: Chunk size for processing multiple samples. If None,
            all N_sample samples are processed together. Use chunking when N_sample
            is large to reduce memory. Defaults to None.
        use_conditioning: Whether to use conditioning information. When False,
            conditioning is dropped and model must generate unconditionally.
            This is used for conditioning dropout. Defaults to True.

    Returns:
        DiffusionOutput: Object containing:
            - x_denoised: Denoised (predicted clean) coordinates [N_sample, N_atom, 3]
                Used to compute loss against ground_truth.coordinate
                Conditioned atoms should match ground truth exactly
            - sigma: Noise levels used for each sample [N_sample]
                Used for noise-level-dependent loss weighting
                Typically higher noise gets lower weight
    
    Training Loss:
        The typical loss is computed as:
            loss = weighted_mse(x_denoised, ground_truth.coordinate)
        where weighting depends on sigma (noise level) and coordinate_mask
    
    Note:
        - Each sample gets an independent noise level from the scheduler
        - SE(3) augmentation is applied and then reversed
        - Conditioning atoms are preserved exactly (no noise added)
        - Conditioning check ensures fixed atoms match ground truth
        - This function is called during training, so gradients flow through denoise_net
        
    Example:
        For training with 2 samples per structure:
        >>> output = sample_diffusion_training(
        ...     noise_schedulers=train_schedulers,
        ...     denoise_net=diffusion_module,
        ...     input_data=diffusion_input,  # with is_condition_atom mask
        ...     input_embedding=pairformer_output,
        ...     ground_truth=ground_truth,  # clean coordinates
        ...     N_sample=2,
        ...     use_conditioning=True,  # Use conditioning (set False for dropout)
        ... )
        >>> print(output.x_denoised.shape)  # [2, N_atom, 3]
        >>> print(output.sigma.shape)  # [2]
        >>> # Compute loss
        >>> loss = mse_loss(output.x_denoised, ground_truth.coordinate)
    """
    device = ground_truth.coordinate.device
    dtype = ground_truth.coordinate.dtype

    # Step 1: Apply random SE(3) augmentation to ground truth coordinates
    # This makes the model SE(3) equivariant (invariant to global frame)
    # Shape: [N_sample, N_atom, 3]
    x_gt_augment, trans, rot, x_center = centre_random_augmentation(
        x_input_coords=ground_truth.coordinate,
        N_sample=N_sample,
        mask=ground_truth.coordinate_mask,
        dtype=dtype
    )  

    # Step 2: Sample random noise levels for each training sample
    # Each sample gets an independent noise level from the training scheduler
    # Typically uses log-normal or EDM distribution
    coord_noise_scheduler = noise_schedulers['coordinate']

    # Shape: [N_sample]
    sigma = coord_noise_scheduler.sample_noise_level(
        size=(N_sample,), device=device
    ).to(dtype)

    # Step 3: Create conditioning mask for atoms that should remain fixed
    # Conditioned atoms will not receive noise and remain at ground truth
    # Shape: [N_sample, N_atom]
    condition_mask = torch.logical_and(
        ground_truth.coordinate_mask, input_data.is_condition_atom
    ).unsqueeze(dim=0).expand(N_sample, -1)

    # Step 4: Add Gaussian noise to coordinates based on sampled noise levels
    # Conditioned atoms are NOT noised (remain at ground truth positions)
    # Shape: [N_sample, N_atom, 3]
    x_noisy = coord_noise_scheduler.add_noise_with_condition(
        x_gt=x_gt_augment,
        sigma=sigma,
        condition_mask=condition_mask,
    )

    # Step 5: Predict clean coordinates using denoising network
    # Network learns to remove noise and recover clean structure
    # Process all samples together or in chunks based on memory constraints
    if diffusion_chunk_size is None:
        # Process all N_sample samples together
        x_update = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=sigma,
            input_data=input_data,
            input_embedding=input_embedding,
            use_conditioning=use_conditioning,
        )
    else:
        # Process samples in chunks to reduce memory usage
        x_update = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            # Extract chunk of noisy samples
            x_noisy_i = x_noisy[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            # Extract corresponding noise levels
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            # Denoise this chunk
            x_update_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_data=input_data,
                input_embedding=input_embedding,
                use_conditioning=use_conditioning,
            )
            x_update.append(x_update_i)
        # Concatenate all chunks
        x_update = torch.cat(x_update, dim=-3)
    
    # Step 6: Compute final denoised coordinates
    # Conditioned atoms are kept exactly at ground truth positions
    # Shape: [N_sample, N_atom, 3]
    x_denoised = coord_noise_scheduler.denoise_with_conditon(
        x_noisy=x_noisy,
        x_update=x_update,
        x_gt=x_gt_augment,
        sigma=sigma,
        condition_mask=condition_mask
    )
    
    # Step 7: Reverse SE(3) augmentation to return to original frame
    x_denoised = reverse_centre_random_augmentation(x_denoised, trans, rot, x_center)
    
    # Step 8: Verification - ensure conditioned atoms match ground truth exactly
    check_condition_atom_coords(x_denoised, ground_truth.coordinate, condition_mask)
    
    # Return denoised coordinates and noise levels
    # Noise levels (sigma) are used for loss weighting in training
    return DiffusionOutput(x_denoised, sigma)

        

    
