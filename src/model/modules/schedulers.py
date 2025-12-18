"""
Noise schedulers for diffusion-based structure generation.

This module implements noise scheduling algorithms for both training and inference
in diffusion models, following the EDM (Elucidating the Design Space of Diffusion-Based
Generative Models) framework adapted for protein structure generation.

Main Classes:
    InferenceNoiseScheduler: Abstract base class for inference-time noise scheduling
    InferenceNoiseEDMScheduler: Concrete implementation using EDM-style scheduling
        - Manages noise levels during iterative denoising
        - Supports predictor-corrector sampling for improved quality
        - Handles conditional generation with fixed atom positions
    
    TrainingNoiseScheduler: Abstract base class for training-time noise scheduling
    TrainingNoiseEDMScheduler: Concrete implementation using EDM-style scheduling
        - Samples noise levels from log-normal distribution
        - Adds noise to ground truth coordinates for training
        - Supports conditional training with fixed atoms

Key Features:
    - EDM Framework: Implements EDM's noise scaling and denoising formulas
    - Predictor-Corrector: Optional corrector steps for better sampling quality
    - Conditional Generation: Preserves specified atom positions throughout
    - Noise Level Parameterization: Uses sigma (noise level) as time parameter
    - Scaling: Normalizes coordinates for stable training and generation

EDM Formulas:
    Training:
        - Noise scaling: x_noisy = c_in * (x_clean + sigma * noise)
        - c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        - Denoising: x_denoised = c_skip * x_noisy + c_out * x_update
        - c_skip = 1 / (1 + (sigma/sigma_data)^2)
        - c_out = sigma / sqrt(1 + (sigma/sigma_data)^2)
    
    Inference:
        - Similar formulas but with iterative refinement
        - Predictor-corrector adds small noise and denoises at each step
        - Step size controlled by eta parameter

Noise Schedule:
    - Training: Log-normal distribution p(log(sigma)) ~ N(p_mean, p_std)
    - Inference: Deterministic schedule from s_max to s_min with rho spacing
    - Typical values: sigma_data=16.0, s_max=160.0, s_min=0.0004
"""

import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple
import torch
from src.utils.model.torch_utils import append_dims

logger = logging.getLogger(__name__)


class InferenceNoiseScheduler(ABC):
    """
    Abstract base class for inference-time noise schedulers.
    
    This class defines the interface for noise schedulers used during inference/generation.
    Schedulers manage the noise levels during iterative denoising and handle conditional
    generation where certain atom positions are fixed.
    
    Subclasses must implement:
        - set_noise_schedule: Define the noise schedule (sigma values over time)
        - get_noise_level: Get noise level for a specific denoising step
        - sample_init_noise_with_condition: Initialize noise at start of sampling
        - add_noise_with_condition: Add corrector noise during predictor-corrector sampling
        - update_with_condition: Update coordinates after denoising step
        - denoise_with_conditon: Compute denoised coordinates from noisy input
    """
    
    def __init__(self):
        """Initialize the inference noise scheduler."""
        pass
    
    @abstractmethod
    def set_noise_schedule(self, *args, **kwargs) -> None:
        """
        Set up the noise schedule for inference sampling.
        
        This method should create a sequence of noise levels (sigma values) that
        decrease from high to low, defining the denoising trajectory.
        
        Returns:
            None, stores the schedule internally
        """
        pass
    
    @abstractmethod
    def get_noise_level(self, *args, **kwargs) -> torch.Tensor:
        """
        Get the noise level for a specific denoising step.
        
        Returns the sigma value (noise level) to use at the current step,
        potentially modified by predictor-corrector parameters.
        
        Returns:
            torch.Tensor: Noise level for the current step
        """
        pass
    
    @abstractmethod
    def sample_init_noise_with_condition(self, *args, **kwargs) -> torch.Tensor:
        """
        Sample initial noise for starting the denoising process.
        
        Should return random Gaussian noise for non-conditioned atoms and
        ground truth positions for conditioned atoms.
        
        Returns:
            torch.Tensor: Initial noisy coordinates
        """
        pass
    
    @abstractmethod
    def add_noise_with_condition(self, *args, **kwargs) -> torch.Tensor:
        """
        Add corrector noise to current coordinates.
        
        Used in predictor-corrector sampling to add small noise before
        the denoising step, improving sampling quality.
        
        Returns:
            torch.Tensor: Coordinates with added noise
        """
        pass
    
    @abstractmethod
    def update_with_condition(self, *args, **kwargs) -> torch.Tensor:
        """
        Update coordinates after a denoising step.
        
        Combines the denoising prediction with the current noisy coordinates
        to move towards the final structure, while preserving conditioned atoms.
        
        Returns:
            torch.Tensor: Updated coordinates for next step
        """
        pass
    
    @abstractmethod
    def denoise_with_conditon(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute clean coordinates from noisy input and network prediction.
        
        Applies EDM-style scaling formulas to combine noisy input with
        network prediction, producing denoised coordinates.
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                Either denoised coordinates or tuple of (denoised, unscaled_noisy)
        """
        pass
    

class InferenceNoiseEDMScheduler(InferenceNoiseScheduler):
    """
    EDM-style noise scheduler for inference-time structure generation.
    
    This scheduler implements the EDM (Elucidating the Design Space of Diffusion-Based
    Generative Models) framework adapted for protein structure generation. It manages
    noise levels during iterative denoising and supports predictor-corrector sampling
    for improved generation quality.
    
    Key Features:
        - Deterministic noise schedule from s_max to s_min
        - Predictor-corrector sampling (optional) for better quality
        - EDM-style scaling for numerical stability
        - Conditional generation support (fixed atom positions)
    
    Noise Schedule:
        The noise levels (sigma) follow a power-law schedule:
            sigma(t) = sigma_data * (s_max^(1/rho) + t * (s_min^(1/rho) - s_max^(1/rho)))^rho
        where t goes from 0 (high noise) to 1 (low noise)
    
    Predictor-Corrector:
        At each step, optionally adds corrector noise:
            t_hat = c_tau * (1 + gamma)
        where gamma controls the amount of correction
    
    Scaling:
        Coordinates are scaled by c_in before denoising:
            c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        This normalizes the input to have approximately unit variance
    
    Attributes:
        sigma_data: Data scaling parameter (typically 16.0 for structures)
        s_max: Maximum noise level at start of sampling
        s_min: Minimum noise level at end of sampling
        rho: Controls spacing of noise levels in schedule
        use_predictor_corrector_sampler: Whether to use predictor-corrector
        gamma0: Corrector strength parameter
        gamma_min: Minimum noise level for correction
        noise_scale_lambda: Corrector noise scaling factor
        step_scale_eta: Step size scaling factor
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
        use_predictor_corrector_sampler: bool = True,
        gamma0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale_lambda: float = 1.003,
        step_scale_eta: float = 1.5,
    ) -> None:
        """
        Initialize the EDM inference noise scheduler.

        Args:
            s_max: Maximum noise level at the start of sampling. Higher values
                provide more diversity but may reduce quality. Typical: 160.0.
            s_min: Minimum noise level at the end of sampling. Very small value
                to ensure clean final structure. Typical: 4e-4.
            rho: Power-law exponent controlling noise schedule spacing. Higher
                values concentrate more steps at low noise levels. Typical: 7.
            sigma_data: Data scale parameter for normalization. Should match the
                scale of coordinate data (typically 16.0 for Angstroms). In original
                EDM paper this is 1.0 for images.
            use_predictor_corrector_sampler: Whether to use predictor-corrector
                sampling. True improves quality at small computational cost.
            gamma0: Corrector strength parameter. Controls how much noise is added
                in corrector step. Typical: 0.8.
            gamma_min: Minimum noise level below which correction is disabled.
                Prevents unnecessary correction at low noise. Typical: 1.0.
            noise_scale_lambda: Scaling factor for corrector noise. Fine-tunes
                the amount of correction. Typical: 1.003.
            step_scale_eta: Step size scaling factor. Controls denoising step size.
                Typical: 1.5.

        Returns:
            None
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        self.use_predictor_corrector_sampler = use_predictor_corrector_sampler
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta

    def set_noise_schedule(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Set up the noise schedule for inference sampling.
        
        Creates a deterministic sequence of noise levels (sigma values) that decrease
        from s_max to s_min following a power-law schedule. More steps generally improve
        quality but increase computation time.
        
        The schedule follows:
            sigma(i) = sigma_data * (s_max^(1/rho) + (i/N_step) * 
                       (s_min^(1/rho) - s_max^(1/rho)))^rho
        
        where i ranges from 0 to N_step, with sigma(N_step) = 0 for the final clean state.

        Args:
            N_step: Number of denoising steps. More steps generally improve quality
                but increase computation. Typical values: 50-200. Defaults to 200.
            device: Target device for the schedule tensor. Should match the device
                of the model and data. Defaults to CPU.
            dtype: Target dtype for the schedule tensor. Should match the model's
                dtype (typically float32). Defaults to torch.float32.

        Returns:
            None, but stores the noise schedule internally in self.noise_schedule
            Shape: [N_step + 1] containing sigma values from high to low
            
        Note:
            - The final noise level is set to exactly 0 for clean generation
            - This method must be called before get_noise_level
            - The schedule is stored internally and reused for all steps
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        self.noise_schedule = t_step_list

    def get_noise_level(
        self,
        step_idx: int,
        N_sample: int,
    ) -> torch.Tensor:
        """
        Get the noise level for a specific denoising step.
        
        Retrieves the sigma value (noise level) for the current denoising step.
        If predictor-corrector sampling is enabled, increases the noise level
        slightly to allow for a correction step.
        
        Predictor-Corrector Logic:
            - Without correction: t_hat = sigma(step_idx)
            - With correction: t_hat = sigma(step_idx) * (1 + gamma)
            - gamma is 0 when sigma < gamma_min (disable correction at low noise)

        Args:
            step_idx: Current step index in the denoising process. Should be
                in range [0, N_step-1]. Step 0 has highest noise.
            N_sample: Number of samples being processed. Used to expand the
                noise level tensor to match batch size.

        Returns:
            torch.Tensor: Noise level for this step, shape [N_sample]
                Broadcast to match the number of samples being processed
                
        Side Effects:
            Stores c_tau_last, c_tau, and t_hat internally for use by other methods
            
        Note:
            - Must call set_noise_schedule before using this method
            - The returned noise level is used by the denoising network
            - Predictor-corrector adds small noise then denoises for better quality
        """
        # Get noise levels for current and next step
        self.c_tau_last = self.noise_schedule[step_idx]  # Current step sigma
        self.c_tau = self.noise_schedule[step_idx + 1]    # Next step sigma

        if not self.use_predictor_corrector_sampler:
            # Simple predictor: use current noise level
            self.t_hat = self.c_tau_last
        else:
            # Predictor-corrector: increase noise level for correction step
            # gamma controls correction strength, disabled at low noise
            gamma = float(self.gamma0) if self.c_tau > self.gamma_min else 0
            self.t_hat = self.c_tau_last * (gamma + 1)
        
        # Expand to match number of samples
        return self.t_hat.expand(N_sample)

    def sample_init_noise_with_condition(
        self,
        size: tuple,
        x_gt: torch.Tensor,
        condition_mask: torch.Tensor, 
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,        
    ) -> torch.Tensor:
        """
        Sample initial noise to start the denoising process.
        
        Creates initial noisy coordinates by sampling from a Gaussian distribution
        scaled by the initial noise level (s_max). For conditioned atoms, uses
        ground truth positions instead of noise.
        
        This is the starting point (t=0, highest noise) for the iterative denoising
        process that will gradually reduce noise to generate the final structure.

        Args:
            size: Shape of the coordinates to generate, typically (N_sample, N_atom, 3)
            x_gt: Ground truth coordinates for conditioned atoms.
                Shape: [N_sample, N_atom, 3] or broadcastable
            condition_mask: Boolean mask indicating which atoms to condition on.
                Shape: [N_sample, N_atom], True for atoms to keep fixed
            device: Device for tensor creation. Should match model device.
            dtype: Data type for tensor creation. Should match model dtype.

        Returns:
            torch.Tensor: Initial noisy coordinates, shape specified by size parameter
                Non-conditioned atoms: Gaussian noise scaled by s_max
                Conditioned atoms: Exact ground truth positions
                
        Note:
            - Conditioned atoms are set to ground truth and never change
            - Non-conditioned atoms start as pure noise and are iteratively denoised
            - Must call set_noise_schedule before using this method
        """
        # Sample Gaussian noise scaled by initial noise level
        x_l = (
            self.noise_schedule[0] *  # s_max (initial noise level)
            torch.randn(
                size=size, device=device, dtype=dtype
            )
        )

        # Replace conditioned atoms with ground truth positions
        x_l = torch.where(
            append_dims(condition_mask, x_gt.ndim),
            x_gt,
            x_l
        )

        return x_l
    
    def add_noise_with_condition(
        self,
        x_l: torch.Tensor,
        condition_mask: torch.Tensor,     
        scale: bool = True
    ) -> torch.Tensor:
        """
        Add corrector noise to current coordinates (predictor-corrector sampling).
        
        In predictor-corrector sampling, this method adds a small amount of noise
        before the denoising step. This corrector step improves sampling quality
        by allowing the network to make better predictions.
        
        Corrector Noise:
            delta_sigma = sqrt(t_hat^2 - c_tau_last^2)
            x_noisy = x_l + lambda * delta_sigma * noise
        
        where t_hat = c_tau_last * (1 + gamma) from get_noise_level
        
        Scaling (if enabled):
            c_in = 1 / sqrt(sigma_data^2 + t_hat^2)
            x_noisy = c_in * x_noisy
        
        This normalization ensures the network input has approximately unit variance.

        Args:
            x_l: Current coordinates at this denoising step.
                Shape: [N_sample, N_atom, 3]
            condition_mask: Boolean mask for conditioned atoms.
                Shape: [N_sample, N_atom], True for atoms to keep fixed
            scale: Whether to apply EDM scaling. Should be True for proper
                functioning with the denoising network. Defaults to True.

        Returns:
            torch.Tensor: Coordinates with added corrector noise and optional scaling
                Shape: [N_sample, N_atom, 3]
                Conditioned atoms remain unchanged (zero noise)
                Non-conditioned atoms have added Gaussian noise
                
        Note:
            - If use_predictor_corrector_sampler=False, returns x_l unchanged
            - Conditioned atoms never receive noise (zeros used for them)
            - The added noise amount depends on gamma parameter
            - Scaling is crucial for network stability
        """
        if self.use_predictor_corrector_sampler:
            # Calculate the amount of corrector noise to add
            # Based on the difference between t_hat and c_tau_last
            delta_noise_level = torch.sqrt(self.t_hat**2 - self.c_tau_last**2)

            # Sample noise: zero for conditioned atoms, Gaussian for others
            noise = torch.where(
                append_dims(condition_mask, x_l.ndim), 
                torch.zeros_like(x_l),  # No noise for conditioned atoms
                torch.randn_like(x_l)   # Gaussian noise for others
            )

            # Add scaled corrector noise
            x_noisy = x_l + self.noise_scale_lambda * delta_noise_level * noise
        else:
            # No correction: just use current coordinates
            x_noisy = x_l

        if not scale:
            return x_noisy
        else:
            # Scale positions to dimensionless vectors with approximately unit variance
            # As in EDM:
            #     r_noisy = (c_in * x_noisy)
            #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
            c_in = 1 / torch.sqrt(self.sigma_data**2 + self.t_hat**2)
            x_noisy = c_in * x_noisy
            return x_noisy

    def update_with_condition(
        self, 
        x_noisy: torch.Tensor,
        x_update: torch.Tensor,
        x_gt: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update coordinates after a denoising step using predictor formula.
        
        This method implements the predictor step in predictor-corrector sampling.
        It takes the denoised prediction and updates the current coordinates to
        move towards the final clean structure.
        
        Update Formula (Euler method):
            1. Denoise: x_clean = denoise(x_noisy, x_update)
            2. Compute gradient: delta = (x_noisy - x_clean) / sigma
            3. Take step: x_new = x_noisy + eta * dt * delta
        
        where:
            - dt = c_tau - t_hat (step from current to next noise level)
            - eta is the step size scaling factor
            - delta approximates the score function gradient

        Args:
            x_noisy: Noisy coordinates after adding corrector noise.
                Shape: [N_sample, N_atom, 3]
                This should be the output of add_noise_with_condition
            x_update: Network prediction for clean coordinates.
                Shape: [N_sample, N_atom, 3]
                Output from the denoising network
            x_gt: Ground truth coordinates for conditioned atoms.
                Shape: [N_sample, N_atom, 3]
                Used to enforce conditioning constraints
            condition_mask: Boolean mask for conditioned atoms.
                Shape: [N_sample, N_atom]
                True for atoms that should stay fixed

        Returns:
            torch.Tensor: Updated coordinates for next step
                Shape: [N_sample, N_atom, 3]
                Non-conditioned atoms: Updated using predictor formula
                Conditioned atoms: Exact ground truth positions
                
        Note:
            - This implements one Euler step in the probability flow ODE
            - Conditioned atoms are always set to ground truth
            - Step size eta controls how far to move (typically 1.5)
            - Called after the denoising network makes a prediction
        """
        # Get noise level dimension-matched to coordinates
        sigma = append_dims(self.t_hat, x_noisy.ndim)
        
        # Denoise to get clean prediction
        # Also returns unscaled x_noisy for gradient calculation
        x_denoised, x_noisy = self.denoise_with_conditon(
            x_noisy=x_noisy,
            x_update=x_update,
            return_x_noisy=True,
        )

        # Compute score function approximation (gradient of log probability)
        # delta approximates: d(log p)/dx
        delta = (x_noisy - x_denoised) / sigma
        
        # Compute time step
        dt = self.c_tau - self.t_hat
        
        # Take Euler step: x_new = x + eta * dt * gradient
        x_l = x_noisy + self.step_scale_eta * dt * delta

        # Enforce conditioning: replace conditioned atoms with ground truth
        x_l = torch.where(
            append_dims(condition_mask, x_l.ndim), 
            x_gt,  # Use ground truth for conditioned atoms
            x_l    # Use updated prediction for others
        )

        return x_l
        
    def denoise_with_conditon(
        self, 
        x_noisy: torch.Tensor,
        x_update: torch.Tensor,
        x_gt: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        return_x_noisy: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute denoised coordinates from noisy input and network prediction.
        
        This method implements the EDM denoising formula that combines the noisy
        input with the network's prediction to produce the denoised output. The
        formula uses skip connections and scaling factors for numerical stability.
        
        EDM Denoising Formula:
            x_denoised = c_skip * x_noisy + c_out * x_update
        
        where:
            s_ratio = sigma / sigma_data
            c_skip = 1 / (1 + s_ratio^2) = sigma_data^2 / (sigma_data^2 + sigma^2)
            c_out = sigma / sqrt(1 + s_ratio^2) = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        
        The c_skip term preserves information from the noisy input (like a residual
        connection), while c_out scales the network prediction. The balance between
        them depends on the noise level sigma.
        
        Key Properties:
            - At high noise: c_skip ≈ 0, c_out ≈ sigma (trust network more)
            - At low noise: c_skip ≈ 1, c_out ≈ 0 (trust input more)
            - Smooth interpolation between these extremes

        Args:
            x_noisy: Noisy coordinates (scaled by c_in from add_noise_with_condition).
                Shape: [N_sample, N_atom, 3]
            x_update: Network prediction for the update.
                Shape: [N_sample, N_atom, 3]
                This is the raw output from the denoising network
            x_gt: Ground truth coordinates for conditioned atoms (optional).
                Shape: [N_sample, N_atom, 3]
                Used to enforce conditioning if provided
            condition_mask: Boolean mask for conditioned atoms (optional).
                Shape: [N_sample, N_atom]
                Must be provided together with x_gt
            return_x_noisy: Whether to also return the unscaled noisy coordinates.
                Useful for computing gradients in update_with_condition.

        Returns:
            torch.Tensor or tuple:
                If return_x_noisy=False:
                    x_denoised: Denoised coordinates [N_sample, N_atom, 3]
                If return_x_noisy=True:
                    (x_denoised, x_noisy): Tuple of denoised and unscaled noisy coords
                    
        Note:
            - The input x_noisy must have been scaled by add_noise_with_condition
            - This method unscales x_noisy before applying the denoising formula
            - Conditioned atoms (if specified) are set to exact ground truth
            - The scaling factors ensure numerical stability across noise levels
        """
        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * x_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        # Get noise level with matching dimensions
        sigma = append_dims(self.t_hat, x_update.ndim)
        s_ratio = sigma / self.sigma_data

        # Unscale x_noisy (reverse the c_in scaling from add_noise_with_condition)
        # This must be done because x_noisy was scaled by c_in = 1/sqrt(sigma_data^2 + sigma^2)
        x_noisy = x_noisy * torch.sqrt(self.sigma_data**2 + sigma**2)
        
        # Apply EDM denoising formula
        # Combines scaled noisy input (skip connection) with scaled network prediction
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy          # c_skip * x_noisy
            + sigma / torch.sqrt(1 + s_ratio**2) * x_update  # c_out * x_update
        )

        # Enforce conditioning if specified
        if x_gt is not None and condition_mask is not None:
            x_denoised = torch.where(
                append_dims(condition_mask, x_denoised.ndim), 
                x_gt,         # Use ground truth for conditioned atoms
                x_denoised    # Use denoised prediction for others
            )

        # Return denoised coordinates, optionally with unscaled noisy coordinates
        if return_x_noisy:
            return x_denoised, x_noisy
        else:
            return x_denoised
    
        
class TrainingNoiseScheduler(ABC):
    """
    Abstract base class for training-time noise schedulers.
    
    This class defines the interface for noise schedulers used during training.
    Training schedulers randomly sample noise levels and add noise to ground truth
    coordinates to create training samples for the denoising network.
    
    Subclasses must implement:
        - sample_noise_level: Sample random noise levels for training
        - add_noise_with_condition: Add noise to ground truth coordinates
        - denoise_with_conditon: Compute clean coordinates from noisy input
    
    Training Process:
        1. Sample random noise level sigma from distribution
        2. Add Gaussian noise: x_noisy = x_clean + sigma * noise
        3. Network predicts clean coordinates from x_noisy
        4. Compute loss between prediction and x_clean
    """
    
    def __init__(self):
        """Initialize the training noise scheduler."""
        pass
    
    @abstractmethod
    def sample_noise_level(self, *args, **kwargs) -> torch.Tensor:
        """
        Sample random noise levels for training samples.
        
        Should return sigma values sampled from a distribution (typically
        log-normal) that covers the range of noise levels used during inference.
        
        Returns:
            torch.Tensor: Sampled noise levels (sigma values)
        """
        pass
    
    @abstractmethod
    def add_noise_with_condition(self, *args, **kwargs) -> torch.Tensor:
        """
        Add noise to ground truth coordinates for training.
        
        Should add Gaussian noise scaled by the sampled sigma values, while
        preserving conditioned atoms at their ground truth positions.
        
        Returns:
            torch.Tensor: Noisy coordinates
        """
        pass

    @abstractmethod
    def denoise_with_conditon(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute clean coordinates from noisy input and network prediction.
        
        Should apply EDM-style scaling formulas to combine noisy input with
        network prediction, producing denoised coordinates for loss computation.
        
        Returns:
            torch.Tensor: Denoised coordinates
        """
        pass


class TrainingNoiseEDMScheduler(TrainingNoiseScheduler):
    """
    EDM-style noise scheduler for training diffusion models.
    
    This scheduler implements the EDM (Elucidating the Design Space of Diffusion-Based
    Generative Models) training approach adapted for protein structure generation. It
    samples noise levels from a log-normal distribution and adds scaled noise to ground
    truth coordinates for training the denoising network.
    
    Key Features:
        - Log-normal noise level distribution for diverse training samples
        - EDM-style scaling for numerical stability
        - Conditional training support (fixed atom positions)
        - Matches the parameterization used in InferenceNoiseEDMScheduler
    
    Noise Level Distribution:
        sigma ~ exp(N(p_mean, p_std)) * sigma_data
        
        where:
            - log(sigma) follows a Gaussian distribution
            - p_mean controls the center of the distribution
            - p_std controls the spread
            - sigma_data scales the entire distribution
    
    Why Log-Normal:
        - Noise levels span multiple orders of magnitude (0.0004 to 160)
        - Log-normal naturally covers this wide range
        - Provides more training samples at intermediate noise levels
        - Matches the inference schedule's distribution
    
    Training Process:
        1. Sample sigma ~ log-normal distribution
        2. Add noise: x_noisy = x_clean + sigma * noise
        3. Scale: x_scaled = c_in * x_noisy
        4. Network predicts x_update from x_scaled
        5. Denoise: x_denoised = c_skip * x_scaled + c_out * x_update
        6. Compute loss: L = ||x_denoised - x_clean||^2
    
    Attributes:
        sigma_data: Data scaling parameter (typically 16.0 for structures)
        p_mean: Mean of log(sigma) distribution
        p_std: Standard deviation of log(sigma) distribution
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """
        Initialize the EDM training noise scheduler.

        Args:
            p_mean: Mean of the log-normal distribution for noise levels.
                Controls the average noise level during training. More negative
                values favor lower noise levels. Typical: -1.2.
            p_std: Standard deviation of the log-normal distribution.
                Controls the spread of noise levels. Higher values give more
                diversity in training samples. Typical: 1.5.
            sigma_data: Data scale parameter for normalization. Should match
                the scale of coordinate data and the value used in inference.
                Typical: 16.0 for Angstroms. Original EDM uses 1.0 for images.

        Returns:
            None
            
        Note:
            - These parameters should be tuned to match the inference schedule
            - p_mean and p_std control which noise levels are seen during training
            - sigma_data must match between training and inference schedulers
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        logger.info(f"EDM scheduler sigma_data: {self.sigma_data}")

    def sample_noise_level(self, size: torch.Size, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Sample random noise levels from log-normal distribution for training.
        
        Samples sigma values that determine how much noise to add to ground truth
        coordinates. Each training sample gets an independent noise level, providing
        diversity in training and ensuring the model learns to denoise at all levels.
        
        Sampling Formula:
            z ~ N(0, 1)
            log(sigma) = p_mean + p_std * z
            sigma = exp(log(sigma)) * sigma_data
        
        This log-normal distribution naturally spans multiple orders of magnitude,
        providing good coverage of the noise levels used during inference.

        Args:
            size: Shape of noise level tensor to sample, typically (N_sample,)
                where N_sample is the number of training samples per batch
            device: Device for tensor creation (should match model device)

        Returns:
            torch.Tensor: Sampled noise levels (sigma values)
                Shape: specified by size parameter, typically [N_sample]
                Values typically range from ~0.001 to ~100 depending on parameters
                Each value is an independent sample from the log-normal distribution
                
        Note:
            - Each training sample should get a different noise level
            - The distribution parameters (p_mean, p_std) control the range
            - Values match the scale used during inference (s_min to s_max)
            - Log-normal ensures good coverage across orders of magnitude
        """
        # Sample from standard normal
        rnd_normal = torch.randn(size=size, device=device)
        
        # Transform to log-normal: sigma = exp(N(p_mean, p_std)) * sigma_data
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        
        return noise_level
    
    def add_noise_with_condition(
        self, 
        x_gt: torch.Tensor, 
        sigma: torch.Tensor, 
        condition_mask: torch.Tensor, 
        scale: bool = True
    ) -> torch.Tensor:
        """
        Add noise to ground truth coordinates for training samples.
        
        Creates noisy training samples by adding Gaussian noise scaled by sigma.
        Conditioned atoms are not noised and remain at their ground truth positions,
        allowing the model to learn conditional generation.
        
        Noise Addition Formula:
            noise ~ N(0, I)  for non-conditioned atoms
            noise = 0        for conditioned atoms
            x_noisy = x_gt + sigma * noise
        
        Scaling (if enabled):
            c_in = 1 / sqrt(sigma_data^2 + sigma^2)
            x_noisy_scaled = c_in * x_noisy
        
        The scaling normalizes inputs to have approximately unit variance across
        different noise levels, improving training stability.

        Args:
            x_gt: Ground truth clean coordinates.
                Shape: [N_sample, N_atom, 3]
                These are the target coordinates the model learns to recover
            sigma: Noise levels sampled from log-normal distribution.
                Shape: [N_sample] or broadcastable
                Different samples can have different noise levels
            condition_mask: Boolean mask for atoms to condition on.
                Shape: [N_sample, N_atom]
                True for atoms that should stay fixed (no noise added)
            scale: Whether to apply EDM scaling normalization.
                Should be True for proper training. Defaults to True.

        Returns:
            torch.Tensor: Noisy coordinates for training
                Shape: [N_sample, N_atom, 3]
                Non-conditioned atoms: x_gt + sigma * noise (possibly scaled)
                Conditioned atoms: Exact ground truth positions
                
        Note:
            - Conditioned atoms never receive noise (zeros used)
            - Each sample can have different noise level (from sigma)
            - Scaling by c_in is crucial for training stability
            - The noise is element-wise Gaussian (independent per coordinate)
        """
        # Sample Gaussian noise: zero for conditioned atoms, random for others
        noise = torch.where(
            append_dims(condition_mask, x_gt.ndim), 
            torch.zeros_like(x_gt),  # No noise for conditioned atoms
            torch.randn_like(x_gt)   # Gaussian noise for others
        ) 
        
        # Expand sigma to match coordinate dimensions
        sigma = append_dims(sigma, noise.ndim)

        # Add scaled noise to ground truth
        noise_hat = sigma * noise
        x_noisy = x_gt + noise_hat
        
        if not scale:
            # Return unscaled noisy coordinates
            return x_noisy
        else:
            # Scale positions to dimensionless vectors with approximately unit variance
            # As in EDM:
            #     r_noisy = (c_in * x_noisy)
            #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
            c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
            x_noisy = c_in * x_noisy
            return x_noisy
        
    def denoise_with_conditon(
        self, 
        x_noisy: torch.Tensor,
        x_update: torch.Tensor,
        x_gt: torch.Tensor,
        sigma: torch.Tensor,
        condition_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute denoised coordinates from noisy input and network prediction for training.
        
        This method implements the EDM denoising formula used during training to combine
        the scaled noisy input with the network's prediction. The result is compared
        against ground truth to compute the training loss.
        
        EDM Denoising Formula:
            x_denoised = c_skip * x_noisy + c_out * x_update
        
        where:
            s_ratio = sigma / sigma_data
            c_skip = 1 / (1 + s_ratio^2) = sigma_data^2 / (sigma_data^2 + sigma^2)
            c_out = sigma / sqrt(1 + s_ratio^2) = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        
        Note: x_noisy here is the scaled version (multiplied by c_in), so we need to
        adjust c_skip to account for this:
            c_skip_adjusted = c_skip / c_in = sigma_data / sqrt(1 + s_ratio^2)
        
        Key Properties:
            - At high noise (large sigma): Relies more on network prediction (x_update)
            - At low noise (small sigma): Relies more on noisy input (x_noisy)
            - Skip connection stabilizes training and speeds convergence

        Args:
            x_noisy: Noisy coordinates scaled by c_in from add_noise_with_condition.
                Shape: [N_sample, N_atom, 3]
                This is the input to the denoising network
            x_update: Network's prediction for the denoising update.
                Shape: [N_sample, N_atom, 3]
                Raw output from the denoising network
            x_gt: Ground truth clean coordinates.
                Shape: [N_sample, N_atom, 3]
                Used to enforce conditioning and compute loss
            sigma: Noise levels for each sample.
                Shape: [N_sample] or broadcastable
                The sigma values used to add noise
            condition_mask: Boolean mask for conditioned atoms.
                Shape: [N_sample, N_atom]
                True for atoms that should stay at ground truth

        Returns:
            torch.Tensor: Denoised coordinates for loss computation
                Shape: [N_sample, N_atom, 3]
                Non-conditioned atoms: Combination of x_noisy and x_update via EDM formula
                Conditioned atoms: Exact ground truth positions
                
        Training Loss:
            The typical loss is:
                loss = MSE(x_denoised, x_gt) weighted by 1/sigma^2
                
        Note:
            - This method assumes x_noisy was scaled by c_in in add_noise_with_condition
            - The adjusted c_skip accounts for the input scaling
            - Conditioned atoms are always set to ground truth
            - The output is compared to x_gt to compute the training loss
        """
        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * x_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        # Expand sigma to match coordinate dimensions
        sigma = append_dims(sigma, x_update.ndim)
        s_ratio = sigma / self.sigma_data
        
        # Apply EDM denoising formula
        # Note: c_skip is adjusted because x_noisy was scaled by c_in
        x_denoised = (
            self.sigma_data / torch.sqrt(1 + s_ratio**2) * x_noisy  # Adjusted c_skip * x_noisy
            + sigma / torch.sqrt(1 + s_ratio**2) * x_update         # c_out * x_update
        )
        
        # Enforce conditioning: set conditioned atoms to ground truth
        x_denoised = torch.where(
            append_dims(condition_mask, x_denoised.ndim), 
            x_gt,         # Use ground truth for conditioned atoms
            x_denoised    # Use denoised prediction for others
        )
        
        return x_denoised
        