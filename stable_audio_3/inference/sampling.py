import torch
import typing as tp
from tqdm import trange, tqdm
import torch.distributions as dist
import numpy as np

from ..data.utils import create_padding_mask_from_lengths, compute_effective_seq_len_from_conditioning
from .audio_utils import prepare_audio


def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift = None,
    effective_seq_len: tp.Union[int, torch.Tensor, None] = None,
    fallback_seq_len: tp.Optional[int] = None,
    include_endpoint: bool = True,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling.

    Returns a 1D tensor of shape (N,) where N = steps+1 (if include_endpoint)
    or steps (if not), OR a 2D tensor of shape (batch_size, N) when
    effective_seq_len is a tensor and dist_shift produces per-element schedules.

    Args:
        steps: Number of sampling steps.
        sigma_max: Starting noise level (1.0 for full generation, <1.0 for variations).
        dist_shift: Optional distribution shift object (FluxDistributionShift,
            DistributionShift, LogSNRShift, etc.). Applied to warp the linear schedule.
        effective_seq_len: Sequence length for dist_shift. Scalar int or
            tensor of shape (batch_size,) for per-element schedules.
        fallback_seq_len: Fallback when effective_seq_len is None (typically x.shape[-1]).
        include_endpoint: If True, schedule includes 0 as final value (RF samplers).
            If False, excludes 0 (v-diffusion DDIM).
        device: Device for the output tensor.
    """
    n_points = steps + 1 if include_endpoint else steps

    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            # Clamp per-element sequence lengths to avoid zeros causing log/NaN issues
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            # Clamp scalar sequence length to at least 1
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        # Ensure the first timestep remains aligned with sigma_max after shifting.
        # This keeps the schedule consistent with the initialization in sample_diffusion(),
        # which mixes init_data using sigma_max.
        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                # For batched/per-element schedules, enforce sigma_max at the first time index.
                t[..., 0] = sigma_max_tensor

    return t


def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        mean_logsnr (float): Mean of the logSNR Gaussian distribution
        std_logsnr (float): Standard deviation of the logSNR Gaussian distribution

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from Gaussian distribution
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def sample_timesteps_logsnr_uniform(batch_size, min_logsnr=-6, max_logsnr=5.0):
    """
    Sample timesteps for diffusion training by sampling logSNR values and converting to t.

    Args:
        batch_size (int): Number of timesteps to sample
        min_logsnr (float): Minimum logSNR value
        max_logsnr (float): Maximum logSNR value

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing timestep values t in [0, 1]
    """
    # Sample logSNR from uniform distribution
    logsnr = torch.rand(batch_size) * (max_logsnr - min_logsnr) + min_logsnr

    # Convert logSNR to timesteps using the logistic function
    # Since logSNR = ln((1-t)/t), we can solve for t:
    # t = 1 / (1 + exp(logsnr))
    t = torch.sigmoid(-logsnr)

    # Clamp values to ensure numerical stability
    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):
    """

    shape: shape of the output tensor
    left_trunc: left truncation point, fraction of probability to be discarded
    right_trunc: right truncation boundary, should be 1 (never seen at test time)
    """

    # Step 1: Sample from the logistic normal distribution (sigmoid of normal)
    logits = torch.randn(shape)

    # Step 2: Apply the CDF transformation of the normal distribution
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)

    # Step 3: Define the truncation bounds on the CDF
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    # Step 4: Rescale linear CDF values into the truncated region (between lower_bound and upper_bound)
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values

    # Step 5: Map back to logistic-normal space using inverse CDF
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))

    # Step 6: Rescale values so that min is 0 and max is just below 1
    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

def sample_discrete_euler(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr_tensor = t[:, i].to(x.dtype)  # (batch_size,)
            t_prev = t[:, i + 1].to(x.dtype)  # (batch_size,)
            dt = t_prev - t_curr_tensor  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            dt_broadcast = dt.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_prev = t[i + 1]
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            dt = t_prev - t_curr
            dt_broadcast = dt

        v = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr_tensor[:, None, None] * v
            callback({'x': x, 't': t_curr_tensor, 'sigma': t_curr_tensor, 'i': i, 'denoised': denoised})

        x = x + dt_broadcast * v

    # If we are on the last timestep, output the denoised data
    return x

def sample_rk4(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta

    Args:
        sigmas: Pre-computed schedule tensor of shape (steps+1,).
            Per-element schedules not supported for RK4.
    """
    # Make tensor of ones to broadcast the single t values
    ts = x.new_ones([x.shape[0]])

    t = sigmas

    t = t.to(x.device)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        # Broadcast the current timestep to the correct shape
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # we solve backwards in our formulation

        k1 = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr * k1
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i, 'denoised': denoised})

        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)

        # Clamp t_prev to avoid evaluating model at exactly t=0
        # (models aren't trained at t=0 and may return garbage/NaN)
        t_prev_eval = t_prev.clamp(min=1e-5)
        k4 = model(x + dt * k3, t_prev_eval * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # If we are on the last timestep, output the denoised data
    return x

def sample_flow_dpmpp(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    old_denoised = None

    # Clamp t to avoid numerical issues with log(0) and division by zero
    # This prevents inf/-inf values that can cause NaN propagation
    log_snr = lambda t: ((1-t).clamp(min=1e-10) / t.clamp(min=1e-10)).log()

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i]  # (batch_size,)
            t_next = t[:, i + 1]  # (batch_size,)
            t_prev = t[:, i - 1] if i > 0 else None
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i]
            t_next = t[i + 1]
            t_prev = t[i - 1] if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr.expand(x.shape[0])

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'denoised': denoised})

        alpha_t = 1 - t_next_broadcast

        # For rectified flow, compute the DPM++ coefficient directly without log_snr
        # to avoid numerical issues at t=0 or t=1
        # The formula is: (-h).expm1() = (t_next - t_curr) / [(1 - t_next) * t_curr]
        # Note: t_next < t_curr, so this is negative
        # We'll compute this directly instead of going through log_snr
        dt = t_next_broadcast - t_curr_broadcast
        # Clamp to avoid division by zero when t_curr or t_next are at boundaries
        dpmpp_coeff = dt / ((1 - t_next_broadcast).clamp(min=1e-10) * t_curr_broadcast.clamp(min=1e-10))

        # Check if this is the first step or the last step (t_next == 0)
        is_first_step = old_denoised is None
        is_last_step = (t_next_broadcast == 0).all() if per_element_schedule else (t_next == 0)

        if is_first_step or is_last_step:
            # First-order update using the directly computed coefficient
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised
        else:
            # Second-order update with Richardson extrapolation
            if per_element_schedule:
                t_prev_broadcast = t_prev.view(-1, 1, 1)
            else:
                t_prev_broadcast = t_prev
            # Compute r = h_last / h in log-SNR space for second-order correction
            # h = log_snr(t_next) - log_snr(t_curr), h_last = log_snr(t_curr) - log_snr(t_prev)
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised
    return x

def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for distilled models

    Args:
        sigmas: Pre-computed schedule tensor. Shape (steps+1,) for global schedule
            or (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i].to(x.dtype)  # (batch_size,)
            t_next = t[:, i + 1].to(x.dtype)  # (batch_size,)
            # Reshape for broadcasting with x: (batch_size,) -> (batch_size, 1, 1)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
        else:
            # Global schedule: t has shape (steps+1,)
            t_curr = t[i].to(x.dtype)
            t_next = t[i + 1].to(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next

        # Model forward
        if per_element_schedule:
            t_curr_tensor = t_curr  # already (batch_size,)
        else:
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'sigma_hat': t_curr, 'denoised': denoised})

        x = (1 - t_next_broadcast) * denoised + t_next_broadcast * torch.randn_like(x)

    return x



@torch.no_grad()
def sample_diffusion(
    model,
    noise: torch.Tensor,
    cond_inputs: dict,
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    # Varlen support
    conditioning: tp.Optional[tp.List[dict]] = None,
    sample_rate: int = 44100,
    pretransform = None,
    mask_padding_attention: bool = False,
    use_effective_length_for_schedule: bool = False,
    headroom_seconds: float = 5.0,
    padding_mask: tp.Optional[torch.Tensor] = None,
    # Timestep schedule
    dist_shift = None,
    # Sampler options
    sampler_type: str = None,
    batch_cfg: bool = True,
    rescale_cfg: bool = False,
    # CFG options
    apg_scale: float = 1.0,
    # Init data (variation / img2img)
    init_data: tp.Optional[torch.Tensor] = None,
    init_noise_level: float = 1.0,
    # Other
    callback = None,
    disable_tqdm: bool = False,
    decode: bool = True,
    chunked_decode: tp.Optional[bool] = None,
    **sampler_kwargs
) -> torch.Tensor:
    """
    Unified sampling function for diffusion models. Handles all diffusion objectives,
    varlen support (padding_mask + effective_seq_len), timestep scheduling, and init_data
    for variation/img2img.

    Args:
        model: The diffusion model backbone (model.model, not the wrapper)
        noise: Initial noise tensor of shape (B, C, T)
        cond_inputs: Pre-processed conditioning inputs dict (merged positive + negative)
        diffusion_objective: One of "v", "rectified_flow", "rf_denoiser"
        steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        conditioning: List of conditioning dicts (for computing varlen from seconds_total)
        sample_rate: Audio sample rate
        pretransform: Optional pretransform for decoding latents and computing downsampling_ratio
        mask_padding_attention: Whether to create padding_mask for attention
        use_effective_length_for_schedule: Whether to use effective_seq_len for dist_shift
        padding_mask: Optional pre-computed padding mask (B, T). If provided, skips
            internal mask computation. Use this to ensure consistency with training masks.
        headroom_seconds: Extra seconds beyond seconds_total for valid region
        dist_shift: Distribution shift object for warping the timestep schedule, or None
        sampler_type: Sampler type. For RF: "euler", "rk4", "dpmpp", "pingpong".
            For v-diffusion: "v-ddim", "v-ddim-cfgpp", or k-diffusion types like "dpmpp-2m-sde".
        batch_cfg: Whether to use batched CFG
        rescale_cfg: Whether to use rescaled CFG
        apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG
        init_data: Optional pre-encoded latent tensor for variation/img2img (shape: B, C, T)
        init_noise_level: Noise level (sigma_max) when using init_data. 1.0 = full noise (no variation).
        callback: Optional callback for progress reporting
        disable_tqdm: Whether to disable progress bar
        decode: Whether to decode latents using pretransform
        **sampler_kwargs: Additional kwargs passed to sampler

    Returns:
        Generated samples (decoded audio if decode=True, else latents)
    """
    device = noise.device
    batch_size = noise.shape[0]
    latent_seq_len = noise.shape[-1]

    # Compute downsampling ratio
    downsampling_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

    # Default sampler_type per objective
    if sampler_type is None:
        sampler_type = "pingpong" if diffusion_objective == "rf_denoiser" else "euler"


    # Compute effective_seq_len for dist_shift if enabled
    effective_seq_len = None
    if use_effective_length_for_schedule and conditioning is not None:
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )

    # Create padding_mask for attention if enabled (skip if pre-computed mask provided)
    if padding_mask is None and mask_padding_attention and conditioning is not None:
        raw_effective_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )
        if raw_effective_len is not None:
            headroom_tokens = int(headroom_seconds * sample_rate / downsampling_ratio)
            valid_lengths = (raw_effective_len + headroom_tokens).clamp(max=latent_seq_len).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_seq_len)

    # Determine sigma_max for schedule
    sigma_max = init_noise_level if init_data is not None else 1.0

    # Mix init_data with noise for variation/img2img
    # For k-diffusion v-diffusion samplers, init_data is passed through to sample_k
    # which handles mixing internally with its own sigma scaling
    k_diff_sampler_types = {"k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2",
                            "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde", "dpmpp-2m"}

    if init_data is not None:
        noise = init_data * (1 - sigma_max) + noise * sigma_max

    # Build common sampler kwargs (conditioning + model-level params only).
    # disable_tqdm and callback are passed explicitly to samplers that use them,
    # not included here, to avoid leaking into model forward() calls.
    common_kwargs = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": batch_cfg,
        "rescale_cfg": rescale_cfg,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
        **sampler_kwargs
    }


    if diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        # Remove v-diffusion-specific kwargs that don't apply to RF
        common_kwargs.pop("sigma_min", None)
        common_kwargs.pop("sigma_max", None)
        common_kwargs.pop("rho", None)

        # Build schedule
        sigmas = build_schedule(
            steps=steps, sigma_max=sigma_max,
            dist_shift=dist_shift, effective_seq_len=effective_seq_len,
            fallback_seq_len=latent_seq_len, include_endpoint=True, device=device
        )

        # Route to sampler
        if sampler_type == "euler":
            sampled = sample_discrete_euler(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "rk4":
            sampled = sample_rk4(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "dpmpp":
            sampled = sample_flow_dpmpp(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        elif sampler_type == "pingpong":
            sampled = sample_flow_pingpong(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
        else:
            raise ValueError(f"Unknown sampler_type for {diffusion_objective}: {sampler_type}")

    else:
        raise ValueError(f"Unknown diffusion_objective: {diffusion_objective}")

    # Decode if requested
    if decode and pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled, chunked=chunked_decode)

        # Zero out audio beyond valid region (padding positions decode to garbage)
        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(downsampling_ratio, dim=-1)
            # Trim or pad to match sampled length
            if audio_mask.shape[-1] > sampled.shape[-1]:
                audio_mask = audio_mask[..., :sampled.shape[-1]]
            elif audio_mask.shape[-1] < sampled.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, sampled.shape[-1] - audio_mask.shape[-1]), value=False)
            sampled = sampled * audio_mask.to(sampled.dtype)

    return sampled

def generate_diffusion_latent_flowedit(
        model,
        src_inv_cfg_scale=6,
        tar_inv_cfg_scale=6,
        src_lfe_cfg_scale=6,
        tar_lfe_cfg_scale=6,
        src_conditioning_inputs: dict = {},
        tar_conditioning_inputs: dict = {},
        tar_prompt: str = "",
        n_avg: int = 1,
        batch_size: int = 1,
        sample_size: int = 2097152,
        # sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        deterministic_inverse = False,
        noise_amt = 1.0,
        inv_steps = 10,
        lfe_steps = 10,
        return_intermediate_latents = False,
        intermediate_latents_interval = 5,
        intermediate_latents_steps = None,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """


    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    # print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    # assert src_conditioning_tensors is not None, "Must provide src_conditioning_tensors"
    # assert tar_conditioning_tensors is not None, "Must provide tar_conditioning_tensors"
    # src_conditioning_tensors = model.conditioner(src_conditioning_inputs, device)
    # tar_conditioning_tensors = model.conditioner(tar_conditioning_inputs, device)
    
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    src_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()}
    src_conditioning_inputs_lfe = {k: v.expand(n_avg, *v.shape[1:]).type(model_dtype) if v is not None else v for k, v in src_conditioning_inputs.items()} # allowing batch processing of lfe
    tar_conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()}
    tar_conditioning_inputs_lfe = {k: v.expand(n_avg, *v.shape[1:]).type(model_dtype) if v is not None else v for k, v in tar_conditioning_inputs.items()} # allowing batch processing of latent flowedit
    intermediate_latents = []
    
    if lfe_steps > 0:
        # print('in latent flowedit')
        # latent flowedit
        z_t_lfe = init_audio.unsqueeze(0).repeat(n_avg, 1, 1, 1) # (n_avg, batch_size, channels, length)??
        # print('z_t_lfe.shape', z_t_lfe.shape)
        _capture_steps_set = set(intermediate_latents_steps) if intermediate_latents_steps is not None else None
        
        for ind, i in enumerate(np.linspace(1, 0, lfe_steps+1)[:-1]):
            t = torch.Tensor([i]).repeat(n_avg).to(device)
            noise = torch.randn_like(z_t_lfe).to(device)
            z_t_src = (1 - i) * init_audio + i * noise
    
            z_t_tar = (z_t_lfe) - init_audio + z_t_src
            # z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)

            z_t_src = z_t_src.view(-1, *z_t_src.shape[2:]) # (n_avg * batch_size, channels, length)
            z_t_tar = z_t_tar.view(-1, *z_t_tar.shape[2:]) # (n_avg * batch_size, channels, length)

            v_tar = model.model(z_t_tar, t, **tar_conditioning_inputs_lfe, cfg_scale=tar_lfe_cfg_scale)
            v_src = model.model(z_t_src, t, **src_conditioning_inputs_lfe, cfg_scale=src_lfe_cfg_scale)
            v_delta = v_tar - v_src

            v_delta = v_delta.view(n_avg, batch_size, *v_delta.shape[1:])
            v_delta = v_delta.mean(0, keepdim=True) # (1, batch_size, channels, length)
            
            z_t_lfe = z_t_lfe - v_delta/lfe_steps
            if return_intermediate_latents:
                _capture = (
                    ind in _capture_steps_set
                    if _capture_steps_set is not None
                    else ind % intermediate_latents_interval == 0
                )
                if _capture:
                    intermediate_latents.append(z_t_lfe[0].clone())
        sampled = z_t_lfe[0]

    intermediate_sampled = []
    if return_intermediate_latents:
        for ind, intermediate_latent in enumerate(intermediate_latents):
            if model.pretransform is not None:
                intermediate_latent_val = intermediate_latent.to(next(model.pretransform.parameters()).dtype)
                intermediate_latent_val = model.pretransform.decode(intermediate_latent_val)
            intermediate_sampled.append(intermediate_latent_val)
    # del src_conditioning_tensors
    # del tar_conditioning_tensors
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled, intermediate_sampled