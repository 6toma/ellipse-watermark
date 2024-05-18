import torch
import numpy as np


def ddim_sample(net,
                num_samples,
                sample_dim,
                num_steps,
                device='cuda',
                eta=0,
                beta_schedule="linear",
                beta_start: float = 0.0001,
                beta_end: float = 0.02,):
    """
    DDIM sampling method.

    Args:
        net: The trained Model instance.
        num_samples: Number of samples to generate.
        sample_dim: Dimensionality of the samples.
        num_steps: Number of sampling steps.
        eta: Controls the variance of the noise, eta=0 means deterministic DDIM.

    Returns:
        Generated samples.
    """
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    elif beta_schedule == "scaled_linear":
        # this schedule is very specific to the latent diffusion model.
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    final_alpha_cumprod = torch.tensor(1.0)

    # latents
    x_t = torch.randn([num_samples, sample_dim], device=device)

    for t in reversed(range(num_steps)):
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.int64, device=x_t.device)
        epsilon_theta = net(x_t, t_tensor)

        prev_timestep = t - 1
        alpha_t = alpha_cumprod[t]
        alpha_t_prev = alpha_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

        sigma_t = eta * np.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * np.sqrt(1 - alpha_t / alpha_t_prev)

        x_0 = (x_t - np.sqrt(1 - alpha_t) * epsilon_theta) / np.sqrt(alpha_t)

        if t > 0:
            z = torch.randn_like(x_t)
            x_t = np.sqrt(alpha_t_prev) * x_0 + np.sqrt(1 - alpha_t_prev - sigma_t ** 2) * epsilon_theta + sigma_t * z
        else:
            x_t = np.sqrt(alpha_t_prev) * x_0

    return x_t

# def ddim_sample_step(prev_value, timestep)