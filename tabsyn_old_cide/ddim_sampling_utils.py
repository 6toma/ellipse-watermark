import torch
import numpy as np
from typing import Union, List, Tuple

rho = 7
SIGMA_MIN = 0.002
SIGMA_MAX = 80


def ddim_sample(net,
                num_samples,
                sample_dim,
                num_steps,
                w_pattern='rand',
                device='cpu',
                eta=0,
                beta_schedule="linear",
                beta_start: float = 0.0001,
                beta_end: float = 0.02, ):
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

    shape = [num_samples, sample_dim]
    x_t, key, channel, radius = get_noise(shape, pattern=w_pattern)

    # print(str(x_t))
    # latents
    x_t = torch.randn([num_samples, sample_dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_t.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    for t in reversed(range(num_steps)):
        t_tensor = torch.full((x_t.shape[0],), t_steps[t], dtype=torch.float32, device=x_t.device)
        model_output = net(x_t, t_tensor)

        prev_timestep = t - 1
        alpha_t = alpha_cumprod[t]
        alpha_t_prev = alpha_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

        x_0 = (x_t - np.sqrt(1 - alpha_t) * model_output) / np.sqrt(alpha_t)
        if t > 0:
            x_t = np.sqrt(alpha_t_prev) * x_0 + np.sqrt(1 - alpha_t_prev) * model_output
        else:
            x_t = x_0

    return x_t, key, channel, radius


def ddim_invert(net,
                data,
                num_steps,
                device='cpu',
                beta_schedule="linear",
                beta_start: float = 0.0001,
                beta_end: float = 0.02, ):
    """
    DDIM inversion method.

    Args:
        net: The trained Model instance.
        data: The observed data to invert.
        num_steps: Number of inversion steps.

    Returns:
        Initial latents.
    """
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
    elif beta_schedule == "scaled_linear":
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float32, device=device) ** 2

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    final_alpha_cumprod = torch.tensor(1.0, device=device)

    shape = data.shape

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_t = data
    for t in range(num_steps):
        t_tensor = torch.full((x_t.shape[0],), t_steps[t], dtype=torch.float32, device=x_t.device)
        model_output = net(x_t, t_tensor)

        next_timestep = t + 1
        alpha_t = alpha_cumprod[t]
        alpha_t_next = alpha_cumprod[next_timestep] if next_timestep < num_steps else final_alpha_cumprod

        x_0 = (x_t - np.sqrt(1 - alpha_t) * model_output) / np.sqrt(alpha_t)

        if t < num_steps - 1:
            x_t = np.sqrt(alpha_t_next) * x_0 + np.sqrt(1 - alpha_t_next) * model_output
        else:
            x_t = x_0

    return x_t


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def _get_pattern(shape, w_pattern='ring'):
    gt_init = torch.randn(shape)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = gt_patch.clone().detach()
        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)
            for j in range(gt_patch.shape[1]):
                gt_patch[j, tmp_mask[j]] = gt_patch_tmp[j, i].item()

    return gt_patch


def get_noise(shape: Union[torch.Size, List, Tuple], pattern='ring'):
    # for now we hard code all hyperparameters
    w_channel = 0  # id for watermarked channel
    w_radius = 10  # watermark radius
    w_pattern = pattern  # watermark pattern
    # get watermark key and mask
    np_mask = _circle_mask(shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)

    # expanded_mask = torch_mask.unsqueeze(0).expand(shape[0], -1, -1)
    #
    # # Assign the expanded mask to each row of w_mask along the specified channel
    # for i in range(shape[0]):
    #     w_mask[i, :] = expanded_mask[i, :, w_channel]

    y = shape[0]
    x = shape[1]
    num_repeats = (y + x - 1) // x  # This ensures you have enough repeats

    # Repeat the torch_mask to cover the required number of rows
    repeated_mask = torch_mask.repeat(num_repeats, 1)

    # Truncate to match the exact size of w_mask
    expanded_mask = repeated_mask[:y, :]

    # Apply the expanded_mask to w_mask
    w_mask[:, :] = expanded_mask

    w_key = _get_pattern(shape, w_pattern=w_pattern)

    init_latents = torch.randn(shape)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_channel, w_radius


def detect(data, w_key, w_channel, w_radius, net, num_steps):
    threshold = 300

    # ddim inversion

    inverted_latents = ddim_invert(net, data, num_steps)

    # print(str(inverted_latents))

    shape = data.shape

    np_mask = _circle_mask(shape[-1], r=int(w_radius))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    # w_mask[:, int(w_channel)] = torch_mask
    # print(str(torch_mask.shape))
    # expanded_mask = torch_mask.unsqueeze(0).expand(shape[0], -1, -1)
    # # Assign the expanded mask to each row of w_mask along the specified channel
    # for i in range(shape[0]):
    #     w_mask[i, :] = expanded_mask[i, :, w_channel]

    y = shape[0]
    x = shape[1]
    num_repeats = (y + x - 1) // x  # This ensures you have enough repeats

    # Repeat the torch_mask to cover the required number of rows
    repeated_mask = torch_mask.repeat(num_repeats, 1)

    # Truncate to match the exact size of w_mask
    expanded_mask = repeated_mask[:y, :]

    # Apply the expanded_mask to w_mask
    w_mask[:, :] = expanded_mask

    # calculate the distance
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    # print('\n\n' + str(inverted_latents_fft))
    # print('\n\n' + str(inverted_latents_fft[w_mask]))
    # print('\n\n' + str(w_key))
    # print('\n\n' + str(w_key[w_mask]))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

    print(dist)
    # print(str(w_mask.shape))
    # print(str(w_key.shape))
    # print(str(inverted_latents_fft.shape))

    if dist <= threshold:
        return True

    return False
