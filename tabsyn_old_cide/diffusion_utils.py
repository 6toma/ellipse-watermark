"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import numpy as np
# from _get_noise import get_noise
from scipy.stats import betaprime

from typing import Union, List, Tuple


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1


# def sample(net, num_samples, dim, w_pattern='ring', num_steps=50, device='mps'):
def sample(net, num_samples, dim, w_pattern='ring', num_steps=50, device='cpu'):
    # encode key watermark in HERE

    # Sample zT ∼ N(0,σ2(T)I),tmax = T
    latents = torch.randn([num_samples, dim], device=device)
    #
    # # add noise to initial noise vector, test data quality after sampling
    # noise_level = 0.1
    # noise = torch.randn(latents.size(), device=latents.device) * noise_level
    # latents = latents + noise

    # shape = [num_samples, dim]
    # latents = get_noise(shape, pattern=w_pattern)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(t_cur)
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, device)

    return x_next


def sample_step(net, num_steps, i, t_cur, t_next, x_next, device):
    x_cur = x_next
    # Increase noise temporarily.
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
    # Euler step.

    denoised = net(x_hat.to(device), t_hat.to(device)).to(torch.float32)
    d_cur = (x_hat.to(device) - denoised.to(device)) / t_hat.to(device)
    x_next = x_hat.to(device) + (t_next.to(device) - t_hat.to(device)) * d_cur.to(device)

    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next.to(device), t_next.to(device)).to(torch.float32)
        d_prime = (x_next.to(device) - denoised.to(device)) / t_next.to(device)
        x_next = x_hat.to(device) + (t_next.to(device) - t_hat.to(device)) * (0.5 * d_cur.to(device) + 0.5 * d_prime.to(device))

    return x_next


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels=None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N,
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1 - 1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def _get_pattern(shape, w_pattern='ring', generator=None):
    gt_init = torch.randn(shape, generator=generator)

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


# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str) -> torch.Tensor:
# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str, generator=None) -> Tuple[
def get_noise(shape: Union[torch.Size, List, Tuple], pattern='ring', generator=None): # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # for now we hard code all hyperparameters
    w_channel = 0  # id for watermarked channel
    w_radius = 10  # watermark radius
    w_pattern = pattern  # watermark pattern
    # get watermark key and mask
    np_mask = _circle_mask(shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    # w_mask[:, w_channel] = torch_mask

    expanded_mask = torch_mask.unsqueeze(0).expand(shape[0], -1, -1)

    # Assign the expanded mask to each row of w_mask along the specified channel
    for i in range(shape[0]):
        w_mask[i, :] = expanded_mask[i, :, w_channel]

    w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator)

    # inject watermark
    # assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
    # assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

    init_latents = torch.randn(shape, generator=generator)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents
