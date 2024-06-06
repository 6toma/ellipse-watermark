import torch
from typing import Union, List, Tuple
import numpy as np


# def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
#     # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
#     x0 = y0 = size // 2
#     x0 += x_offset
#     y0 += y_offset
#     y, x = np.ogrid[:size, :size]
#     y = y[::-1]
#
#     return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2

def _circle_mask(shape, r=10, x_offset=0, y_offset=0):
    """
    Generates a circular mask for the given shape.
    """
    height, width = shape
    y, x = np.ogrid[:height, :width]
    y = y[::-1]
    x_center = width // 2 + x_offset
    y_center = height // 2 + y_offset
    return ((x - x_center) ** 2 + (y - y_center) ** 2) <= r ** 2


def elliptical_mask(shape, x_radius=None, y_radius=None, x_offset=0, y_offset=0):
    height, width = shape
    y, x = np.ogrid[:height, :width]
    y = y[::-1]

    # Calculate the radii if not provided
    if x_radius is None:
        x_radius = width
    if y_radius is None:
        y_radius = height

    x_center = width / 2 + x_offset
    y_center = height / 2 + y_offset

    # Generate the elliptical mask
    mask = ((x - x_center) ** 2 / x_radius ** 2 + (y - y_center) ** 2 / y_radius ** 2) <= 1
    return mask

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

        min_dim = min(shape[-2], shape[-1]) // 2
        for i in range(min_dim, 0, -1):
            tmp_mask = elliptical_mask((shape[-2], shape[-1]))
            tmp_mask = torch.tensor(tmp_mask)
            # gt_patch[tmp_mask] = gt_patch_tmp[0, i].item()

            for j in range(gt_patch.shape[1]):
                # gt_patch[j, tmp_mask[j]] = gt_patch_tmp[i, j].item()
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        # for i in range(shape[-1] // 2, 0, -1):
        #     tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
        #     tmp_mask = torch.tensor(tmp_mask)
        #
        #     for j in range(gt_patch.shape[1]):
        #         gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def get_noise(shape: Union[torch.Size, List, Tuple], pattern, generator=None):
    # for now we hard code all hyperparameters
    w_channel = 0  # id for watermarked channel

    smallest_dim = min(shape[-2], shape[-1])
    w_radius = smallest_dim // 2
    w_pattern = pattern  # watermark pattern

    # get watermark key and mask
    # np_mask = _circle_mask((shape[-2], shape[-1]), r=w_radius)
    np_mask = elliptical_mask((shape[-2], shape[-1]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, w_channel] = torch_mask

    w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator)

    # inject watermark
    assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
    assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

    init_latents = torch.randn(shape, generator=generator)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_channel, w_radius


def detect(inverted_latents, w_key, w_channel, w_radius):
    threshold = 77

    # check if one key matches
    shape = inverted_latents.shape

    # np_mask = _circle_mask((shape[-2], shape[-1]), r=int(w_radius))
    np_mask = elliptical_mask((shape[-2], shape[-1]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, int(w_channel)] = torch_mask

    # calculate the distance
    inverted_latents = inverted_latents.to("cpu")
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()
    # dist = torch.cdist(inverted_latents_fft[w_mask], w_key[w_mask])
    # dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(inverted_latents_fft[w_mask], w_key[w_mask]), 2), dim=0))
    # dist = torch.norm(inverted_latents_fft[w_mask] - w_key[w_mask]).item()

    print('distance: ', dist)

    if dist <= threshold:
        return True, dist

    return False, dist
