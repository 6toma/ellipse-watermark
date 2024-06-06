import torch
import numpy as np
from typing import Union, List, Tuple


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
    width, height = shape
    y, x = np.ogrid[:height, :width]
    y = y[::-1]

    # Calculate the radii if not provided
    if x_radius is None:
        x_radius = width / 2
    if y_radius is None:
        y_radius = height / 2

    x_center = width / 2 + x_offset
    y_center = height / 2 + y_offset

    # Generate the elliptical mask
    mask = ((x - x_center) ** 2 / x_radius ** 2 + (y - y_center) ** 2 / y_radius ** 2) <= 1
    return mask


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
        # for i in range(shape[-1] // 2, 0, -1):
        # print(gt_patch_tmp.shape)
        min_dim = min(shape[0], shape[1]) // 2
        # max_dim = max(shape[0], shape[1])
        print(shape)
        for i in range(min_dim, 0, -1):
            # print(i)
            # tmp_mask = _circle_mask((shape[0], shape[1]), r=i)
            tmp_mask = elliptical_mask((shape[-1], shape[-2]))
            tmp_mask = torch.tensor(tmp_mask)
            # gt_patch[tmp_mask] = gt_patch_tmp[0, i].item()
            # gt_patch[tmp_mask] = gt_patch_tmp[0, i].item()
            for j in range(gt_patch.shape[-1]):
                gt_patch[j, tmp_mask[j]] = gt_patch_tmp[i, j].item()

            # for j in range(min_dim):
            #     print(j)
            #     gt_patch[tmp_mask] = gt_patch_tmp[i, j].item()

    return gt_patch


def get_noise(shape: Union[torch.Size, List, Tuple], pattern='ring'):
    w_channel = 0  # id for watermarked channel
    w_pattern = pattern  # watermark pattern

    smallest_dim = min(shape[-2], shape[-1])
    w_radius = smallest_dim // 2

    # get watermark key and mask
    # np_mask = _circle_mask(shape[-1], r=w_radius)
    # np_mask = _circle_mask((shape[-2], shape[-1]), r=w_radius)
    np_mask = elliptical_mask((shape[-1], shape[-2]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)

    # expanded_mask = torch_mask.unsqueeze(0).expand(shape[0], -1, -1)
    #
    # # Assign the expanded mask to each row of w_mask along the specified channel
    # for i in range(shape[0]):
    #     w_mask[i, :] = expanded_mask[i, :, w_channel]

    # y = shape[0]
    # x = shape[1]
    # num_repeats = (y + x - 1) // x  # This ensures you have enough repeats
    #
    # # Repeat the torch_mask to cover the required number of rows
    # repeated_mask = torch_mask.repeat(num_repeats, 1)
    #
    # # Truncate to match the exact size of w_mask
    # expanded_mask = repeated_mask[:y, :]

    # Apply the expanded_mask to w_mask
    w_mask[:, :] = torch_mask

    w_key = _get_pattern(shape, w_pattern=w_pattern)

    init_latents = torch.randn(shape)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_channel, w_radius

def detect(inverted_latents, w_key, w_channel, w_radius):
    threshold = 4148

    shape = inverted_latents.shape

    # np_mask = _circle_mask((shape[-2], shape[-1]), r=int(w_radius))
    np_mask = elliptical_mask((shape[-1], shape[-2]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, :] = torch_mask

    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()
    # dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(inverted_latents_fft[w_mask], w_key[w_mask]), 2), dim=0)).to(torch.float32).item()
    # dist = torch.norm(inverted_latents_fft[w_mask] - w_key[w_mask])
    # dist = torch.cdist(inverted_latents_fft[w_mask], w_key[w_mask])

    print(dist)

    if dist <= threshold:
        return True, dist

    return False, dist

# def detect(inverted_latents, w_key, w_channel, w_radius):
#     threshold = 4148
#
#     shape = inverted_latents.shape
#
#     # np_mask = _circle_mask(shape[-1], r=int(w_radius))
#     np_mask = _circle_mask((shape[-2], shape[-1]), r=int(w_radius))
#     torch_mask = torch.tensor(np_mask)
#     w_mask = torch.zeros(shape, dtype=torch.bool)
#     # w_mask[:, int(w_channel)] = torch_mask
#     # print(str(torch_mask.shape))
#     # expanded_mask = torch_mask.unsqueeze(0).expand(shape[0], -1, -1)
#     # # Assign the expanded mask to each row of w_mask along the specified channel
#     # for i in range(shape[0]):
#     #     w_mask[i, :] = expanded_mask[i, :, w_channel]
#
#     # y = shape[0]
#     # x = shape[1]
#     # num_repeats = (y + x - 1) // x  # This ensures you have enough repeats
#     #
#     # # Repeat the torch_mask to cover the required number of rows
#     # repeated_mask = torch_mask.repeat(num_repeats, 1)
#     #
#     # # Truncate to match the exact size of w_mask
#     # expanded_mask = repeated_mask[:y, :]
#
#     # Apply the expanded_mask to w_mask
#     w_mask[:, :] = torch_mask
#     # print(inverted_latents_fft.shape)
#     # print(w_key.shape)
#     # calculate the distance
#     inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
#     dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()
#
#     print(dist)
#
#     if dist <= threshold:
#         return True, dist
#
#     return False, dist
