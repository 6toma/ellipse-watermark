import torch
from typing import Union, List, Tuple
import numpy as np

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
        x_radius = width / 4

    scale_factor = x_radius / width

    if y_radius is None:
        y_radius = height * scale_factor
        # y_radius = height / 2

    x_center = width / 2 + x_offset
    y_center = height / 2 + y_offset

    # Generate the circle mask
    mask = ((x - x_center) ** 2 / y_radius ** 2 + (y - y_center) ** 2 / y_radius ** 2) <= 1

    # Generate the elliptical mask
    # mask = ((x - x_center) ** 2 / x_radius ** 2 + (y - y_center) ** 2 / y_radius ** 2) <= 1
    return mask

def generate_patch(shape, w_pattern='ring', generator=None):
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
        max_dim = max(shape[-2], shape[-1]) // 2

        for i in range(min_dim, 0, -1):
            new_x_radius = i
            scale_factor = new_x_radius / min_dim
            new_y_radius = max_dim * scale_factor


            tmp_mask = elliptical_mask((shape[-2], shape[-1]), x_radius=new_x_radius, y_radius=new_y_radius)
            tmp_mask = torch.tensor(tmp_mask)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


def get_noise(shape: Union[torch.Size, List, Tuple], pattern, generator=None):
    # for now we hard code all hyperparameters
    w_channel = 0  # id for watermarked channel

    smallest_dim = min(shape[-2], shape[-1])
    w_radius = smallest_dim // 2
    w_pattern = pattern  # watermark pattern

    # get watermark key and mask

    np_mask = elliptical_mask((shape[-2], shape[-1]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, w_channel] = torch_mask

    w_key = generate_patch(shape, w_pattern=w_pattern, generator=generator)

    # inject watermark
    assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
    assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

    init_latents = torch.randn(shape, generator=generator)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()

    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_channel, w_radius


def detect(inverted_latents, w_key, w_channel, threshold):
    inverted_latents = inverted_latents.to("cpu")

    inverted_latents = inverted_latents.squeeze(0).squeeze(0)
    # print(inverted_latents.shape)
    inverted_latents = adjust_tensor_shape(inverted_latents, w_key.squeeze(0).squeeze(0))
    # print(inverted_latents.shape)
    inverted_latents = inverted_latents.unsqueeze(0).unsqueeze(0)

    # check if one key matches
    shape = inverted_latents.shape

    np_mask = elliptical_mask((shape[-2], shape[-1]))
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, int(w_channel)] = torch_mask

    inverted_latents = inverted_latents.to("cpu")

    # calculate the distance
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))

    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

    print('distance: ', dist)
    print('threshold: ', threshold)
    print(dist <= threshold)

    return dist <= threshold, dist


def adjust_tensor_shape(tensor1, tensor2):
    shape1 = tensor1.shape
    shape2 = tensor2.shape

    changed = False
    # Adjust rows
    if shape1[0] < shape2[0]:
        # Add rows by sampling randomly
        rows_to_add = shape2[0] - shape1[0]
        additional_rows = tensor1[np.random.choice(shape1[0], rows_to_add, replace=True), :]
        tensor1 = np.vstack([tensor1, additional_rows])
        changed = True
    elif shape1[0] > shape2[0]:
        # Remove rows randomly
        tensor1 = tensor1[np.random.choice(shape1[0], shape2[0], replace=False), :]

    return torch.from_numpy(tensor1) if changed else tensor1
