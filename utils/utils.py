import math

import torch
import numpy as np

from utils.image import resize_img


class ResizeConfig:
    # [PYRAMID PARAMETERS]
    scale_factor = 0.75  # Pyramid scale factor pow(0.5, 1/6)
    min_size = 25  # Image minimal size at the coarser scale
    max_size = 250  # Image maximal size at the coarser scale
    noise_amp = 0.1  # Additive noise cont weight

    scale_factor_init = scale_factor
    useGPU = True


def adjust_scales(real, config):
    minwh = min(real.shape[2], real.shape[3])
    maxwh = max(real.shape[2], real.shape[3])
    config.num_scales = math.ceil(math.log(config.min_size / minwh, config.scale_factor_init)) + 1
    scale2stop = math.ceil(math.log(min([config.max_size, maxwh]) / maxwh, config.scale_factor_init))
    config.stop_scale = config.num_scales - scale2stop
    config.start_scale = min(config.max_size / maxwh, 1)
    resized_real = resize_img(real, config.start_scale, config)
    config.scale_factor = math.pow(config.min_size/min(resized_real.shape[2], resized_real.shape[3]), 1/config.stop_scale)
    scale2stop = math.ceil(math.log(min([config.max_size, maxwh]) / maxwh, config.scale_factor_init))
    config.stop_scale = config.num_scales - scale2stop
    return resized_real


def creat_reals_pyramid(real, reals, config):
    for i in range(config.stop_scale+1):
        scale = math.pow(config.scale_factor, config.stop_scale-i)
        curr_real = resize_img(real, scale, config)
        reals.append(curr_real)
    return reals


def create_grid(h, w, device, min_v=0, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def create_flatten_grid(h, w, device, min_v=0, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)])
    grid_y = torch.unsqueeze(grid_y.flatten(), dim=0)
    grid_x = torch.unsqueeze(grid_x.flatten(), dim=0)
    return grid_x.to(device), grid_y.to(device)


def shuffle_grid(h, w, device, min_v=0, max_v=1):
    np_y, np_x = np.meshgrid(np.linspace(start=min_v, stop=max_v, num=w), np.linspace(start=min_v, stop=max_v, num=h))
    np_grid = np.stack([np_x, np_y], axis=-1)
    new_grid = np.zeros_like(np_grid)
    new_grid[:, :w // 2, :] = np_grid[:, w // 2:, :]
    new_grid[:, w // 2:, :] = np_grid[:, :w // 2, :]
    np2torch = torch.FloatTensor(new_grid).to(device)
    return np2torch


