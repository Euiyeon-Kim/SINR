import math

import torch

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


# To Do - Improve making pyramid process
def creat_reals_pyramid(real, reals, config):
    for i in range(config.stop_scale+1):
        scale = math.pow(config.scale_factor, config.stop_scale-i)
        curr_real = resize_img(real, scale, config)
        reals.append(curr_real)
    return reals


def create_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def calcul_gp(discriminator, real, fake, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real.size())
    alpha = alpha.to(device)

    interpolated = alpha * real + ((1 - alpha) * fake)
    interpolated = interpolated.to(device)
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
    interpolated_prob_out = discriminator(interpolated)

    gradients = torch.autograd.grad(outputs=interpolated_prob_out, inputs=interpolated,
                                    grad_outputs=torch.ones(interpolated_prob_out.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp
