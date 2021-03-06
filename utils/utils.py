import os

import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter


def create_grid(h, w, device, min_v=-1, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)], indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def read_img(path):
    img = (np.array(Image.open(path).convert('RGB')) / 127.5) - 1.
    return img


def sample_B(mapping_size, scale, device):
    B_gauss = torch.randn((mapping_size, 2)).to(device) * scale
    return B_gauss


def grid_to_fourier_inp(grid, B):
    x_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * grid), B)
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return mapped_input


def prepare_siren_inp(path, device):
    img = torch.FloatTensor(read_img(path)).to(device)
    h, w, _ = img.shape
    grid = create_grid(h, w, device=device)
    return img, grid


def prepare_fourier_inp(path, device, mapping_size=256, scale=10):
    img, grid = prepare_siren_inp(path, device)
    B = sample_B(mapping_size, scale, device)
    mapped_input = grid_to_fourier_inp(grid, B)
    return img, B, mapped_input


def make_exp_dirs(exp_name, log=True):
    os.makedirs(f'exps/{exp_name}/img', exist_ok=True)
    os.makedirs(f'exps/{exp_name}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{exp_name}/logs', exist_ok=True)
    if log:
        return SummaryWriter(f'exps/{exp_name}/logs')


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")