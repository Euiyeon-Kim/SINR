import torch
import numpy as np
from torchvision.utils import save_image
from scipy.ndimage.filters import gaussian_filter


def create_grid(h, w, device, min_v=0, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)], indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid.to(device)


def create_flatten_grid(h, w, device, min_v=0, max_v=1):
    grid_y, grid_x = torch.meshgrid([torch.linspace(min_v, max_v, steps=h),
                                     torch.linspace(min_v, max_v, steps=w)], indexing='ij')
    grid_y = torch.unsqueeze(grid_y.flatten(), dim=0)
    grid_x = torch.unsqueeze(grid_x.flatten(), dim=0)
    return grid_x.to(device), grid_y.to(device)


def shuffle_grid(h, w, device, min_v=-1, max_v=1):
    np_y, np_x = np.meshgrid(np.linspace(start=min_v, stop=max_v, num=w), np.linspace(start=min_v, stop=max_v, num=h))
    np_grid = np.stack([np_x, np_y], axis=-1)
    new_grid = np.zeros_like(np_grid)
    new_grid[:, :w // 2, :] = np_grid[:, w // 2:, :]
    new_grid[:, w // 2:, :] = np_grid[:, :w // 2, :]
    # new_grid[:h//2, :, :] = np_grid[h//2:, :, :]
    # new_grid[h//2:, :, :] = np_grid[:h//2, :, :]
    # new_grid = gaussian_filter(new_grid, sigma=0.5)
    np2torch = torch.FloatTensor(new_grid).to(device)
    return np2torch


def cutout_grid(h, w, device, min_v=-1, max_v=1):
    np_y, np_x = np.meshgrid(np.linspace(start=min_v, stop=max_v, num=w), np.linspace(start=min_v, stop=max_v, num=h))
    np_grid = np.stack([np_x, np_y], axis=-1)
    tmp = np.split(np_grid, 4, axis=1)
    cutoutted = np.concatenate((tmp[0], tmp[-1]), axis=1)
    np2torch = torch.FloatTensor(cutoutted).to(device)
    return np2torch


def visualize_grid(grid, path, device, name=None):
    h, w, _ = grid.shape
    r = torch.ones((h, w)).to(device) * 128.
    g = (grid[:, :, 0] + 1.) * 127.5
    b = (grid[:, :, 1] + 1.) * 127.5
    img = torch.stack([r, g, b], dim=0) / 255.
    if name:
        print(f'=============== {name} ===============')
        print(f'x_min: {torch.min(grid[:, :, 1]).item():.4f}, x_max: {torch.max(grid[:, :, 1]).item():.4f}')
        print(f'y_min: {torch.min(grid[:, :, 0]).item():.4f}, y_max: {torch.max(grid[:, :, 0]).item():.4f}')
    save_image(img, path)

