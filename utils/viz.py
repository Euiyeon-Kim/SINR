import numpy as np
import torch
from torchvision.utils import save_image


def visualize_grid(grid, path, device, name='None'):
    h, w, _ = grid.shape
    r = torch.ones((h, w)).to(device) * 128.
    g = grid[:, :, 0] * 255.
    b = grid[:, :, 1] * 255.
    img = torch.stack([r, g, b], dim=0) / 255.
    print(f'=============== {name} ===============')
    print(f'x_min: {torch.min(grid[:, :, 1]).item():.4f}, x_max: {torch.max(grid[:, :, 1]).item():.4f}')
    print(f'y_min: {torch.min(grid[:, :, 0]).item():.4f}, y_max: {torch.max(grid[:, :, 0]).item():.4f}')
    save_image(img, path)
