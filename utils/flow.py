import torch
from torch import nn


def warp(x, real_grid, flow, device):
    """
    :param x: Fourier input (H, W, 256*2)
    :param flow (2, H, W)
    :return:
    """
    x = torch.unsqueeze(x, dim=0).permute(0, 3, 1, 2)
    real_grid = torch.unsqueeze(real_grid, dim=0).permute(0, 3, 1, 2)

    B, C, H, W = x.shape
    xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
    grid = torch.cat((xx, yy), 1).float().to(device)

    vgrid = torch.autograd.Variable(grid) + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # [2, H, W] -> [H, W, 2]

    # Flow만큼 mapped_input 옮기기
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    moved_real_grid = nn.functional.grid_sample(real_grid, vgrid, align_corners=True)

    # 1인 마스크를 flow로 옮겼을 때 1보다 작으면 0, 크면 1
    # 픽셀 값이 채워지면 1 아니면 0
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    mask = mask.masked_fill_(mask < 0.999, 0)
    mask = mask.masked_fill_(mask > 0, 1)

    ret = (output * mask).permute(0, 2, 3, 1).squeeze()

    return ret, moved_real_grid

