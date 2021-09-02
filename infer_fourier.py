import os

import numpy as np

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.utils import create_grid

EXP_NAME = 'fourier_siren_mountain'
PATH = './inputs/mountains.jpg'
PTH_NAME = 'final'
INF_H = 366
INF_W = 585
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/infer', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, depth=5).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    B = torch.load(f'exps/{EXP_NAME}/ckpt/B.pt')
    # B = torch.randn((MAPPING_SIZE, 2)).to(device) * 10
    grid = create_grid(INF_H, INF_W, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    model.eval()
    pred = model(mapped_input)

    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/infer/rand_{INF_H}_{INF_W}.jpg')

