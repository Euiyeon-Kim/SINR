import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from model import SirenModel
from utils.utils import create_grid
from utils.utils import ResizeConfig as config


EXP_NAME = 'poc'
PATH = 'stone.png'
PTH_NAME = 'final'
INF_H = 512
INF_W = 512


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/infer', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2, num_c=3).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))

    grid = create_grid(INF_H, INF_W, device=device)
    in_f = grid.shape[-1]

    model.eval()
    pred = model(grid)

    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/infer/{INF_H}_{INF_W}.jpg')

