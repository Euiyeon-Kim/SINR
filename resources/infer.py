import os

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.grid import create_grid

EXP_NAME = 'single_balloon'
PATH = '../inputs/balloons.png'
PTH_NAME = 'final'
INF_H = 512
INF_W = 768


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/infer', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2, num_c=3, depth=3).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))

    grid = create_grid(INF_H, INF_W, device=device)
    in_f = grid.shape[-1]

    model.eval()
    pred = model(grid)

    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/infer/{INF_H}_{INF_W}.jpg')

