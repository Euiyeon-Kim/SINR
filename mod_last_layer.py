import numpy as np

import torch

from models.siren import SirenModel
from utils.grid import create_grid
from utils.utils import make_exp_dirs, get_device, prepare_fourier_inp

'''
'''
EXP_NAME = 'balloons_fourier/modulate_last_layer'
PTH_PATH = 'exps/balloons_fourier/ckpt/final.pth'
B_PATH = 'exps/balloons_fourier/ckpt/B.pt'

W0 = 50
MAPPING_SIZE = 256


if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME, log=True)

    device = get_device()
    model = SirenModel(coord_dim=2*MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH)

    grid = create_grid(infer_h, infer_w, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
