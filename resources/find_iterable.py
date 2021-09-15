import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.viz import visualize_grid


'''
    Model weight만 fixed로 두고 coord와 B를 찾도록 함
    -> 마찬가지로 안됨
'''


EXP_NAME = 'balloons_fourier'
PATH = '../inputs/balloons.png'
PTH_NAME = 'final'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-5

MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/find_iter', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_iter/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_iter/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_iter/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/find_iter/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, depth=5, w0=W0).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    model.eval()

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    img = torch.FloatTensor(img).to(device)

    find_B = torch.randn((MAPPING_SIZE, 2)) * 10.
    find_B = find_B.to(device).detach().requires_grad_(True)
    optim_B = torch.optim.Adam({find_B}, lr=LR)

    find_coord = torch.rand((h, w, 2)).to(device).detach().requires_grad_(True)
    optim_coord = torch.optim.Adam({find_coord}, lr=LR)

    loss_fn = nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()

        optim_B.zero_grad()
        x_proj = (2. * np.pi * find_coord) @ find_B.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        pred = model(mapped_input)
        B_loss = loss_fn(pred, img)
        B_loss.backward()
        optim_B.step()

        optim_coord.zero_grad()
        x_proj = (2. * np.pi * find_coord) @ find_B.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        pred = model(mapped_input)
        coord_loss = loss_fn(pred, img)
        coord_loss.backward()
        optim_coord.step()

        print(B_loss.item(), coord_loss.item())
        writer.add_scalar("B_loss", B_loss.item(), i)
        writer.add_scalar("coord_loss", coord_loss.item(), i)

        if (i+1) % 100 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/find_iter/img/{i}.jpg')
            visualize_grid(find_coord, f'exps/{EXP_NAME}/find_iter/img/{i}_grid.jpg', device)
