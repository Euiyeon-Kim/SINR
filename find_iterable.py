import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import create_grid


EXP_NAME = 'fourier_siren_mountain'
PATH = './inputs/mountains_patch/1_9.jpg'
PTH_NAME = 'final'

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

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, depth=5).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    model.eval()

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    img = torch.FloatTensor(img).to(device)

    find_B = torch.randn((MAPPING_SIZE, 2)) * 10.
    find_B = find_B.to(device).detach().requires_grad_(True)
    optim_B = torch.optim.Adam({find_B}, lr=LR)

    find_coord = create_grid(h, w, device)
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

        if (i+1) % 1000 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/find_iter/img/{i}.jpg')

    torch.save(B, f'exps/{EXP_NAME}/find_iter/ckpt/B.pt')
