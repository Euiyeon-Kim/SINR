import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import FourierReLU
from utils.grid import create_grid, visualize_grid

'''
    Learned B를 loading해서 fixed,
    Model weight도 fixed로 두고 coord 를 찾도록 함
    -> 얘도 안됨
'''

EXP_NAME = 'fourier_relu/find_coord/balloon'
PATH = '../inputs/balloons.png'
PTH_PATH = '../exps/fourier_relu/origin/balloon/ckpt/final.pth'
B_PATH = '../exps/fourier_relu/origin/balloon/ckpt/B.pt'

MAX_ITERS = 10000
LR = 1e-4

MAPPING_SIZE = 64


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    model = FourierReLU(coord_dim=MAPPING_SIZE * 2, num_c=3, hidden_node=256, depth=5).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    model.eval()

    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)

    B_real = torch.load(B_PATH)
    real_x = (2. * np.pi * grid) @ B_real.t()
    real_input = torch.cat([torch.sin(real_x), torch.cos(real_x)], dim=-1)
    origin = model(real_input)
    save_image(origin.permute(2, 0, 1), f'exps/{EXP_NAME}/img/origin.jpg')

    find_coord = torch.zeros((h, w, 2)).to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find_coord}, lr=LR)
    loss_fn = nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()

        x_proj = (2. * np.pi * find_coord) @ B_real.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        pred = model(mapped_input)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        print(loss.item())
        writer.add_scalar("loss", loss.item(), i)

        if i % 100 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i+1}.jpg')
            visualize_grid(find_coord, f'exps/{EXP_NAME}/img/{i+1}_grid.jpg', device)

    torch.save(find_coord, f'exps/{EXP_NAME}/ckpt/coord.pt')
