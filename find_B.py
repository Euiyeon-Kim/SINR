import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import create_grid

EXP_NAME = 'fourier_siren_mountain'
PATH = './inputs/mountains_patch/1_9.jpg'
PTH_NAME = 'final'

MAX_ITERS = 100000
LR = 1e-5

MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/find_B', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_B/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_B/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_B/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/find_B/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, depth=5).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    model.eval()

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)  # .permute(2, 0, 1)
    # grid = RandomCrop(64)(grid).permute(1, 2, 0)

    B_real = torch.load(f'exps/{EXP_NAME}/ckpt/B.pt')
    real_x = (2. * np.pi * grid) @ B_real.t()
    real_input = torch.cat([torch.sin(real_x), torch.cos(real_x)], dim=-1)
    origin = model(real_input)
    save_image(origin.permute(2, 0, 1), f'exps/{EXP_NAME}/find_B/img/origin.jpg')

    B = torch.randn((MAPPING_SIZE, 2)) * 10.
    B = B.to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({B}, lr=LR)
    loss_fn = nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()

        x_proj = (2. * np.pi * grid) @ B.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        pred = model(mapped_input)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        print(loss.item())
        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 1000 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/find_B/img/{i}.jpg')

    torch.save(B, f'exps/{EXP_NAME}/find_B/ckpt/B.pt')
