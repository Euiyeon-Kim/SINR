import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import create_grid
from utils.viz import visualize_grid

'''
    Learned B를 loading해서 fixed,
    Model weight도 fixed로 두고 coord 를 찾도록 함
    -> 얘도 안됨
'''

EXP_NAME = 'balloons_fourier'
PATH = 'inputs/balloons.png'
PTH_NAME = 'final'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/find_coord/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_coord/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_coord/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/find_coord/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    model.eval()

    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)

    B_real = torch.load(f'exps/{EXP_NAME}/ckpt/B.pt')  # torch.load(f'exps/{EXP_NAME}/ckpt/B.pt')
    real_x = (2. * np.pi * grid) @ B_real.t()
    real_input = torch.cat([torch.sin(real_x), torch.cos(real_x)], dim=-1)
    origin = model(real_input)
    save_image(origin.permute(2, 0, 1), f'exps/{EXP_NAME}/find_coord/img/origin.jpg')

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

        if (i+1) % 100 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/find_coord/img/{i}.jpg')
            visualize_grid(find_coord, f'exps/{EXP_NAME}/find_coord/img/{i}_grid.jpg', device)

    torch.save(find_coord, f'exps/{EXP_NAME}/find_coord/ckpt/coord.pt')
