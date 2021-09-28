import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.grid import create_grid
from models.siren import SirenModel

'''
    픽셀 수 스케일링 해보기
'''

EXP_NAME = 'learn-multi'
DATA_ROOT = 'inputs/balloons_multiscale'

W0 = 50
MAX_ITERS = 2000
LR = 1e-4

SCALE = 10
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    # B_gauss = torch.randn((MAPPING_SIZE, 2)).to(device) * SCALE

    grids = []
    reals = []
    real_paths = sorted(glob(f'{DATA_ROOT}/*'))
    for idx, p in enumerate(real_paths):
        img = Image.open(p).convert('RGB')
        img = np.array(img) / 255.
        img = torch.FloatTensor(img).to(device)
        h, w, c = img.shape

        os.makedirs(f'exps/{EXP_NAME}/img/{idx}', exist_ok=True)
        grid = create_grid(h, w, device=device)
        grids.append(grid)
        reals.append(img)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for step in range(MAX_ITERS):
        for idx, (real, grid) in enumerate(zip(reals, grids)):
            model.train()
            optim.zero_grad()

            # x_proj = (2. * np.pi * grid) @ B_gauss.t()
            # mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

            pred = model(grid)
            loss = loss_fn(pred, real)

            loss.backward()
            optim.step()

            print(f'{idx}|{step}: loss:{loss.item()}')
            writer.add_scalar(f"{idx}_loss", loss.item(), step)

            if (step+1) % 50 == 0:
                pred = pred.permute(2, 0, 1)
                save_image(pred, f'exps/{EXP_NAME}/img/{idx}/{step}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')