'''
    Train multiscale INR
    `grid + output's 1/W, 1/H -> use as positional information`
'''

import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.grid import create_grid
from utils.grid import ResizeConfig as config


EXP_NAME = 'multi_concat'
PATH = '../stone.png'
MAX_ITERS = 1000
LR = 1e-4


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)

    from utils.grid import creat_reals_pyramid, adjust_scales
    img = torch.unsqueeze(img.permute(2, 0, 1), dim=0)
    adjust_scales(img, config)
    reals = creat_reals_pyramid(img, [], config)

    model = SirenModel(coord_dim=4, num_c=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    grids = []
    for idx, cur_scale in enumerate(reals):
        os.makedirs(f'exps/{EXP_NAME}/img/{idx}', exist_ok=True)
        img = torch.squeeze(cur_scale).permute(1, 2, 0)
        h, w, c = img.shape
        hw = torch.FloatTensor([1./h, 1./w]).to(device)
        hw = hw.repeat(h, w, 1)
        grid = create_grid(h, w, device=device)
        pos_info = torch.cat((grid, hw), dim=-1)
        grids.append(pos_info)
        reals[idx] = img

    for step in range(MAX_ITERS):
        for idx, (grid, real) in enumerate(zip(grids, reals)):
            h, w, _ = grid.shape
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