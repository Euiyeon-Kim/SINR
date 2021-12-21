import os
from glob import glob

import numpy as np

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import sample_B, create_grid, read_img


EXP_NAME = f'flood/bf_siren_64_5/bush'
PATH = '../inputs/wild_bush_patch_96*128/5_17.jpg'

PATCH_H = 96
PATCH_W = 128

W0 = 50
MAX_ITERS = 1000
LR = 1e-4

SCALE = 10
MAPPING_SIZE = 64


if __name__ == '__main__':
    path = PATH

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(PATCH_H, PATCH_W, device)
    B = sample_B(MAPPING_SIZE, SCALE, device)
    mapped_input = torch.sin((2. * np.pi * grid) @ B.t())

    img_idx = path.split('/')[-1][:-4]

    os.makedirs(f'exps/{EXP_NAME}/{img_idx}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/{img_idx}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/{img_idx}/logs')

    img = torch.FloatTensor(read_img(path)).to(device)
    model = SirenModel(coord_dim=MAPPING_SIZE, num_c=3, hidden_node=64, depth=5, w0=50).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        optim.zero_grad()

        pred = model(mapped_input)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i % 10 == 0:
            pred_img = (pred * 255.).detach().cpu().numpy()
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/{img_idx}/img/{i}.jpg')

    pred_img = (pred * 255.).detach().cpu().numpy()
    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/{img_idx}/img/last.jpg')
