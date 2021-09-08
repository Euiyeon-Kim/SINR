import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.maml import SirenModel
from utils.utils import create_grid
from utils.viz import visualize_grid

EXP_NAME = 'learnit_mountains_patch/poc'
PTH_PATH = 'exps/learnit_mountains_patch/maml/ckpt/final.pth'
PATH = 'inputs/mountains.jpg'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-5


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.float32(img) / 255
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)

    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    model.eval()

    pred = model(grid)
    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/recon.jpg')
    visualize_grid(grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    find = grid.to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find}, lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.eval()
        optim.zero_grad()

        pred = model(find)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        if (i+1) % 10 == 0:
            pred = pred.permute(2, 0, 1)
            visualize_grid(find, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_grid.jpg', device)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_find.jpg')