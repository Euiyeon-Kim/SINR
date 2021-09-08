import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.utils import create_grid
from models.maml import SirenModel

EXP_NAME = 'learnit_mountains_patch/maml'
PTH_PATH = 'exps/learnit_mountains_patch/ckpt/20000.pth'
PATH = 'inputs/mountains.jpg'

W0 = 50
TEST_RANGE = 200
LR = 1e-2


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
    in_f = grid.shape[-1]

    model = SirenModel(coord_dim=in_f, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))

    optim = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    pred = model(grid)
    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/meta.jpg')

    for i in range(TEST_RANGE):
        model.train()
        optim.zero_grad()

        pred = model(grid)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i % 20 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
