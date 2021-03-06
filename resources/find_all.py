import os

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel


EXP_NAME = 'balloons_fourier/learnit_with_transform'
PATH = '89.png'
PTH_NAME = '14999'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/find_all', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_all/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_all/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/find_all/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/find_all/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    model.eval()

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    img = torch.FloatTensor(img).to(device)

    find = torch.randn((h, w, 512)).to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find}, lr=LR)
    loss_fn = nn.MSELoss()

    for i in range(MAX_ITERS):
        model.eval()

        pred = model(find)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        print(loss.item())
        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 100 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/find_all/img/{i}.jpg')
