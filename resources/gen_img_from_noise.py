import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel

'''
    SIREN(noise) = RGB
    coord 대신에 noise 투입
'''


EXP_NAME = 'gen_img_from_noise'
PATH = '../inputs/balloons.png'
MAX_ITERS = 400
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
    # grid = create_grid(h, w, device=device)
    # in_f = grid.shape[-1]

    model = SirenModel(coord_dim=2, num_c=c).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        optim.zero_grad()

        grid = torch.normal(mean=0, std=1.0, size=(h, w, 2)).to(device)
        pred = model(grid)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 50 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')


    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')