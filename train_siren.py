import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.grid import create_grid
from utils.fourier import get_fourier, viz_fourier

EXP_NAME = 'stripe'
PATH = './inputs/stripe.jpg'

W0 = 50
MAX_ITERS = 2000
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
    grid = create_grid(h, w, device=device)
    in_f = grid.shape[-1]

    model = SirenModel(coord_dim=in_f, num_c=c, w0=W0).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        optim.zero_grad()

        pred = model(grid)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 100 == 0:
            pred_img = (pred * 255.).detach().cpu().numpy()
            fourier_info = get_fourier(pred_img)
            inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/img/', prefix=f'{i}_')

            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')


    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')