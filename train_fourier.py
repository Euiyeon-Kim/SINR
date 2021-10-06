import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import prepare_fourier_inp
from utils.fourier import get_fourier, viz_fourier

EXP_NAME = 'stripe_fourier_log'
PATH = './inputs/stripe.jpg'

W0 = 50
MAX_ITERS = 1000
LR = 1e-4

SCALE = 10
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img, B, mapped_input = prepare_fourier_inp(PATH, device, mapping_size=MAPPING_SIZE, scale=SCALE)
    model = SirenModel(coord_dim=MAPPING_SIZE * 2, num_c=3, w0=W0).to(device)

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

        if (i+1) % 1 == 0:
            pred_img = (pred * 255.).detach().cpu().numpy()
            fourier_info = get_fourier(pred_img)
            inr_viz_dict = viz_fourier(fourier_info, dir=None)

            import cv2
            cv2.imwrite(f'exps/{EXP_NAME}/img/{i}_mag.jpg', inr_viz_dict['gray']['mag'])
            cv2.imwrite(f'exps/{EXP_NAME}/img/{i}_phase.jpg', inr_viz_dict['gray']['phase'])

            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
    torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')