import os

import numpy as np

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.dataloader import PatchINR
from utils.utils import sample_B, create_grid
from utils.fourier import get_fourier, viz_fourier

W0 = 50

EXP_NAME = f'balloon_patch'
PATH = './inputs/balloons.png'

EPOCH = 100
LR = 1e-4

SCALE = 10
MAPPING_SIZE = 256
PATCH_SIZE = 5
BATCH_SIZE = 64

if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    dataset = PatchINR(PATH, patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = sample_B(MAPPING_SIZE, SCALE, device)
    model = SirenModel(coord_dim=MAPPING_SIZE * 2, num_c=PATCH_SIZE * PATCH_SIZE * 3, w0=W0).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    step_per_epoch = len(dataloader)
    for i in range(EPOCH):

        for step, data in enumerate(dataloader):
            coord, patch = data
            coord, patch = coord.to(device), patch.to(device)
            x_proj = (2. * np.pi * coord) @ B.t()
            mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

            model.train()
            optim.zero_grad()

            pred = model(mapped_input)
            loss = loss_fn(pred, patch)

            loss.backward()
            optim.step()

            writer.add_scalar("loss", loss.item(), step_per_epoch*i + step)


        model.eval()
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