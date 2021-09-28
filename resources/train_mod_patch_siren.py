import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from models.siren import ModulatedSirenModel
from models.encoder import Encoder
from utils.grid import create_grid

EXP_NAME = 'mod_bird'
PATCHES_ROOT = './inputs/birds_patch'
MAX_ITERS = 1000000
LR = 1e-4

LATENT_DIM = 256
PATCH_SIZE = 32


class Custom(Dataset):
    def __init__(self, data_root=PATCHES_ROOT, res=PATCH_SIZE):
        self.res = res
        self.paths = glob(f'{data_root}/*.jpg')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = np.float32(img) / 255
        img = np.transpose(img, (2, 0, 1))
        return img


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    dataset = Custom(data_root=PATCHES_ROOT)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(PATCH_SIZE, PATCH_SIZE, device=device)

    encoder = Encoder().to(device)
    model = ModulatedSirenModel(coord_dim=2, num_c=3, w0=30, latent_dim=LATENT_DIM).to(device)

    optim = torch.optim.Adam(list(model.parameters())+list(encoder.parameters()), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    i = 0
    while i < MAX_ITERS:
        for patch in dataloader:
            patch = patch.to(device)
            model.train()
            optim.zero_grad()

            z = torch.squeeze(encoder(patch))
            pred = torch.unsqueeze(model(z, grid).permute(2, 0, 1), dim=0)

            loss = loss_fn(pred, patch)

            loss.backward()
            optim.step()

            print(f'{i}|{MAX_ITERS}: {loss.item():.9f}')
            writer.add_scalar("loss", loss.item(), i)

            i += 1
            if i % 500 == 0:
                save_image(patch[0], f'exps/{EXP_NAME}/img/{i}_origin.jpg')
                save_image(pred[0], f'exps/{EXP_NAME}/img/{i}_pred.jpg')


    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')