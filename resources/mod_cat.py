import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.grid import create_grid
from models.siren import ModulatedSirenModel
from models.encoder import Encoder

EXP_NAME = 'cat_mod'
EPOCH = 1000
PATCH = 32
LR = 1e-4


class Custom(Dataset):
    def __init__(self, data_root='./inputs/cat', res=PATCH):
        self.res = res
        self.paths = glob(f'{data_root}/*.jpg')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img = img.resize((self.res, self.res), Image.BICUBIC)
        img = np.float32(img) / 255
        img = np.transpose(img, (2, 0, 1))
        return img


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    dataset = Custom()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(PATCH, PATCH, device=device)

    encoder = Encoder().to(device)
    model = ModulatedSirenModel(coord_dim=2, w0=30, num_c=3, latent_dim=256).to(device)

    optim = torch.optim.Adam(list(encoder.parameters())+list(model.parameters()), lr=LR)
    loss_fn = torch.nn.MSELoss()

    cnt = 0
    for e in range(EPOCH):
        for idx, data in enumerate(dataloader):
            img = data.to(device)

            optim.zero_grad()
            z = torch.squeeze(encoder(img))
            pred = torch.unsqueeze(model(z, grid).permute(2, 0, 1), dim=0)
            loss = loss_fn(pred, img)
            loss.backward()
            optim.step()

            print(f'{e}|{EPOCH}  {idx}|{len(dataloader)} : {loss.item():.9f}')
            writer.add_scalar("loss", loss.item(), cnt)
            cnt += 1

            if cnt % 50 == 0:
                save_image(img[0], f'exps/{EXP_NAME}/img/{cnt}_origin.jpg')
                save_image(pred[0], f'exps/{EXP_NAME}/img/{cnt}_pred.jpg')

            if cnt % 2000 == 0:
                torch.save(encoder.state_dict(), f'exps/{EXP_NAME}/ckpt/{cnt}_encoder.pth')
                torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/{cnt}_model.pth')
