import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop
from torch.utils.tensorboard import SummaryWriter

from models.siren import ModulatedSirenModel
from models.encoder import VGGEncoder
from utils.utils import create_grid

EXP_NAME = 'mod_balloon'
PATH = './inputs/balloons.png'
MAX_ITERS = 10000
LR = 1e-4

LATENT_DIM = 256
PATCH_SIZE = 64


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device).permute(2, 0, 1)
    grid = create_grid(PATCH_SIZE, PATCH_SIZE, device=device)
    in_f = grid.shape[-1]

    encoder = VGGEncoder().to(device)
    model = ModulatedSirenModel(coord_dim=in_f, num_c=c, latent_dim=LATENT_DIM).to(device)

    optim = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': encoder.parameters(), 'lr': 1e-5}
            ], lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        encoder.train()
        model.train()

        patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(img), dim=0)

        optim.zero_grad()

        z = torch.squeeze(encoder(patch))
        pred = torch.unsqueeze(model(z, grid).permute(2, 0, 1), dim=0)

        loss = loss_fn(pred, patch)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 50 == 0:
            save_image(patch[0], f'exps/{EXP_NAME}/img/{i}_origin.jpg')
            save_image(pred[0], f'exps/{EXP_NAME}/img/{i}_pred.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')