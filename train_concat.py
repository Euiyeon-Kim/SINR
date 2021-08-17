import os
import numpy as np
from PIL import Image

import torch
from torchsummary import summary
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.utils import create_grid
from model import SirenModel, MappingNet

EXP_NAME = 'concat'
PATH = 'stone.png'
MAX_ITERS = 1000
LR = 1e-4
LATENT_DIM = 32


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rec_latent = torch.FloatTensor(np.zeros(LATENT_DIM)).to(device)
    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)
    rec_latents = rec_latent.repeat(h, w, 1)
    inp = torch.cat((grid, rec_latents), dim=-1)

    in_f = grid.shape[-1]

    model = SirenModel(coord_dim=in_f+LATENT_DIM, num_c=c).to(device)
    summary(model, (h, w, c+LATENT_DIM))

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        optim.zero_grad()

        pred = model(inp)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 50 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')

