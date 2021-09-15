import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomResizedCrop, InterpolationMode

from models.maml_with_mod import MAML
from utils.utils import create_grid

'''


'''

EXP_NAME = 'balloons/learnit_var_patch_with_mod_64'
PATH = '../inputs/balloons.png'

W0 = 50
PATCH_SIZE = 64

BATCH_SIZE = 1
INNER_STEPS = 2
MAX_ITERS = 20000

HIDDEN_NODE = 256
LATENT_DIM = 16
OUTER_LR = 1e-5
INNER_LR = 1e-2


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    img = torch.unsqueeze(torch.FloatTensor(img).permute(2, 0, 1).to(device), dim=0)
    grid = create_grid(PATCH_SIZE, PATCH_SIZE, device=device)

    maml = MAML(coord_dim=2, num_c=3, w0=W0, inner_steps=INNER_STEPS, inner_lr=INNER_LR,
                hidden_node=HIDDEN_NODE, latent_dim=LATENT_DIM, device=device).to(device)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = torch.nn.MSELoss()

    # Outer loops
    for outer_step in range(MAX_ITERS):
        data = RandomResizedCrop(PATCH_SIZE, interpolation=InterpolationMode.BICUBIC)(img)
        data = data.permute(0, 2, 3, 1).to(device)

        maml.train()
        z = torch.normal(mean=0.0, std=1.0, size=(BATCH_SIZE, LATENT_DIM)).to(device)

        pred = maml(z, grid, data, True)

        loss = loss_fn(pred, data)
        print(f'{outer_step}/{MAX_ITERS}: {loss.item()}')

        outer_optimizer.zero_grad()
        loss.backward()
        outer_optimizer.step()

        writer.add_scalar("loss", loss.item(), outer_step)

        if (outer_step + 1) % 100 == 0:
            pred = pred[0].permute(2, 0, 1)
            data = data[0].permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_pred.jpg')
            save_image(data, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_data.jpg')
            z = torch.normal(mean=0.0, std=1.0, size=(1, LATENT_DIM)).to(device)
            meta = maml.model(z[0], grid).permute(2, 0, 1)
            save_image(meta, f'exps/{EXP_NAME}/img/{outer_step}_meta.jpg')

        if (outer_step + 1) % 1000 == 0:
            torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/{outer_step}.pth')