import os

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.maml import MAML
from utils.dataloader import Custom
from utils.grid import create_grid


EXP_NAME = 'balloon/learnit_fixed_32'
DATA_ROOT = 'inputs/ballons_patch_32'

W0 = 50
PATCH_SIZE = 32

BATCH_SIZE = 1
INNER_STEPS = 2
MAX_ITERS = 20000

OUTER_LR = 1e-5
INNER_LR = 1e-2


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    dataset = Custom(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(PATCH_SIZE, PATCH_SIZE, device=device)

    maml = MAML(coord_dim=2, num_c=3, w0=W0).to(device)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = torch.nn.MSELoss()

    # Outer loops
    outer_step = 0
    while outer_step < MAX_ITERS:
        for data in dataloader:
            if outer_step > MAX_ITERS:
                break

            data = data.to(device)

            maml.train()
            pred = maml(grid, data, True)

            loss = loss_fn(pred, data)
            print(f'{outer_step}/{MAX_ITERS}: {loss.item()}')

            outer_optimizer.zero_grad()
            loss.backward()
            outer_optimizer.step()

            writer.add_scalar("loss", loss.item(), outer_step)
            outer_step += 1

            if (outer_step + 1) % 100 == 0:
                pred = pred[0].permute(2, 0, 1)
                data = data[0].permute(2, 0, 1)
                save_image(pred, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_pred.jpg')
                save_image(data, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_data.jpg')
                meta = maml.model(grid).permute(2, 0, 1)
                save_image(meta, f'exps/{EXP_NAME}/img/{outer_step}_meta.jpg')

            if (outer_step + 1) % 1000 == 0:
                torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/{outer_step}.pth')

