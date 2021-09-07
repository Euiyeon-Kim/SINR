import os

from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.maml import MAML
from utils.utils import create_grid


EXP_NAME = 'learnit_stone'
PATH = 'inputs/stone.png'
RES = 32

BATCH_SIZE = 1
INNER_STEPS = 2
MAX_ITERS = 150000

OUTER_LR = 1e-5
INNER_LR = 1e-2


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    w, h = img.size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(h, w, device=device)

    maml = MAML(coord_dim=2, num_c=3, w0=50).to(device)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = torch.nn.MSELoss()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0, 0.15), scale=(0.5, 1.25)),
        transforms.ToTensor()
    ])

    origin = Image.open(PATH).convert('RGB')

    # Outer loops
    outer_step = 0
    for i in range(MAX_ITERS):
        data = transform(origin).permute(1, 2, 0)
        data = torch.unsqueeze(data, dim=0).to(device)

        maml.train()
        pred = maml(grid, data, True)

        loss = loss_fn(pred, data)
        print(f'{outer_step}/{MAX_ITERS}: {loss.item()}')

        outer_optimizer.zero_grad()
        loss.backward()
        outer_optimizer.step()

        writer.add_scalar("loss", loss.item(), outer_step)
        outer_step += 1

        if outer_step % 10 == 0:
            pred = pred[0].permute(2, 0, 1)
            data = data[0].permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item()}_pred.jpg')
            save_image(data, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item()}_data.jpg')

        if outer_step % 100 == 0:
            torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/{outer_step}.pth')




