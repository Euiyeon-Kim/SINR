import os

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import prepare_siren_inp
from utils.fourier import get_fourier, viz_fourier

W0 = 200

EXP_NAME = f'stripe_log/256_5_{W0}'
PATH = './inputs/stripe.jpg'

MAX_ITERS = 1000
LR = 1e-4

if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img, grid = prepare_siren_inp(PATH, device)

    model = SirenModel(coord_dim=2, num_c=3, w0=W0, depth=5).to(device)
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

        if i % 10 == 0:
            pred_img = (pred * 255.).detach().cpu().numpy()
            fourier_info = get_fourier(pred_img)
            inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/img/', prefix=f'{i}_')

            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')