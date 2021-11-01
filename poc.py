import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.siren import FourierReLU
from utils.dataloader import Custom
from utils.utils import make_exp_dirs, create_grid, prepare_fourier_inp

EXP_NAME = 'tmp'
PATH = 'inputs/balloons.png'

if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img, B, inp = prepare_fourier_inp(PATH, device)
    # img = torch.FloatTensor(np.array(Image.open(PATH).convert('RGB')) / 255.).to(device)
    # h, w, _ = img.shape
    # grid = create_grid(h, w, device)
    # B = torch.load('exps/poc/origin/ckpt/B.pt', map_location=device)
    # inp_B = torch.zeros_like(B)
    # x_proj = (2. * np.pi * grid) @ B.t()
    # inp = torch.sin(x_proj).to(device)

    model = FourierReLU(256, 3, 256, 1).to(device)
    # model.load_state_dict(torch.load(f'exps/poc/origin/ckpt/final.pth'))
    # model.eval()
    # B = torch.randn((256, 2)) * 10.
    # B = B.to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    for i in range(1000):
        optim.zero_grad()

        # x_proj = (2. * np.pi * grid) @ B.t()
        # inp = torch.sin(x_proj)

        pred = model(inp)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i % 10 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')

    torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')
    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
