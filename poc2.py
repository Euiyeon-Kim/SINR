import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.siren import FourierReLU
from utils.dataloader import PatchINR
from utils.utils import make_exp_dirs, create_grid, sample_B
EXP_NAME = 'poc/11'
PATH = 'inputs/balloons.png'
BATCH = 32
PATCH = 11

if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = torch.FloatTensor(np.array(Image.open(PATH).convert('RGB')) / 255.).to(device)
    h, w, _ = img.shape
    dataset = PatchINR(PATH, patch_size=PATCH)
    dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)

    grid = create_grid(PATCH, PATCH, device)
    B2 = torch.randn((16, 2)).to(device) * 10.
    x_proj = (2. * np.pi * grid) @ B2.t()
    mapped_grid = torch.sin(x_proj)

    B = torch.randn((128, 2)).to(device) * 10.
    model = FourierReLU(128, 5123, 512, 1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1000):
        for step, data in enumerate(dataloader):
            coord, patch = data
            coord, patch = coord.to(device), patch.to(device)

            x_proj = (2. * np.pi * coord) @ B.t()
            inp = torch.sin(x_proj)

            optim.zero_grad()
            pred_weights = model(inp)

            total_loss = 0
            for b in range(BATCH):
                pred_weight = pred_weights[b]
                splits = torch.split(pred_weight, 256, dim=-1)
                w1 = torch.stack((splits[:16]), dim=-1)
                b1 = splits[16:17][0]
                w2 = torch.stack((splits[17:20]), dim=1).T
                b2 = splits[-1]
                l1_output = F.relu(F.linear(mapped_grid, w1, b1))
                l2_output = torch.sigmoid(F.linear(l1_output, w2, b2))
                total_loss += loss_fn(patch[1], l2_output)

            total_loss /= BATCH
            total_loss.backward()
            optim.step()

            writer.add_scalar("loss", total_loss.item(), epoch*len(dataloader)+step)

            if step % 500 == 0:
                patch = patch[0].permute(2, 0, 1)
                pred = l2_output.permute(2, 0, 1)
                save_image(pred, f'exps/{EXP_NAME}/img/{epoch}_{step}.jpg')
                save_image(patch, f'exps/{EXP_NAME}/img/{epoch}_{step}_gt.jpg')

        torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')
        torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
