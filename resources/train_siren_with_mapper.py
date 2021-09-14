import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from models.adversarial import MappingConv
from utils.viz import visualize_grid

EXP_NAME = 'SIREN(MappingConv(z))'
PATH = '../inputs/balloons.png'

W0 = 50
MAX_ITERS = 100000
LR = 1e-4


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)

    mapper = MappingConv().to(device)
    model = SirenModel(coord_dim=2, num_c=3).to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(mapper.parameters()), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        optim.zero_grad()

        noise_coord = torch.normal(mean=0, std=1.0, size=(1, 2, h // 4, w // 4)).to(device)
        upsampled_coord = torch.nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(noise_coord)
        generated_coord = torch.squeeze(mapper(upsampled_coord)).permute(1, 2, 0)

        pred = model(generated_coord)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i == 0 or (i+1) % 100 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')
            visualize_grid(torch.squeeze(noise_coord).permute(1, 2, 0), f'exps/{EXP_NAME}/img/{i}_noise.jpg', device)
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/img/{i}_grid.jpg', device)

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')