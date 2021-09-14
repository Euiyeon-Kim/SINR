import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision.transforms import RandomResizedCrop
from torch.utils.tensorboard import SummaryWriter

from models.maml import SirenModel
from models.encoder import RGB2CoordConv
from utils.utils import create_grid
from utils.viz import visualize_grid


EXP_NAME = 'balloons/learnit_var_patch_64/conv_reverser'
PTH_PATH = 'exps/balloons/learnit_var_patch_64/inr_origin/ckpt/final.pth'
PATH = 'inputs/balloons.png'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4
RGB_LAMBDA = 2


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.float32(img) / 255
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)
    grid = create_grid(h, w, device=device)

    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.trainable = False
    model.eval()

    pred = model(grid)
    pred = pred.permute(2, 0, 1)
    save_image(pred, f'exps/{EXP_NAME}/recon.jpg')
    visualize_grid(grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    mapper = RGB2CoordConv(in_c=3, out_c=2, nfc=64, num_layers=5).to(device)
    optim = torch.optim.Adam(mapper.parameters(), lr=LR)
    coord_loss_fn = torch.nn.MSELoss()
    rgb_loss_fn = torch.nn.L1Loss()

    for i in range(MAX_ITERS):
        model.eval()
        optim.zero_grad()

        # sampled_grid = RandomResizedCrop(64)(grid.permute(2, 0, 1)).permute(1, 2, 0)
        # generated = torch.unsqueeze(model(sampled_grid).permute(2, 0, 1), dim=0)
        # pred_coord = torch.squeeze(mapper(generated)).permute(1, 2, 0)
        pred_coord = torch.squeeze(mapper(torch.unsqueeze(img.permute(2, 0, 1), dim=0))).permute(1, 2, 0)

        pred = model(pred_coord)

        # rgb_loss = rgb_loss_fn(torch.squeeze(generated).permute(1, 2, 0), pred)
        # coord_loss = coord_loss_fn(pred_coord, sampled_grid)
        rgb_loss = rgb_loss_fn(pred, img)
        coord_loss = coord_loss_fn(pred_coord, grid)

        loss = rgb_loss * RGB_LAMBDA + coord_loss

        loss.backward()
        optim.step()

        writer.add_scalar("rgb_loss", rgb_loss.item(), i)
        writer.add_scalar("coord_loss", coord_loss.item(), i)
        writer.add_scalar("loss", loss.item(), i)

        if i == 0 or (i+1) % 500 == 0:
            pred = pred.permute(2, 0, 1)
            # visualize_grid(sampled_grid, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_ans.jpg', device)
            visualize_grid(pred_coord, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_pred_grid.jpg', device)
            # save_image(generated, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_input.jpg')
            save_image(pred, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_pred_img.jpg')
