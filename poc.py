import os

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import RandomCrop, Resize
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from models.maml import SirenModel
from utils.utils import create_grid
from utils.viz import visualize_grid

'''
    Patch에 대한 learnit 후 전체 이미지에 대한 INR을 학습했을 때
    MSE loss로 좌표가 찾아지는지를 test
    --> 초기값으로 주는 좌표가 만드는 이미지 픽셀 값이 중요함
    --> 초기값으로 주는 좌표에서 크게 달라지지 않음
'''


EXP_NAME = 'balloons/learnit_var_patch_64/mse_find_coord_from_0'
PTH_PATH = 'exps/balloons/learnit_var_patch_64/ckpt/19999.pth'
PATH = 'inputs/balloons.png'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4


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

    # find = grid.to(device).detach().requires_grad_(True)
    find = torch.zeros((h, w, 2)).to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find}, lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.eval()
        optim.zero_grad()

        pred = model(find)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i == 0 or (i+1) % 500 == 0:
            pred = pred.permute(2, 0, 1)
            visualize_grid(find, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_grid.jpg', device)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}_{loss.item():.4f}_find.jpg')