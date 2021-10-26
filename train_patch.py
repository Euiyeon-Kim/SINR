import os

import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.siren import SirenModel
from utils.utils import sample_B
from utils.dataloader import PatchINR, PatchINRVal
from utils.fourier import get_fourier, viz_fourier

W0 = 50

EXP_NAME = f'balloon_patch'
PATH = './inputs/balloons.png'

EPOCH = 50
LR = 1e-4

SCALE = 10
MAPPING_SIZE = 256
PATCH_SIZE = 5
BATCH_SIZE = 512

if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    dataset = PatchINR(PATH, patch_size=PATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = sample_B(MAPPING_SIZE, SCALE, device)
    model = SirenModel(coord_dim=MAPPING_SIZE * 2, num_c=PATCH_SIZE * PATCH_SIZE * 3, w0=W0).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    step_per_epoch = len(dataloader)
    for i in range(EPOCH):

        for step, data in enumerate(dataloader):
            coord, patch = data
            coord, patch = coord.to(device), patch.to(device)
            x_proj = (2. * np.pi * coord) @ B.t()
            mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

            model.train()
            optim.zero_grad()

            pred = model(mapped_input)
            loss = loss_fn(pred, patch)

            loss.backward()
            optim.step()
            print(f'{i}/{EPOCH} | {step}/{step_per_epoch}  loss: {loss.item()}')
            # writer.add_scalar("loss", loss.item(), step_per_epoch*i + step)

    model.eval()
    img = np.array(Image.open(PATH).convert('RGB'))
    count = np.zeros_like(img).astype(np.float32)
    results = np.zeros_like(img).astype(np.float32)
    adder = np.ones((PATCH_SIZE, PATCH_SIZE, 3)).astype(np.float32)

    h, w, c = img.shape
    ph, pw = h - PATCH_SIZE + 1, w - PATCH_SIZE + 1
    num_inf = ph * pw
    for idx in range(num_inf):
        c_h = (idx // pw) + (PATCH_SIZE // 2)
        c_w = (idx % pw) + (PATCH_SIZE // 2)
        coord = torch.FloatTensor([(c_h - (PATCH_SIZE // 2)) / ph, (c_w - (PATCH_SIZE // 2)) / pw]).to(device)
        x_proj = (2. * np.pi * coord) @ B.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        output = torch.reshape(model(mapped_input).detach(), (PATCH_SIZE, PATCH_SIZE, 3)).cpu().numpy()
        results[c_h - (PATCH_SIZE // 2): c_h + (PATCH_SIZE // 2 + 1), c_w - (PATCH_SIZE // 2): c_w + (PATCH_SIZE // 2 + 1),:] += output
        count[c_h - (PATCH_SIZE // 2): c_h + (PATCH_SIZE // 2 + 1), c_w - (PATCH_SIZE // 2): c_w + (PATCH_SIZE // 2 + 1),:] += adder

    pred = (results / count) * 255.
    import cv2
    print(np.max(pred), np.min(pred))
    cv2.imwrite('tmp.jpg', cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
    torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')