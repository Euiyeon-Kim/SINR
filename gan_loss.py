import os

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import RandomCrop
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from utils.utils import shuffle_grid
from utils.viz import visualize_grid
from utils.loss import calcul_gp
from models.maml import SirenModel
from models.adversarial import Discriminator


EXP_NAME = 'balloons/learnit_var_patch_64/inr_origin/patchify_shuffle_coord_64'
PTH_PATH = 'exps/balloons/learnit_var_patch_64/inr_origin/ckpt/final.pth'
PATH = 'inputs/balloons.png'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

N_CRITIC = 5
GEN_ITER = 3
GP_LAMBDA = 10


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.float32(img) / 255
    h, w, _ = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.unsqueeze(torch.FloatTensor(img).permute(2, 0, 1).to(device), dim=0)
    origin_grid = shuffle_grid(h, w, device=device)
    grid = shuffle_grid(h, w, device=device)

    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.trainable = False
    model.eval()

    recon = model(grid)
    recon = recon.permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')
    visualize_grid(grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    find = grid.to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find}, lr=LR)

    d = Discriminator(in_c=3, nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):
        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_prob_out = d(img)
            d_real_loss = -real_prob_out.mean()
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            pred = torch.unsqueeze(model(find).permute(2, 0, 1), dim=0)
            fake_prob_out = d(pred)
            d_fake_loss = fake_prob_out.mean()  # Minimize D(G(z))
            d_fake_loss.backward(retain_graph=True)

            gradient_penalty = calcul_gp(d, img, pred, device) * GP_LAMBDA
            gradient_penalty.backward()

            d_optim.step()

        # Log discriminator losses
        d_loss = d_real_loss + d_fake_loss + gradient_penalty
        critic = d_real_loss - d_fake_loss
        writer.add_scalar("d/total", d_loss.item(), iter)
        writer.add_scalar("d/critic", critic.item(), iter)
        writer.add_scalar("d/gp", gradient_penalty.item(), iter)

        for i in range(GEN_ITER):
            optim.zero_grad()

            pred = torch.unsqueeze(model(find).permute(2, 0, 1), dim=0)
            fake_prob_out = d(pred)
            adv_loss = -fake_prob_out.mean()

            g_loss = adv_loss
            g_loss.backward(retain_graph=True)
            optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.flush()

        # Log image
        if (iter + 1) % 10 == 0:
            save_image(torch.abs(recon-pred), f'exps/{EXP_NAME}/img/{iter}_{torch.mean(torch.abs(recon-pred)):.4f}.jpg')
            save_image(pred, f'exps/{EXP_NAME}/img/{iter}_generated_img.jpg')
            visualize_grid(find, f'exps/{EXP_NAME}/img/{iter}_found.jpg', device)
            visualize_grid(find-origin_grid, f'exps/{EXP_NAME}/img/{iter}_diff_{torch.mean(torch.abs(find-origin_grid)):.4f}.jpg', device)
