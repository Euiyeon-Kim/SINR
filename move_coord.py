import os
import numpy as np

import torch
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop

from models.siren import FourierReLU
from models.adversarial import Discriminator, MappingConv

from utils.loss import calcul_gp
from utils.utils import make_exp_dirs, prepare_siren_inp
from utils.grid import visualize_grid
from utils.flow import warp

'''
    INR: fourier 256 encoding, Relu network로 학습
    flow_generator: input / 4 Resolution noise를 bilinear로 upsample하고 flow 생성 학습
'''

EXP_NAME = 'flow/move_debug'
PATH = 'inputs/balloons.png'

PTH_PATH = 'exps/flow/origin/ckpt/final.pth'
B_PATH = 'exps/flow/origin/ckpt/B.pt'
MAPPING_SIZE = 256

LR = 1e-4
MAX_ITERS = 1000000
N_CRITIC = 5
GEN_ITER = 3
GP_LAMBDA = 10
PATCH_SIZE = 64
SCALE_FLOW = 1000.

if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/grid', exist_ok=True)
    writer = make_exp_dirs(EXP_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inr = FourierReLU(coord_dim=MAPPING_SIZE * 2, num_c=3, hidden_node=256, depth=5).to(device)
    inr.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH).to(device)

    # Read image
    img, origin_grid = prepare_siren_inp(PATH, device)
    h, w, _ = img.shape
    origin_img = img.permute(2, 0, 1)

    # Prepare grid
    visualize_grid(origin_grid, f'exps/{EXP_NAME}/base_grid.jpg', device)
    x_proj = (2. * np.pi * origin_grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    # Recon
    for param in inr.parameters():
        param.trainable = False
    recon = inr(mapped_input).permute(2, 0, 1)
    recon = (recon + 1.) / 2.  # (-1, 1) -> (0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    # Prepare model
    flow_generator = MappingConv(in_c=4, out_c=2).to(device)
    m_optim = torch.optim.Adam(flow_generator.parameters(), lr=LR, betas=(0.5, 0.999))
    d = Discriminator(in_c=3, nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Generate fake image
            noise = torch.normal(mean=0, std=1.0, size=(1, h, w)).to(device) * 100.
            fg_inp = torch.unsqueeze(torch.concat((noise, origin_img), 0), 0)
            generated_flow = flow_generator(fg_inp)[0]

            generated_mapped_inp, moved_real_grid = warp(mapped_input, origin_grid, generated_flow, device)
            generated_img = inr(generated_mapped_inp)
            generated_img = generated_img.permute(2, 0, 1)
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_img), dim=0)

            # Real patch
            real_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(origin_img), dim=0)

            d_real = d(real_patch)
            loss_r = -d_real.mean()

            d_generated = d(fake_patch)
            loss_f = d_generated.mean()

            gradient_penalty = calcul_gp(d, real_patch, fake_patch, device) * GP_LAMBDA
            d_loss = loss_r + loss_f + gradient_penalty

            d_loss.backward()
            d_optim.step()

        # Log discriminator losses
        critic = - loss_r - loss_f
        writer.add_scalar("d/total", d_loss.item(), iter)
        writer.add_scalar("d/critic - max", critic.item(), iter)
        writer.add_scalar("d/gp", gradient_penalty.item(), iter)

        for i in range(GEN_ITER):
            flow_generator.train()
            m_optim.zero_grad()

            # Train with fake image
            noise = torch.normal(mean=0, std=1.0, size=(1, h, w)).to(device) * 100.
            fg_inp = torch.unsqueeze(torch.concat((noise, origin_img), 0), 0)
            generated_flow = flow_generator(fg_inp)[0]

            generated_mapped_inp, moved_real_grid = warp(mapped_input, origin_grid, generated_flow, device)
            generated_img = inr(generated_mapped_inp)
            generated_img = generated_img.permute(2, 0, 1)
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_img), dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            adv_loss.backward()
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", adv_loss.item(), iter)
        writer.add_scalar("g/critic - min", adv_loss.item(), iter)
        writer.flush()

        if (iter + 1) % 10 == 0:
            generated_img = (generated_img + 1.) / 2.   # (-1, 1) -> (0, 1)
            save_image(generated_img, f'exps/{EXP_NAME}/img/{iter}.jpg')
            viz_moved_real_grid = moved_real_grid.squeeze().permute(1, 2, 0)
            visualize_grid(viz_moved_real_grid, f'exps/{EXP_NAME}/grid/{iter}.jpg', device)