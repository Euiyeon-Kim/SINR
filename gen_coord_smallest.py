import os
import numpy as np

import torch
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop

from models.siren import FourierReLU
from models.adversarial import Discriminator, MappingConv

from utils.loss import calcul_gp
from utils.utils import make_exp_dirs, prepare_siren_inp
from utils.grid import visualize_grid, shuffle_grid, cutout_grid
from utils.flow import warp

'''
    INR: fourier 256 encoding, Relu network로 학습
    flow_generator: input / 4 Resolution noise를 bilinear로 upsample하고 flow 생성 학습
'''

EXP_NAME = 'flow/debug'
PATH = 'inputs/small_balloons.png'

PTH_PATH = 'exps/flow/origin/ckpt/final.pth'
B_PATH = 'exps/flow/origin/ckpt/B.pt'
MAPPING_SIZE = 256

LR = 1e-4
MAX_ITERS = 2000
N_CRITIC = 5
GEN_ITER = 3
GP_LAMBDA = 10
NOISE_SCALE = 1


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/grid', exist_ok=True)
    writer = make_exp_dirs(EXP_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inr = FourierReLU(coord_dim=MAPPING_SIZE * 2, num_c=3, hidden_node=256, depth=5).to(device)
    inr.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH).to(device)

    # Read image - 제일 작은 스케일 이미지 원본
    real_img, origin_grid = prepare_siren_inp(PATH, device)
    h, w, _ = real_img.shape
    d_ref_img = real_img.permute(2, 0, 1)

    # Origin 크기로 INR 학습했을 때 제일 작은 scale uniform grid 그냥 넣으면 나오는 결과
    x_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * origin_grid), B)
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    visualize_grid(origin_grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    # Recon
    for param in inr.parameters():
        param.trainable = False
    recon = inr(mapped_input).permute(2, 0, 1).detach()
    viz_recon = (recon + 1.) / 2.  # (-1, 1) -> (0, 1)
    save_image(viz_recon, f'exps/{EXP_NAME}/recon.jpg')

    # Prepare model
    coord_generator = MappingConv(in_c=1, out_c=2).to(device)
    m_optim = torch.optim.Adam(coord_generator.parameters(), lr=LR, betas=(0.5, 0.999))
    d = Discriminator(in_c=3, nfc=32).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Generate fake image
            noise = torch.normal(mean=0, std=1.0, size=(1, h, w)).to(device) * NOISE_SCALE
            # gridxnoise = torch.einsum('chw,hwg->ghw', noise, origin_grid)
            fg_inp = torch.unsqueeze(noise, 0)     # torch.unsqueeze(torch.concat((noise, recon), 0), 0)
            generated_coord = coord_generator(fg_inp)[0].permute(1, 2, 0)
            generated_x_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * generated_coord), B)
            generated_mapped_input = torch.cat([torch.sin(generated_x_proj), torch.cos(generated_x_proj)], dim=-1)

            generated_img = inr(generated_mapped_input)
            generated_img = generated_img.permute(2, 0, 1)
            # fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_img), dim=0)
            fake_patch = torch.unsqueeze(generated_img, dim=0)

            # Real patch
            # real_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(origin_img), dim=0)
            real_patch = torch.unsqueeze(d_ref_img, dim=0)

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
            coord_generator.train()
            m_optim.zero_grad()

            # Train with fake image
            noise = torch.normal(mean=0, std=1.0, size=(1, h, w)).to(device) * NOISE_SCALE
            # gridxnoise = torch.einsum('chw,hwg->ghw', noise, origin_grid)
            fg_inp = torch.unsqueeze(noise, 0)  # torch.unsqueeze(torch.concat((noise, recon), 0), 0)
            generated_coord = coord_generator(fg_inp)[0].permute(1, 2, 0)
            generated_x_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * generated_coord), B)
            generated_mapped_input = torch.cat([torch.sin(generated_x_proj), torch.cos(generated_x_proj)], dim=-1)

            generated_img = inr(generated_mapped_input)
            generated_img = generated_img.permute(2, 0, 1)

            # fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_img), dim=0)
            fake_patch = torch.unsqueeze(generated_img, dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()
            # reg_loss = torch.mean(torch.abs(generated_flow))
            g_loss = adv_loss   # - reg_loss * REG_LAMBDA
            g_loss.backward()
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic - min", adv_loss.item(), iter)
        # writer.add_scalar("g/flow_reg", reg_loss.item(), iter)

        writer.flush()

        if (iter + 1) % 10 == 0:
            generated_img = (generated_img + 1.) / 2.   # (-1, 1) -> (0, 1)
            save_image(generated_img, f'exps/{EXP_NAME}/img/{iter}.jpg')
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/grid/{iter}.jpg', device)

    torch.save(coord_generator.state_dict(), f'exps/{EXP_NAME}/ckpt/final_G.pth')