import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop
from torch.utils.tensorboard import SummaryWriter

from utils.viz import visualize_grid
from utils.grid import create_grid
from utils.loss import calcul_gp
from models.siren import SirenModel
from models.adversarial import Discriminator, FMappingConv

'''
    INR weight fix
    H*W 크기의 Noise coordinate를 받아서 Conv mapper로 H*W*256 크기의 ff input 생성
    INR 통과시켜서 생성된 H*W 크기의 이미지를 patch size로 크롭해서 Discriminator input으로 사용
'''


EXP_NAME = 'ff_conv_w_balloons_32'
PATH = '../inputs/balloons.png'
PTH_PATH = '../exps/fourier_siren_balloons/ckpt/final.pth'
B_PATH = '../exps/fourier_siren_balloons/ckpt/B.pt'
MAX_ITERS = 1000000
LR = 1e-4

N_CRITIC = 5
GEN_ITER = 2
PATCH_SIZE = 32
GP_LAMBDA = 10.0
ADV_LAMBDA = 0.1

SCALE = 10
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/w/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/w/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/w/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/w/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).permute(2, 0, 1).to(device)
    grid = create_grid(h, w, device=device)
    visualize_grid(grid, f'exps/{EXP_NAME}/w/base_grid.jpg', device)

    model = SirenModel(coord_dim=2 * MAPPING_SIZE, num_c=3).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.trainable = False
    learned_B = torch.load(B_PATH)
    x_proj = (2. * np.pi * grid) @ learned_B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    recon = model(mapped_input).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    loss_fn = torch.nn.MSELoss()
    mapper = FMappingConv(in_c=1, out_c=MAPPING_SIZE * 2).to(device)

    m_optim = torch.optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))
    d = Discriminator(nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    origin_coord = torch.unsqueeze(grid.permute(2, 0, 1), dim=0)
    for iter in range(MAX_ITERS):
        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(img), dim=0)
            real_prob_out = d(real_patch)
            d_real_loss = -real_prob_out.mean()
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.normal(mean=0, std=1.0, size=(1, 1, h, w)).to(device)
            generated_coord = torch.squeeze(mapper(noise_coord)).permute(1, 2, 0)
            generated = model(generated_coord).permute(2, 0, 1).detach()
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated), dim=0)
            # fake_patch = torch.unsqueeze(generated, dim=0)

            fake_prob_out = d(fake_patch)
            d_fake_loss = fake_prob_out.mean()  # Minimize D(G(z))
            d_fake_loss.backward(retain_graph=True)

            gradient_penalty = calcul_gp(d, real_patch, fake_patch, device) * GP_LAMBDA
            gradient_penalty.backward()

            d_optim.step()

        # Log discriminator losses
        d_loss = d_real_loss + d_fake_loss + gradient_penalty
        critic = d_real_loss - d_fake_loss
        writer.add_scalar("d/total", d_loss.item(), iter)
        writer.add_scalar("d/critic", critic.item(), iter)
        writer.add_scalar("d/gp", gradient_penalty.item(), iter)

        for i in range(GEN_ITER):
            mapper.train()
            m_optim.zero_grad()

            noise_coord = torch.normal(mean=0, std=1.0, size=(1, 1, h, w)).to(device)
            generated_coord = torch.squeeze(mapper(noise_coord)).permute(1, 2, 0)
            generated = model(generated_coord).permute(2, 0, 1)
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated), dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            g_loss = adv_loss
            g_loss.backward(retain_graph=True)
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.flush()

        # Log image
        if (iter + 1) % 10 == 0:
            save_image(generated, f'exps/{EXP_NAME}/w/img/{iter}_all.jpg')
            save_image(fake_patch, f'exps/{EXP_NAME}/w/img/{iter}_patch.jpg')
            save_image(real_patch, f'exps/{EXP_NAME}/w/img/{iter}_real.jpg')
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/w/img/{iter}_whole.jpg', device, 'generated_coord_all')
