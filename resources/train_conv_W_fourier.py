import os
import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


from utils.viz import visualize_grid
from utils.grid import create_grid
from utils.loss import calcul_gp
from models.siren import SirenModel
from models.adversarial import Discriminator, MappingConv

'''
    Random coord -> W(Conv) -> Generated coord
    Generated coord -> Model -> Generated image 
    Generated image -> Crop patch -> Discriminator

    Result
    W(MLP)랑 마찬가지, W가 conv라고 될 일이 아닌 듯
'''

EXP_NAME = 'balloons_fourier/learnit_var_patch_64_fourier/gan_coord_3'
B_PATH = '../exps/balloons/learnit_var_patch_64_fourier/ckpt/B.pth'
PTH_PATH = '../exps/balloons_fourier/learnit_var_patch_64_fourier/inr_origin/ckpt/final.pth'

PATH = '../inputs/balloons_multiscale/3.png'
# PATH = 'inputs/balloons.png'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

D_PAD = 5
N_CRITIC = 5
GEN_ITER = 2
PATCH_SIZE = 16
GP_LAMBDA = 10.0
RECON_LAMBDA = 10.0

NOISE_SAMPLE = 8
SCALE = 10
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).permute(2, 0, 1).to(device)
    grid = create_grid(h, w, device=device)
    visualize_grid(grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    B_gauss = torch.load(B_PATH)
    x_proj = (2. * np.pi * grid) @ B_gauss.t()
    mapped_origin_grid = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    model = SirenModel(coord_dim=MAPPING_SIZE * 2, num_c=c, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.trainable = False

    recon = model(mapped_origin_grid).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    mapper = MappingConv(in_c=2, out_c=2).to(device)
    m_optim = torch.optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))

    d = Discriminator(in_c=2*MAPPING_SIZE, nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))
    loss_fn = torch.nn.MSELoss()
    pad_fn = torch.nn.ZeroPad2d(D_PAD)

    real_grid = torch.unsqueeze(torch.nn.ZeroPad2d(D_PAD)(mapped_origin_grid.permute(2, 0, 1)), dim=0)
    for iter in range(MAX_ITERS):
        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_prob_out = d(real_grid)

            d_real_loss = -real_prob_out.mean()
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.randn((1, 1, h // NOISE_SAMPLE, w // NOISE_SAMPLE)).expand(1, 2, h // NOISE_SAMPLE, w // NOISE_SAMPLE).to(device)
            upsampled_coord = pad_fn(torch.nn.Upsample((h, w), mode='bilinear', align_corners=True)(noise_coord))
            generated_coord = torch.squeeze(mapper(upsampled_coord)).permute(1, 2, 0)

            x_proj = (2. * np.pi * generated_coord) @ B_gauss.t()
            mapped_generated_coord = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            fake_coord = torch.unsqueeze(torch.nn.ZeroPad2d(D_PAD)(mapped_generated_coord.permute(2, 0, 1)), dim=0)

            fake_prob_out = d(fake_coord)
            d_fake_loss = fake_prob_out.mean()  # Minimize D(G(z))
            d_fake_loss.backward(retain_graph=True)

            gradient_penalty = calcul_gp(d, real_grid, fake_coord, device) * GP_LAMBDA
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

            noise_coord = torch.randn((1, 1, h // NOISE_SAMPLE, w // NOISE_SAMPLE)).expand(1, 2, h // NOISE_SAMPLE, w // NOISE_SAMPLE).to(device)
            upsampled_coord = pad_fn(torch.nn.Upsample((h, w), mode='bilinear', align_corners=True)(noise_coord))
            generated_coord = torch.squeeze(mapper(upsampled_coord)).permute(1, 2, 0)
            x_proj = (2. * np.pi * generated_coord) @ B_gauss.t()
            mapped_generated_coord = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            fake_coord = torch.unsqueeze(torch.nn.ZeroPad2d(D_PAD)(mapped_generated_coord.permute(2, 0, 1)), dim=0)

            recon_noise = torch.randn((1, 1, h // NOISE_SAMPLE, w // NOISE_SAMPLE)).expand(1, 2, h // NOISE_SAMPLE, w // NOISE_SAMPLE).to(device)
            upsampled_recon_noise = pad_fn(torch.nn.Upsample((h, w), mode='bilinear', align_corners=True)(recon_noise))
            recon_coord = torch.squeeze(mapper(upsampled_recon_noise)).permute(1, 2, 0)
            x_proj = (2. * np.pi * recon_coord) @ B_gauss.t()
            mapped_recon_coord = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

            recon = model(mapped_recon_coord).permute(2, 0, 1)
            recon_loss = loss_fn(recon, img)

            fake_prob_out = d(fake_coord)
            adv_loss = -fake_prob_out.mean()

            g_loss = adv_loss + RECON_LAMBDA * recon_loss
            g_loss.backward(retain_graph=True)
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.add_scalar("g/recon", recon_loss.item(), iter)
        writer.flush()

        # Log image
        if iter == 0 or (iter + 1) % 100 == 0:
            generated = torch.unsqueeze(model(mapped_generated_coord).permute(2, 0, 1).detach(), dim=0)
            save_image(recon, f'exps/{EXP_NAME}/img/{iter}_recon.jpg')
            save_image(generated, f'exps/{EXP_NAME}/img/{iter}_fake.jpg')
            visualize_grid(recon_coord, f'exps/{EXP_NAME}/img/{iter}_recon_grid.jpg', device)
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/img/{iter}_fake_grid.jpg', device)
