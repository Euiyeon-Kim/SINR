import os

import torch
from torchvision.utils import save_image

from models.siren import FourierReLU
from models.adversarial import Discriminator
from models.positional_encofing import SinusoidalPositionalEmbedding, GeneratorBlock

from utils.loss import calcul_gp
from utils.utils import make_exp_dirs, prepare_siren_inp
from utils.grid import visualize_grid


EXP_NAME = 'gen_coord_with_PE/small_gen_grid_0'
PATH = 'inputs/small_balloons.png'

PTH_PATH = 'exps/flow/origin/ckpt/final.pth'
B_PATH = 'exps/flow/origin/ckpt/B.pt'
MAPPING_SIZE = 256
NF = 32

LR = 1e-4
MAX_ITERS = 2500
N_CRITIC = 5
GEN_ITER = 3
GP_LAMBDA = 10

EMBEDDING_DIM = 4
NUM_EMBEDDING = 512


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/grid', exist_ok=True)
    writer = make_exp_dirs(EXP_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inr = FourierReLU(coord_dim=MAPPING_SIZE * 2, num_c=3, hidden_node=256, depth=5).to(device)
    inr.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH).to(device)

    # Read image - 이미지 원본
    real_img, origin_grid = prepare_siren_inp(PATH, device)
    h, w, _ = real_img.shape
    d_ref_img = real_img.permute(2, 0, 1)

    # Origin 크기로 INR 학습했을 때 uniform grid 그냥 넣으면 나오는 결과
    x_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * origin_grid), B)
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    visualize_grid(origin_grid, f'exps/{EXP_NAME}/base_grid.jpg', device)

    # Recon
    for param in inr.parameters():
        param.trainable = False
    recon = inr(mapped_input).permute(2, 0, 1).detach()
    viz_recon = (recon + 1.) / 2.  # (-1, 1) -> (0, 1)
    save_image(viz_recon, f'exps/{EXP_NAME}/recon.jpg')

    # Make positional embedding
    head_PE = SinusoidalPositionalEmbedding(EMBEDDING_DIM, 0, NUM_EMBEDDING, False, 200).to(device)
    fixed_noise = torch.randn(1, 1, h, w).to(device)
    pe = head_PE(fixed_noise)

    # Prepare model
    coord_generator = GeneratorBlock(in_channels=EMBEDDING_DIM*2, out_channels=2, base_channels=NF).to(device)
    m_optim = torch.optim.Adam(coord_generator.parameters(), lr=LR, betas=(0.5, 0.999))
    d = Discriminator(in_c=3, nfc=NF).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Generate fake coord
            pe_noise = torch.randn(1, 1, h, w).to(device) + pe
            generated_coord = coord_generator(pe_noise)[0].permute(1, 2, 0)
            gen_c_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * generated_coord), B)
            gen_c_mapped = torch.cat([torch.sin(gen_c_proj), torch.cos(gen_c_proj)], dim=-1)

            generated_img = inr(gen_c_mapped)
            generated_img = generated_img.permute(2, 0, 1)
            fake_patch = torch.unsqueeze(generated_img, dim=0)

            # Real patch
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
            pe_noise = torch.randn(1, 1, h, w).to(device) + pe
            generated_coord = coord_generator(pe_noise)[0].permute(1, 2, 0)
            gen_c_proj = torch.einsum('hwc,fc->hwf', (2. * torch.pi * generated_coord), B)
            gen_c_mapped = torch.cat([torch.sin(gen_c_proj), torch.cos(gen_c_proj)], dim=-1)

            generated_img = inr(gen_c_mapped)
            generated_img = generated_img.permute(2, 0, 1)
            fake_patch = torch.unsqueeze(generated_img, dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            g_loss = adv_loss
            g_loss.backward()
            m_optim.step()

        # Log generator losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic - min", adv_loss.item(), iter)
        writer.flush()

        if (iter + 1) % 100 == 0:
            generated_img = (generated_img + 1.) / 2.   # (-1, 1) -> (0, 1)
            save_image(generated_img, f'exps/{EXP_NAME}/img/{iter}.jpg')
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/grid/{iter}.jpg', device)

    torch.save(coord_generator.state_dict(), f'exps/{EXP_NAME}/ckpt/G.pth')
    torch.save(d.state_dict(), f'exps/{EXP_NAME}/ckpt/D.pth')
    torch.save(pe, f'exps/{EXP_NAME}/ckpt/pe.pt')
