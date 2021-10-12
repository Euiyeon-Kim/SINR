import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomCrop

from utils.grid import create_grid, visualize_grid
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

EXP_NAME = 'balloons/learnit_var_patch_64/conv_w_32'
PATH = '../inputs/balloons.png'
PTH_PATH = '../exps/balloons/learnit_var_patch_64/inr_origin/ckpt/final.pth'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

N_CRITIC = 5
GEN_ITER = 3
PATCH_SIZE = 32
GP_LAMBDA = 10


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
    origin_grid = grid.permute(2, 0, 1)

    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.requires_grad = False
    recon = model(grid).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    mapper = MappingConv(in_c=2, out_c=2).to(device)
    m_optim = torch.optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))

    d = Discriminator(in_c=2, nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):
        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(origin_grid), dim=0)
            real_prob_out = d(real_patch)
            d_real_loss = -real_prob_out.mean()
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.unsqueeze(origin_grid, dim=0) # torch.normal(mean=0, std=1.0, size=(1, 2, h, w)).to(device)
            generated_coord = torch.squeeze(mapper(noise_coord))
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_coord), dim=0)

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

            noise_coord = torch.unsqueeze(origin_grid, dim=0) # torch.normal(mean=0, std=1.0, size=(1, 2, h, w)).to(device)
            generated_coord = torch.squeeze(mapper(noise_coord))
            fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated_coord), dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            adv_loss.backward(retain_graph=True)
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", adv_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.flush()

        # Log image
        if iter == 0 or (iter + 1) % 500 == 0:
            generated = model(generated_coord.permute(1, 2, 0)).permute(2, 0, 1).detach()
            save_image(generated, f'exps/{EXP_NAME}/img/{iter}_all.jpg')
            visualize_grid(generated_coord.permute(1, 2, 0), f'exps/{EXP_NAME}/img/{iter}_whole.jpg', device)
            visualize_grid(fake_patch[0].permute(1, 2, 0), f'exps/{EXP_NAME}/img/{iter}_fake_grid.jpg', device)
            visualize_grid(real_patch[0].permute(1, 2, 0), f'exps/{EXP_NAME}/img/{iter}_real_grid.jpg', device)

    torch.save(mapper.state_dict(), f'exps/{EXP_NAME}/ckpt/mapper.pth')
    torch.save(d.state_dict(), f'exps/{EXP_NAME}/ckpt/D.pth')