import os
import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop

from utils.grid import create_grid, visualize_grid
from utils.loss import calcul_gp
from models.siren import SirenModel
from models.adversarial import Discriminator, MappingNet

'''
    Random coord -> W(MLP) -> Generated coord 
    Generated coord -> Crop patch -> Discriminator
    
    Result
    W(MLP)에서 spatial 정보가 다 날아가서 실패하는 듯
'''

EXP_NAME = 'mlp_w_bird'
PATH = '../inputs/birds.png'
PTH_PATH = '../exps/bird/ckpt/final.pth'

MAX_ITERS = 10000
LR = 1e-4

N_CRITIC = 5
GEN_ITER = 1
PATCH_SIZE = 96
GP_LAMBDA = 0.1


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
    origin_grid = grid.permute(2, 0, 1)

    model = SirenModel(coord_dim=2, num_c=3).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.requires_grad = False
    recon = model(grid).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    mapper = MappingNet(in_f=2, out_f=2).to(device)
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
            d_real_loss = -real_prob_out.mean()  # Maximize D(X) -> Minimize -D(X)
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.normal(mean=0, std=1.0, size=(h, w, 2)).to(device)
            generated_coord = mapper(noise_coord).permute(2, 0, 1)
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

            noise_coord = torch.normal(mean=0, std=1.0, size=(h, w, 2)).to(device)
            generated_coord = mapper(noise_coord).permute(2, 0, 1)
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
        if (iter + 1) % 50 == 0:
            generated = model(generated_coord.permute(1, 2, 0)).permute(2, 0, 1)
            save_image(generated, f'exps/{EXP_NAME}/w/img/{iter}_all.jpg')
            visualize_grid(generated_coord.permute(1, 2, 0), f'exps/{EXP_NAME}/w/img/{iter}_whole.jpg', device, 'generated_coord_all')
            visualize_grid(fake_patch[0].permute(1, 2, 0), f'exps/{EXP_NAME}/w/img/{iter}_fake_grid.jpg', device, 'fake_patch')
            visualize_grid(real_patch[0].permute(1, 2, 0), f'exps/{EXP_NAME}/w/img/{iter}_real_grid.jpg', device, 'real_patch')