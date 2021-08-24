import os
import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop

from utils.utils import create_grid, calcul_gp
from models.siren import SirenModel
from models.adversarial import Discriminator, MappingNet

'''
    Random coord -> W(MLP) -> Generated coord
    Generated coord -> Model -> Generated image 
    Generated image -> Crop patch -> Discriminator
    
    Result
    W(MLP)에서 spatial 정보가 다 날아가서 실패하는 듯
'''

EXP_NAME = 'mlp_w_balloon'
PATH = '../inputs/balloons.png'
PTH_NAME = 'final'
MAX_ITERS = 10000
LR = 1e-4

N_CRITIC = 5
GEN_ITER = 1
PATCH_SIZE = 64
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
    print(torch.min(grid), torch.max(grid))
    exit()
    model = SirenModel(coord_dim=2, num_c=3).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))
    for param in model.parameters():
        param.requires_grad = False
    recon = model(grid).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    mapper = MappingNet(in_f=2, out_f=2).to(device)
    m_optim = torch.optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))

    d = Discriminator(nfc=64).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))

    for iter in range(MAX_ITERS):

        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(img), dim=0)
            real_prob_out = d(real_patch)
            d_real_loss = -real_prob_out.mean()  # Maximize D(X) -> Minimize -D(X)\
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.randn(PATCH_SIZE, PATCH_SIZE, 2).to(device)
            generated_coord = mapper(noise_coord)
            generated = model(generated_coord).permute(2, 0, 1).detach()
            # fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated), dim=0)
            fake_patch = torch.unsqueeze(generated, dim=0)

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

            noise_coord = torch.randn(PATCH_SIZE, PATCH_SIZE, 2).to(device)
            generated_coord = mapper(noise_coord)
            generated = model(generated_coord).permute(2, 0, 1)
            fake_patch = torch.unsqueeze(generated, dim=0)
            # fake_patch = torch.unsqueeze(RandomCrop(size=PATCH_SIZE)(generated), dim=0)

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            adv_loss.backward(retain_graph=True)
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", adv_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.flush()

        # Log image
        if (iter + 1) % 10 == 0:
            save_image(generated, f'exps/{EXP_NAME}/w/img/{iter}_all.jpg')
            save_image(fake_patch, f'exps/{EXP_NAME}/w/img/{iter}_patch.jpg')
            save_image(real_patch, f'exps/{EXP_NAME}/w/img/{iter}_real.jpg')