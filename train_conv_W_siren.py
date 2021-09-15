import os
import numpy as np
from PIL import Image

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.transforms import RandomCrop

from utils.viz import visualize_grid
from utils.utils import create_grid
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

EXP_NAME = 'balloons/multiscale/0'
PTH_PATH = 'exps/balloons/learnit_var_patch_64/inr_origin/ckpt/final.pth'
PATH = 'inputs/balloons_multiscale/0.png'

W0 = 50
MAX_ITERS = 1000000
LR = 1e-4

D_PAD = 5
N_CRITIC = 5
GEN_ITER = 2
PATCH_SIZE = 16
GP_LAMBDA = 0.1
RECON_LAMBDA = 10.0


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

    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    for param in model.parameters():
        param.trainable = False

    recon = model(grid).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/recon.jpg')

    mapper = MappingConv(in_c=2, out_c=2).to(device)
    m_optim = torch.optim.Adam(mapper.parameters(), lr=LR, betas=(0.5, 0.999))

    d = Discriminator(nfc=32).to(device)
    d_optim = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.5, 0.999))
    loss_fn = torch.nn.MSELoss()
    pad_fn = torch.nn.ZeroPad2d(D_PAD)

    recon_noise = torch.zeros((1, 2, h, w)).to(device)
    for iter in range(MAX_ITERS):
        # Train Discriminator
        for i in range(N_CRITIC):
            d.train()
            d_optim.zero_grad()

            # Train with real image
            real_patch = torch.unsqueeze(torch.nn.ZeroPad2d(D_PAD)(img), dim=0)
            real_prob_out = d(real_patch)

            d_real_loss = -real_prob_out.mean()
            d_real_loss.backward(retain_graph=True)

            # Train with fake image
            noise_coord = torch.randn((1, 1, h // 4, w // 4)).expand(1, 2, h // 4, w // 4).to(device)
            upsampled_coord = pad_fn(torch.nn.Upsample(size=(h, w), mode='nearest')(noise_coord))
            generated_coord = torch.squeeze(mapper(upsampled_coord)).permute(1, 2, 0)

            generated = torch.unsqueeze(model(generated_coord).permute(2, 0, 1).detach(), dim=0)
            fake_patch = torch.nn.ZeroPad2d(D_PAD)(generated)

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

            noise_coord = torch.randn((1, 1, h // 4, w // 4)).expand(1, 2, h // 4, w // 4).to(device)
            upsampled_coord = pad_fn(torch.nn.Upsample(size=(h, w), mode='nearest')(noise_coord))
            generated_coord = torch.squeeze(mapper(upsampled_coord)).permute(1, 2, 0)

            generated = torch.unsqueeze(model(generated_coord).permute(2, 0, 1).detach(), dim=0)
            fake_patch = pad_fn(generated)

            recon_coord = torch.squeeze(mapper(pad_fn(recon_noise))).permute(1, 2, 0)
            recon = model(recon_coord).permute(2, 0, 1)
            recon_loss = loss_fn(recon, img)

            fake_prob_out = d(fake_patch)
            d_fake_loss = fake_prob_out.mean()  # Minimize D(G(z))

            fake_prob_out = d(fake_patch)
            adv_loss = -fake_prob_out.mean()

            g_loss = adv_loss + RECON_LAMBDA * recon_loss
            g_loss.backward(retain_graph=True)
            m_optim.step()

        # Log mapper losses
        writer.add_scalar("g/total", g_loss.item(), iter)
        writer.add_scalar("g/critic", -adv_loss.item(), iter)
        writer.flush()

        # Log image
        if (iter + 1) % 10 == 0:
            # save_image(generated, f'exps/{EXP_NAME}/img/{iter}_all.jpg')
            save_image(recon, f'exps/{EXP_NAME}/img/{iter}_recon.jpg')
            save_image(generated, f'exps/{EXP_NAME}/img/{iter}_fake.jpg')
            visualize_grid(generated_coord, f'exps/{EXP_NAME}/img/{iter}_grid.jpg', device)
