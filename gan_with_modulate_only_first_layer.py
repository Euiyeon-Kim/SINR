import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from models.siren import SirenModelMod
from utils.grid import create_grid
from utils.utils import make_exp_dirs, get_device, prepare_fourier_inp
from models.adversarial import Discriminator, MappingNet

EXP_NAME = 'balloons_fourier/learnit_with_transform/mod_first/w*mod+bias'

PATH = 'inputs/balloons.png'
PTH_PATH = 'exps/balloons_fourier/learnit_with_transform/ckpt/14999.pth'
B_PATH = 'exps/balloons_fourier/learnit_with_transform/ckpt/B.pth'

W0 = 50
MAPPING_SIZE = 256

MAX_ITERS = 100000
LR = 1e-2

N_CRITIC = 5
GEN_ITER = 1
GP_LAMBDA = 0.1


if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME, log=True)

    device = get_device()
    model = SirenModelMod(coord_dim=2 * MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH)

    img = torch.FloatTensor(np.array(Image.open(PATH).convert('RGB')) / 255.).to(device)
    infer_h, infer_w, _ = img.shape

    grid = create_grid(infer_h, infer_w, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    for param in model.parameters():
        param.requires_grad = False
    recon = model(mapped_input).permute(2, 0, 1)
    save_image(recon, f'exps/{EXP_NAME}/meta.jpg')

    find = torch.randn((1, 256)).to(device).detach().requires_grad_(True)
    optim = torch.optim.Adam({find}, lr=LR)

    # b_mapper = MappingNet(in_f=64, out_f=256).to(device)
    # w_mapper = MappingNet(in_f=64, out_f=256).to(device)
    # optim = torch.optim.Adam(list(b_mapper.parameters())+list(w_mapper.parameters()), lr=LR, betas=(0.5, 0.999))

    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        optim.zero_grad()

        # z = torch.randn(1, 64).to(device)
        # bias_param = b_mapper(z)
        # mod_param = w_mapper(z)
        pred = model(mapped_input, mod=True, bias_param=None, mod_param=find)
        loss = loss_fn(pred, img)
        print(torch.sum(find))
        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if i % 10 == 0:
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')