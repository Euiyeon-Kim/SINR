import os
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.utils import create_grid
from utils.utils import ResizeConfig as config


EXP_NAME = 'tmp'
PATH = './inputs/balloons.png'
PTH_NAME = 'final'


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/infer', exist_ok=True)

    img = Image.open(PATH).convert('RGB')
    img = np.array(img) / 255.
    h, w, c = img.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.FloatTensor(img).to(device)

    from utils.utils import creat_reals_pyramid, adjust_scales
    img = torch.unsqueeze(img.permute(2, 0, 1), dim=0)
    adjust_scales(img, config)
    reals = creat_reals_pyramid(img, [], config)

    os.makedirs('inputs/balloons_multiscale', exist_ok=True)
    for idx, real in enumerate(reals):
        save_image(real[0], f'inputs/balloons_multiscale/{idx}.jpg')
    exit()

    model = SirenModel(coord_dim=2, num_c=3).to(device)
    model.load_state_dict(torch.load(f'exps/{EXP_NAME}/ckpt/{PTH_NAME}.pth'))

    for idx, cur_scale in enumerate(reals):
        img = torch.squeeze(cur_scale).permute(1, 2, 0)
        h, w, c = img.shape
        grid = create_grid(h, w, device=device)
        in_f = grid.shape[-1]

        model.eval()
        pred = model(grid)

        pred = pred.permute(2, 0, 1)
        save_image(pred, f'exps/{EXP_NAME}/infer/{idx}.jpg')
        save_image(cur_scale, f'exps/{EXP_NAME}/infer/{idx}_origin.jpg')

