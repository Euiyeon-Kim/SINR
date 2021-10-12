import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch

from models.siren import SirenHook
from utils.grid import create_grid
from utils.utils import grid_to_fourier_inp


EXP_NAME = 'mountain_fourier/analysis_fourier'
PATH = './inputs/mountains.jpg'
PTH_PATH = 'exps/mountain_fourier/ckpt/final.pth'
B_PATH = 'exps/mountain_fourier/ckpt/B.pt'

W0 = 50
MAPPING_SIZE = 256


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenHook(coord_dim=2*MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device), strict=False)
    B = torch.load(B_PATH).to(device)

    img = np.array(Image.open(PATH).convert('RGB'))
    infer_h, infer_w, c = img.shape

    grid = create_grid(infer_h, infer_w, device=device)
    mapped_input = grid_to_fourier_inp(grid, B).to(device)

    model.eval()
    pred = model(mapped_input)

    pred = pred.permute(2, 0, 1)
    from torchvision.utils import save_image
    save_image(pred, 'tmp.jpg')
