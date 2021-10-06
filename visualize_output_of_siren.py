import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch

from models.siren import SirenModel
from utils.grid import create_grid


EXP_NAME = 'mountain_fourier/analysis_fourier'
PATH = './inputs/mountains.jpg'
PTH_PATH = 'exps/mountain_fourier/ckpt/final.pth'
B_PATH = 'exps/mountain_fourier/ckpt/B.pt'

W0 = 50
MAPPING_SIZE = 256


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenModel(coord_dim=2*MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH)

    img = np.array(Image.open(PATH).convert('RGB'))
    infer_h, infer_w, c = img.shape

    grid = create_grid(infer_h, infer_w, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layers[0].register_forward_hook(hook)
    model.layers[1].register_forward_hook(hook)
    model.layers[2].register_forward_hook(hook)
    model.layers[3].register_forward_hook(hook)
    model.layers[4].register_forward_hook(hook)

    model.eval()
    pred = model(mapped_input)

    for tmp in outputs:
        print(tmp)
        print(tmp.shape)
        exit()

