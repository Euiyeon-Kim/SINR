import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.grid import create_grid
from utils.fourier import get_fourier, viz_fourier


EXP_NAME = 'birds_fourier/what_siren_do'
PATH = '../inputs/birds.png'
PTH_PATH = 'exps/birds_fourier/ckpt/final.pth'
B_PATH = 'exps/birds_fourier/ckpt/B.pt'


W0 = 50
MAPPING_SIZE = 256


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/tmp', exist_ok=True)

    img = np.array(Image.open(PATH).convert('RGB'))
    fourier_info = get_fourier(img)
    origin_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/origin')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenModel(coord_dim=2*MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH)

    # Analysis Single Scale
    infer_h, infer_w, c = img.shape
    os.makedirs(f'exps/{EXP_NAME}/recon', exist_ok=True)
    grid = create_grid(infer_h, infer_w, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    model.eval()
    pred = model(mapped_input)

    recon_img = (pred * 255.).detach().cpu().numpy()
    fourier_info = get_fourier(recon_img)
    inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/recon')

    save_image(pred.permute(2, 0, 1), f'exps/{EXP_NAME}/recon/pred.jpg')

    infer_h, infer_w = infer_h * 2, infer_w * 2

    os.makedirs(f'exps/{EXP_NAME}/double', exist_ok=True)
    grid = create_grid(infer_h, infer_w, device=device)
    x_proj = (2. * np.pi * grid) @ B.t()
    mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    model.eval()
    pred = model(mapped_input)

    recon_img = (pred * 255.).detach().cpu().numpy()
    fourier_info = get_fourier(recon_img)
    inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/double')

    rgb = np.array(recon_img)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    f = np.fft.fft2(gray)

    smaller_f = f[::2, ::2]
    smaller_shift = np.fft.fftshift(smaller_f)
    smaller_magnitude = np.abs(smaller_shift)
    smaller_phase = np.angle(smaller_shift)

    recon = smaller_magnitude * np.exp(complex(0, 1) * smaller_phase)
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon)))

    viz_magnitude = 20 * np.log(smaller_magnitude)
    random_mag = np.ones_like(smaller_magnitude) * 3000 * 2
    random_mag_fourier = random_mag * np.exp(complex(0, 1) * smaller_phase)
    viz_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(random_mag_fourier)))

    cv2.imwrite(f'exps/{EXP_NAME}/smaller_mag.jpg', viz_magnitude)
    cv2.imwrite(f'exps/{EXP_NAME}/smaller_phase.jpg', viz_phase)

    save_image(pred.permute(2, 0, 1), f'exps/{EXP_NAME}/double/pred.jpg')
