import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.grid import create_grid
from utils.fourier import get_fourier, viz_fourier

EXP_NAME = 'stripe_fourier/analysis_fourier/comp_with_bicubic'
PATH = './inputs/stripe.jpg'
PTH_PATH = 'exps/stripe_fourier/ckpt/final.pth'
B_PATH = 'exps/stripe_fourier/ckpt/B.pt'

W0 = 50
MAPPING_SIZE = 1

SCALE = 1.5
ITER = 1

if __name__ == '__main__':
    img = Image.open(PATH).convert('RGB')
    infer_w, infer_h = img.size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenModel(coord_dim=2*MAPPING_SIZE, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))
    B = torch.load(B_PATH)

    # Multi Scale
    for i in range(ITER):
        os.makedirs(f'exps/{EXP_NAME}/{SCALE}^{i + 1}', exist_ok=True)
        infer_h, infer_w = int(infer_h * SCALE), int(infer_w * SCALE)

        resized = np.array(img.resize((infer_w, infer_h), Image.BICUBIC))
        cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i+1}/resized.jpg', cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        resized_fourier_info = get_fourier(resized)
        resized_viz_dict = viz_fourier(resized_fourier_info, dir=f'exps/{EXP_NAME}/{SCALE}^{i+1}', prefix='resized_')

        grid = create_grid(infer_h, infer_w, device=device)
        x_proj = (2. * np.pi * grid) @ B.t()
        mapped_input = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        model.eval()
        pred = model(mapped_input)

        pred_img = (pred * 255.).detach().cpu().numpy()
        fourier_info = get_fourier(pred_img)
        inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/{SCALE}^{i+1}', prefix='inr_')

        for k, v in resized_fourier_info.items():
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i+1}/diff_{k}_r>i_mag.jpg',
                        np.clip(resized_viz_dict[k]['mag'] - inr_viz_dict[k]['mag'], a_min=0, a_max=None))
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i+1}/diff_{k}_r>i_phase.jpg',
                        np.clip(resized_viz_dict[k]['phase'] - inr_viz_dict[k]['phase'], a_min=0, a_max=None))
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i + 1}/diff_{k}_r<i_mag.jpg',
                        np.clip(inr_viz_dict[k]['mag'] - resized_viz_dict[k]['mag'], a_min=0, a_max=None))
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i + 1}/diff_{k}_r<i_phase.jpg',
                        np.clip(inr_viz_dict[k]['phase'] - resized_viz_dict[k]['phase'], a_min=0, a_max=None))
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i + 1}/diff_{k}_abs_mag.jpg',
                        np.abs(inr_viz_dict[k]['mag'] - resized_viz_dict[k]['mag']))
            cv2.imwrite(f'exps/{EXP_NAME}/{SCALE}^{i + 1}/diff_{k}_abs_phase.jpg',
                        np.abs(inr_viz_dict[k]['phase'] - resized_viz_dict[k]['phase']))

        save_image(pred.permute(2, 0, 1), f'exps/{EXP_NAME}/{SCALE}^{i + 1}/pred.jpg')
