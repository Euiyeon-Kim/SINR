import os

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision.utils import save_image

from models.siren import SirenModel
from utils.grid import create_grid
from utils.fourier import get_fourier, viz_fourier


'''
    Multiscale inference 하면서 INR이 뭘 채워넣는지 확인하기
'''

EXP_NAME = 'stripe/analysis_fourier'
PATH = '../inputs/mountains.jpg'
PTH_PATH = 'exps/stripe/ckpt/final.pth'

W0 = 50
SCALE = 1.4
ITER = 3

if __name__ == '__main__':
    img = np.array(Image.open(PATH).convert('RGB'))
    fourier_info = get_fourier(img)
    origin_viz_dict = viz_fourier(fourier_info, dir=f'..')

    os.makedirs(f'exps/{EXP_NAME}/origin', exist_ok=True)

    img = np.array(Image.open(PATH).convert('RGB'))
    fourier_info = get_fourier(img)
    origin_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/origin')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))

    # Analysis Single Scale
    infer_h, infer_w, c = img.shape
    os.makedirs(f'exps/{EXP_NAME}/recon', exist_ok=True)
    grid = create_grid(infer_h, infer_w, device=device)

    model.eval()
    pred = model(grid)

    recon_img = (pred * 255.).detach().cpu().numpy()
    fourier_info = get_fourier(recon_img)
    inr_viz_dict = viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/recon')

    cv2.imwrite(f'exps/{EXP_NAME}/recon/diff_recon.jpg', np.abs(recon_img - img))
    for k, v in inr_viz_dict.items():
        cv2.imwrite(f'exps/{EXP_NAME}/recon/diff_{k}_mag.jpg', np.abs(origin_viz_dict[k]['mag'] - inr_viz_dict[k]['mag']))
        cv2.imwrite(f'exps/{EXP_NAME}/recon/diff_{k}_phase.jpg', np.abs(origin_viz_dict[k]['phase'] - inr_viz_dict[k]['phase']))

    save_image(pred.permute(2, 0, 1), f'exps/{EXP_NAME}/recon/pred.jpg')

    # Multi Scale
    for i in range(ITER):
        infer_h = int(infer_h * SCALE)
        infer_w = int(infer_w * SCALE)

        os.makedirs(f'exps/{EXP_NAME}/{SCALE}^{i+1}', exist_ok=True)

        grid = create_grid(infer_h, infer_w, device=device)
        model.eval()
        pred = model(grid)

        pred_img = (pred * 255.).detach().cpu().numpy()
        fourier_info = get_fourier(pred_img)
        viz_fourier(fourier_info, dir=f'exps/{EXP_NAME}/{SCALE}^{i+1}')

        save_image(pred.permute(2, 0, 1), f'exps/{EXP_NAME}/{SCALE}^{i+1}/pred.jpg')
