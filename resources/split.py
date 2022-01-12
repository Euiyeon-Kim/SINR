import os

import numpy as np
from PIL import Image


NAME = 'wild_bush'
PATH = f'./inputs/{NAME}.jpg'
PATCH_H = 48*2
PATCH_W = 64*2
OVERLAP_H = 12
OVERLAP_W = 16

if __name__ == '__main__':
    os.makedirs(f'./inputs/{NAME}_patch_{PATCH_H}*{PATCH_W}', exist_ok=True)

    img = Image.open(PATH).convert('RGB')
    w, h = img.size
    cut_h, cut_w = (h - PATCH_H) % (PATCH_H - OVERLAP_H), (w - PATCH_W) % (PATCH_W - OVERLAP_W)
    new_img = np.array(img)[cut_h // 2:h - (cut_h - (cut_h // 2)), cut_w // 2:w - (cut_w - (cut_w // 2))]
    new_h, new_w, _ = new_img.shape
    print(f'{new_h} * {new_w}')
    num_h = int((new_h - PATCH_H) / (PATCH_H - OVERLAP_H) + 1)
    num_w = int((new_w - PATCH_W) / (PATCH_W - OVERLAP_W) + 1)
    h_offset, w_offset = 0, 0
    for i in range(num_h):
        for j in range(num_w):
            patch = new_img[h_offset:h_offset+PATCH_H, w_offset:w_offset+PATCH_W, :]
            Image.fromarray(patch).save(f'./inputs/{NAME}_patch_{PATCH_H}*{PATCH_W}/{i}_{j}.jpg')
            w_offset += PATCH_W - OVERLAP_W
        w_offset = 0
        h_offset += PATCH_H - OVERLAP_H