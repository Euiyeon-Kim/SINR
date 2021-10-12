import os

import numpy as np
from PIL import Image


NAME = 'balloons'
PATH = f'./inputs/{NAME}.png'
PATCH_SIZE = 32
OVERLAP = 16


if __name__ == '__main__':
    os.makedirs(f'./inputs/{NAME}_patch_{PATCH_SIZE}', exist_ok=True)
    img = Image.open(PATH).convert('RGB')
    w, h = img.size
    cut_h, cut_w = (h - PATCH_SIZE) % (PATCH_SIZE - OVERLAP), (w - PATCH_SIZE) % (PATCH_SIZE - OVERLAP)
    new_img = np.array(img)[cut_h // 2:h - (cut_h - (cut_h // 2)), cut_w // 2:w - (cut_w - (cut_w // 2))]
    new_h, new_w, _ = new_img.shape

    num_h = int((new_h - PATCH_SIZE) / (PATCH_SIZE - OVERLAP) + 1)
    num_w = int((new_w - PATCH_SIZE) / (PATCH_SIZE - OVERLAP) + 1)
    h_offset, w_offset = 0, 0
    for i in range(num_h):
        for j in range(num_w):
            patch = new_img[h_offset:h_offset+PATCH_SIZE, w_offset:w_offset+PATCH_SIZE, :]
            Image.fromarray(patch).save(f'./inputs/{NAME}_patch_{PATCH_SIZE}/{i}_{j}.jpg')
            w_offset += PATCH_SIZE - OVERLAP
        w_offset = 0
        h_offset += PATCH_SIZE - OVERLAP