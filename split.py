import os

import numpy as np
from PIL import Image


NAME = 'mountains'
PATH = f'./inputs/{NAME}.jpg'
PATCH_SIZE = 32
OVERLAP = 16


if __name__ == '__main__':
    os.makedirs(f'./inputs/{NAME}_patch', exist_ok=True)
    img = Image.open(PATH).convert('RGB')
    w, h = img.size
    cut_h, cut_w = h % OVERLAP, w % OVERLAP
    new_img = np.array(img)[cut_h // 2:h - (cut_h - (cut_h // 2)), cut_w // 2:w - (cut_w - (cut_w // 2))]
    new_h, new_w, _ = new_img.shape
    # Image.fromarray(new_img).save(f'./inputs/{NAME}_patch/resized.jpg')

    h_offset, w_offset = 0, 0
    for i in range(h // OVERLAP - 1):
        for j in range(w // OVERLAP - 1):
            print(h_offset, w_offset)
            patch = new_img[h_offset:h_offset+PATCH_SIZE, w_offset:w_offset+PATCH_SIZE, :]
            Image.fromarray(patch).save(f'./inputs/{NAME}_patch/{i}_{j}.jpg')
            w_offset += PATCH_SIZE - OVERLAP
        w_offset = 0
        h_offset += PATCH_SIZE - OVERLAP