import cv2
import numpy as np
from PIL import Image

from utils.fourier import get_fourier, viz_fourier

PATH = '../inputs/mountains.jpg'

W0 = 50
MAPPING_SIZE = 256


if __name__ == '__main__':
    img = np.array(Image.open(PATH).convert('RGB'))
    rgb = np.array(img)
    h, w, c = rgb.shape
    rgb = rgb[:h//4*4, :w//4*4, :]

    patch_size = 128
    rh1, rw1 = np.random.randint(h - patch_size), np.random.randint(w - patch_size)
    rh2, rw2 = np.random.randint(h - patch_size), np.random.randint(w - patch_size)
    swap_rgb = np.copy(rgb)
    swap_rgb[rh1:rh1 + patch_size, rw1:rw1 + patch_size, :] = rgb[rh2:rh2 + patch_size, rw2:rw2 + patch_size, :]
    swap_rgb[rh2:rh2 + patch_size, rw2:rw2 + patch_size, :] = rgb[rh1:rh1 + patch_size, rw1:rw1 + patch_size, :]
    cv2.imwrite('swap_rgb.jpg', cv2.cvtColor(swap_rgb, cv2.COLOR_RGB2BGR))

    origin_fourier_info = get_fourier(rgb)
    swap_fourier_info = get_fourier(swap_rgb)

    origin_viz_info = viz_fourier(origin_fourier_info, fixed_mag=5000, dir=None)
    swap_viz_info = viz_fourier(origin_fourier_info, fixed_mag=5000, dir=None)

    swap_r = np.abs(np.fft.ifft2(np.fft.ifftshift(origin_fourier_info['r']['mag'] * np.exp(complex(0, 1) * swap_fourier_info['r']['phase']))))
    swap_g = np.abs(np.fft.ifft2(np.fft.ifftshift(origin_fourier_info['g']['mag'] * np.exp(complex(0, 1) * swap_fourier_info['g']['phase']))))
    swap_b = np.abs(np.fft.ifft2(np.fft.ifftshift(origin_fourier_info['b']['mag'] * np.exp(complex(0, 1) * swap_fourier_info['b']['phase']))))
    cv2.imwrite('swap_r.jpg', swap_r)
    swap_recon = np.stack((swap_b, swap_g, swap_r), axis=-1)
    print(np.min(swap_recon), np.max(swap_recon))
    cv2.imwrite('swap_recon.jpg', swap_recon)
