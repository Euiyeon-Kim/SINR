import cv2
import numpy as np
from PIL import Image


if __name__ == '__main__':
    a = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [1, 2, 3, 4],
         [5, 6, 7, 8]]
    tmp = np.array(a)
    print(tmp[::2, ::2])
    exit()

    recon = np.array(Image.open('exps/birds_fourier/analysis_fourier/1.4^1/gray_mag.jpg'))
    double = np.array(Image.open('exps/birds_fourier/analysis_fourier/1.4^3/gray_mag.jpg'))

    gw, gh = recon.shape
    w, h = double.shape
    cut_w, cut_h = w - gw, h - gh

    cuted = double[cut_w//2:cut_w//2+gw, cut_h//2:cut_h//2+gh]
    cv2.imwrite('tmp.jpg', cuted)
    cv2.imwrite('diff.jpg', np.abs(cuted - recon))

    print(np.min(cuted), np.max(cuted))
    print(np.min(recon), np.max(recon))

    print(cuted.shape)

