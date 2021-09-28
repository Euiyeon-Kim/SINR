import cv2
import numpy as np


def get_fourier(rgb_img):
    rgb = np.array(rgb_img)
    gray = np.array(rgb_img.convert('L'))
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    fourier_info = {}
    for img, name in zip([r, g, b, gray], ['r', 'g', 'b', 'gray']):
        # F(u, v): 어느 주파수 성분이 얼마나, 어디서 부터 시작되어 표현되는가
        f = np.fft.fft2(img)
        shift = np.fft.fftshift(f)

        # F(u, v)를 복소평면 상에 표현했을 때 반지름
        # x축 방향으로 W / u pixel, y축 방향으로 H / v pixel 마다 반복되는 신호가 얼마나 들어있는지
        magnitude = np.abs(shift)
        # F(u, v)를 복소평면 상에 표현했을 때 radian
        phase = np.angle(shift)
        fourier_info[name] = {'mag': magnitude,
                              'phase': phase}
    return fourier_info


def viz_fourier(fourier_info, scale_phase=2, dir='.'):
    for k, v in fourier_info.items():
        recon = v['mag'] * np.exp(complex(0, 1) * v['phase'])
        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(recon)))

        viz_magnitude = 20 * np.log(v['mag'])

        random_mag = np.ones_like(v['mag']) * np.mean(v['mag']) * scale_phase
        random_mag_fourier = random_mag * np.exp(complex(0, 1) * v['phase'])
        viz_phase = np.abs(np.fft.ifft2(np.fft.ifftshift(random_mag_fourier)))

        cv2.imwrite(f'{dir}/{k}_recon.jpg', recon)
        cv2.imwrite(f'{dir}/{k}_mag.jpg', viz_magnitude)
        cv2.imwrite(f'{dir}/{k}_phase.jpg', viz_phase)


def recon_img_by_mag_and_phase(magnitude, phase):
    return np.abs(magnitude * np.exp(complex(0, 1) * phase))