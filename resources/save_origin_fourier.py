import os
from PIL import Image
from utils.fourier import get_fourier, viz_fourier

'''
    원본 이미지의 fourier feature visualize and save
'''
PATH = '../inputs/stripe.jpg'
SAVE = 'origin_fourier/stripe'


if __name__ == '__main__':
    os.makedirs(SAVE, exist_ok=True)

    img = Image.open(PATH)
    fourier_info = get_fourier(img)
    viz_fourier(fourier_info, dir=SAVE)
