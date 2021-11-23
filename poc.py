import torch

from models.adversarial import MappingNet
from utils.utils import make_exp_dirs, get_device, read_img

'''
    noise -> 1/4 크기 이미지
    1/4 크기 이미지 컨디션 + grid -> 원본 크기 이미지 
'''

EXP_NAME = f'poc/x4/balloon'
BIG_PATH = 'inputs/balloons.png'
SMALL_PATH = 'inputs/small_balloons.png'

SCALE = 10
MAPPING_SIZE = 64

MAX_ITERS = 100000
LR = 1e-4

SMALL_COND_RES = 3*3


if __name__ == '__main__':
    # writer = make_exp_dirs(EXP_NAME)
    device = get_device()

    small_img = read_img(SMALL_PATH)

    noise_mapper = MappingNet(in_f=SMALL_COND_RES, out_f=256, hidden_node=256)


