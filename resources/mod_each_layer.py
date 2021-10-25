import numpy as np
from PIL import Image

import torch

from models.siren import SirenModel
from utils.grid import create_grid
from utils.utils import make_exp_dirs, get_device

'''
    뒷 쪽 layer만 튜닝할 수록 원본에서 얼마 안바뀜
'''
EXP_NAME = 'balloons/learnit_with_transform/tune_sec'

PATH = '../inputs/balloons.png'
PTH_PATH = '../exps/balloons/learnit_with_transform/ckpt/final.pth'

W0 = 50


if __name__ == '__main__':
    writer = make_exp_dirs(EXP_NAME, log=True)

    device = get_device()
    model = SirenModel(coord_dim=2, num_c=3, w0=W0).to(device)
    model.load_state_dict(torch.load(PTH_PATH))

    img = torch.FloatTensor(np.array(Image.open(PATH).convert('RGB')) / 255.).to(device)
    infer_h, infer_w, _ = img.shape

    grid = create_grid(infer_h, infer_w, device=device)

    for param in model.layers[4].parameters():
        param.requires_grad = False
    for param in model.layers[1].parameters():
        param.requires_grad = False
    for param in model.layers[2].parameters():
        param.requires_grad = False
    for param in model.layers[3].parameters():
        param.requires_grad = False
    # model.layers[0].linear.bias.requires_grad = False

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    for i in range(1000):
        model.train()
        optim.zero_grad()

        pred = model(grid)
        loss = loss_fn(pred, img)

        loss.backward()
        optim.step()

        if i % 10 == 0:
            from torchvision.utils import save_image
            pred = pred.permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{i}.jpg')