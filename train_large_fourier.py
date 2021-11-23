import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.siren import FourierReLU
from utils.dataloader import LargeINR
from utils.utils import make_exp_dirs, get_device, read_img, sample_B

EXP_NAME = 'flood/origin/bush'
PATH = 'inputs/wild_bush.jpg'

MAX_ITERS = 100000
LR = 1e-4

MAPPING_SIZE = 256
SCALE = 10

if __name__ == '__main__':

    writer = make_exp_dirs(EXP_NAME)
    device = get_device()

    img = torch.FloatTensor(read_img(PATH)).to(device)
    h, w, _ = img.shape
    # B = sample_B(MAPPING_SIZE, SCALE, device)
    B_gauss = np.random.randn(MAPPING_SIZE, 2) * SCALE

    dataset = LargeINR(PATH, B_gauss)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=4096, drop_last=True)
    test_loader = DataLoader(dataset, shuffle=False, batch_size=2048, drop_last=False)

    model = FourierReLU(coord_dim=MAPPING_SIZE, num_c=3, hidden_node=256, depth=5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    step_per_epoch = len(dataloader)
    for i in range(MAX_ITERS):
        model.train()
        for j, data in enumerate(dataloader):
            coord, pixel = data
            coord, pixel = coord.to(device), pixel.to(device)

            optim.zero_grad()
            pred = model(coord)
            loss = loss_fn(pred, pixel)

            loss.backward()
            optim.step()

            writer.add_scalar("loss", loss.item(), i*step_per_epoch+j)

        if i % 10 == 0:
            model.eval()
            pred = np.zeros((h, w, 3))
            for h_idx, data in enumerate(test_loader):
                coord, pixel = data
                coord, pixel = coord.to(device), pixel.to(device)

                pred_w = model(coord)
                pred[h_idx, :, :] = pred_w.detach().cpu().numpy()

            pred = Image.fromarray(np.array(pred * 255.).astype(np.uint8))
            pred.save(f'exps/{EXP_NAME}/img/{i}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
    torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')



