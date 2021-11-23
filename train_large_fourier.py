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

MAX_ITERS = 1000
LR = 1e-4

MAPPING_SIZE = 256
SCALE = 10

if __name__ == '__main__':

    writer = make_exp_dirs(EXP_NAME)
    device = get_device()

    img = torch.FloatTensor(read_img(PATH)).to(device)
    h, w, _ = img.shape

    dataset = LargeINR(PATH)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=2048, drop_last=True)
    test_loader = DataLoader(dataset, shuffle=False, batch_size=2048, drop_last=False)

    B = sample_B(MAPPING_SIZE, SCALE, device)
    model = FourierReLU(coord_dim=MAPPING_SIZE, num_c=3, hidden_node=256, depth=5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        model.train()
        for data in dataloader:
            coord, pixel, _ = data
            coord, pixel = coord.to(device), pixel.to(device)
            mapped_input = torch.sin((2. * np.pi * coord) @ B.t())

            optim.zero_grad()
            pred = model(mapped_input)
            loss = loss_fn(pred, img)

            loss.backward()
            optim.step()

            writer.add_scalar("loss", loss.item(), i)

        if i % 10 == 0:
            model.eval()
            pred = np.zeros((h, w, 3))
            for h_idx, data in enumerate(test_loader):
                coord, pixel, origin_coord = data
                coord, pixel = coord.to(device), pixel.to(device)
                mapped_input = torch.sin((2. * np.pi * coord) @ B.t())
                pred_w = model(mapped_input)
                pred[h_idx, :, :] = pred_w.detach().cpu().numpy()

            pred = Image.fromarray(np.array(pred * 255.).astype(np.uint8))
            pred.save(f'exps/{EXP_NAME}/img/{i}.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')
    torch.save(B, f'exps/{EXP_NAME}/ckpt/B.pt')



