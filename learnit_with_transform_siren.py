import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.utils import save_image

from models.maml import MAML
from utils.utils import make_exp_dirs, get_device, create_grid, sample_B

EXP_NAME = 'balloons/learnit_with_transform'
PATH = 'inputs/balloons.png'

W0 = 50

BATCH_SIZE = 1
INNER_STEPS = 2
MAX_ITERS = 20000

OUTER_LR = 1e-5
INNER_LR = 1e-2


if __name__ == '__main__':
    # Prepare exp
    device = get_device()
    writer = make_exp_dirs(EXP_NAME, log=True)

    # Read Image
    img = Image.open(PATH)
    w, h = img.size

    affine_transformer = T.Compose([T.RandomAffine(degrees=30, translate=(0, 0.15), scale=(0.75, 1.25), shear=15,
                                                   interpolation=T.InterpolationMode.BICUBIC),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()])

    # Prepare model
    maml = MAML(coord_dim=2, num_c=3, w0=W0).to(device)
    outer_optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = torch.nn.MSELoss()

    grid = create_grid(h, w, device)
    for outer_step in range(MAX_ITERS):
        data = torch.unsqueeze(affine_transformer(img), dim=0)
        data = data.permute(0, 2, 3, 1).to(device)

        maml.train()
        pred = maml(grid, data, True)

        loss = loss_fn(pred, data)
        print(f'{outer_step}/{MAX_ITERS}: {loss.item()}')

        outer_optimizer.zero_grad()
        loss.backward()
        outer_optimizer.step()

        writer.add_scalar("loss", loss.item(), outer_step)

        if (outer_step + 1) % 100 == 0:
            pred = pred[0].permute(2, 0, 1)
            data = data[0].permute(2, 0, 1)
            save_image(pred, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_pred.jpg')
            save_image(data, f'exps/{EXP_NAME}/img/{outer_step}_{loss.item():.8f}_data.jpg')
            meta = maml.model(grid).permute(2, 0, 1)
            save_image(meta, f'exps/{EXP_NAME}/img/{outer_step}_meta.jpg')

        if (outer_step + 1) % 1000 == 0:
            torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/{outer_step}.pth')

    torch.save(maml.model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')