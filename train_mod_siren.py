import os
from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from models.siren import ModulatedSirenModel
from models.encoder import VGGEncoder
from utils.utils import create_grid

EXP_NAME = 'mod_balloon'
PATH = './inputs/balloons.png'
MAX_ITERS = 10000
LR = 1e-4

LATENT_DIM = 256
PATCH_SIZE = 120


if __name__ == '__main__':
    os.makedirs(f'exps/{EXP_NAME}/img', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{EXP_NAME}/logs', exist_ok=True)
    writer = SummaryWriter(f'exps/{EXP_NAME}/logs')

    img = Image.open(PATH).convert('RGB').resize((166, 128), Image.BICUBIC)
    # img = np.array(img) / 255.
    h, w = img.size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    grid = create_grid(PATCH_SIZE, PATCH_SIZE, device=device)
    in_f = grid.shape[-1]

    transforms = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ])

    encoder = VGGEncoder().to(device)
    model = ModulatedSirenModel(coord_dim=in_f, num_c=3, w0=60, latent_dim=LATENT_DIM).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for i in range(MAX_ITERS):
        patch = torch.unsqueeze(transforms(img), dim=0).to(device)

        model.train()

        optim.zero_grad()

        z = torch.squeeze(encoder(patch))
        pred = torch.unsqueeze(model(z, grid).permute(2, 0, 1), dim=0)

        loss = loss_fn(pred, patch)

        loss.backward()
        optim.step()

        writer.add_scalar("loss", loss.item(), i)

        if (i+1) % 50 == 0:
            save_image(patch[0], f'exps/{EXP_NAME}/img/{i}_origin.jpg')
            save_image(pred[0], f'exps/{EXP_NAME}/img/{i}_pred.jpg')

    torch.save(model.state_dict(), f'exps/{EXP_NAME}/ckpt/final.pth')