import numpy as np

import torch
import torch.nn as nn
from einops import rearrange


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super(SirenLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.w0 = w0

        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last

        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, a=-b, b=b)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SirenModel(nn.Module):
    def __init__(self, coord_dim, num_c, w0=200, hidden_node=256, depth=5):
        super(SirenModel, self).__init__()
        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.layers(coords)
        return x


class Modulator(nn.Module):
    def __init__(self, in_f, hidden_node=256, depth=5):
        super(Modulator, self).__init__()
        self.layers = nn.ModuleList([])

        for i in range(depth):
            dim = in_f if i == 0 else (hidden_node + in_f)
            self.layers.append(nn.Sequential(
                nn.Linear(dim, hidden_node),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))
        return tuple(hiddens)


class ModulatedSirenModel(nn.Module):
    def __init__(self, coord_dim, num_c, w0=200, hidden_node=256, depth=5, latent_dim=256):
        super(ModulatedSirenModel, self).__init__()
        self.depth = 5
        self.modulator = Modulator(in_f=latent_dim, depth=depth-1)

        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        self.layers = nn.Sequential(*layers)
        self.last_layer = SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True)

    def forward(self, latents, coords):
        x = coords
        mods = self.modulator(latents)
        mods = cast_tuple(mods, self.depth)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            x *= rearrange(mod, 'd -> () d')
        return self.last_layer(x)


if __name__ == '__main__':
    from utils.grid import create_grid
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ModulatedSirenModel(2, 3).to(device)
    latent = torch.randn(256).to(device)
    coord = create_grid(64, 64, device)
    recon = model(latent, coord)
    print(recon.shape)