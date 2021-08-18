from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
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
        super().__init__()
        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.layers(coords)
        return x


class MappingNet(nn.Module):
    def __init__(self, in_f, out_f, hidden_node=128, depth=3):
        super().__init__()
        layers = [nn.Linear(in_f, hidden_node), nn.LeakyReLU(0.2)]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(hidden_node, hidden_node))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_node, out_f))
        layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Modulator(nn.Module):
    def __init__(self, in_f, hidden_node=256, depth=5):
        super().__init__()
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


if __name__ == '__main__':
    mapper = MappingNet(64, 64, 64, 5)
    print(mapper)