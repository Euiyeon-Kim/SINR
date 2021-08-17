from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True):
        super(Linear, self).__init__(in_f, out_f, bias=bias)

    def forward(self, x, params=None):
        if params is None:
            x = super(Linear, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.linear(x, weight, bias)
        return x


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=200, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.w0 = w0

        self.linear = Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last

        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, a=-b, b=b)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, params=None):
        x = self.linear(x, params)
        return x if self.is_last else torch.sin(self.w0 * x)


class SirenModel(nn.Module):
    def __init__(self, coord_dim, num_c, w0=200, hidden_node=256, depth=5):
        super().__init__()
        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, coords, params=None):
        if params is None:
            x = self.layers(coords)
        else:
            x = coords
            for idx, layer in enumerate(self.layers):
                layer_param = OrderedDict()
                layer_param['weight'] = params.get(f'layers.{idx}.linear.weight')
                layer_param['bias'] = params.get(f'layers.{idx}.linear.bias')
                x = layer(x, layer_param)
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


if __name__ == '__main__':
    mapper = MappingNet(64, 64, 64, 5)
    print(mapper)