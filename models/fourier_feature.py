import numpy as np

import torch
import torch.nn as nn


class FFModel(nn.Module):
    def __init__(self, coord_dim, num_c, hidden_node=256, depth=5):
        super(FFModel, self).__init__()
        layers = [nn.Linear(coord_dim, hidden_node), nn.ReLU()]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(hidden_node, hidden_node))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_node, num_c))
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.layers(coords)
        return nn.Sigmoid()(x)

