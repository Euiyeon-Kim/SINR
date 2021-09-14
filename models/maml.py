from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import functional as F


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
        super(SirenLayer, self).__init__()
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


class MAML(nn.Module):
    def __init__(self, coord_dim, num_c, inner_steps=3, inner_lr=1e-2, w0=200, hidden_node=256, depth=5):
        super().__init__()
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.model = SirenModel(coord_dim, num_c, w0, hidden_node, depth)

    def _inner_iter(self, coords, img, params, detach):
        with torch.enable_grad():
            # forward pass
            pred = self.model(coords, params)
            loss = F.mse_loss(pred, img)

            # backward pass
            grads = autograd.grad(loss, params.values(), create_graph=(not detach), only_inputs=True, allow_unused=True)

            # parameter update
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    updated_param = param
                else:
                    updated_param = param - self.inner_lr * grad
                if detach:
                    updated_param = updated_param.detach().requires_grad_(True)
                updated_params[name] = updated_param

        return updated_params

    def _adapt(self, coords, img, params, meta_train):
        for step in range(self.inner_steps):
            params = self._inner_iter(coords, img, params, not meta_train)
        return params

    def forward(self, coords, data, meta_train):
        # a dictionary of parameters that will be updated in the inner loop
        params = OrderedDict(self.model.named_parameters())

        preds = []
        for ep in range(data.size(0)):
            # inner-loop training
            self.train()
            updated_params = self._adapt(coords, data[ep], params, meta_train)

            with torch.set_grad_enabled(meta_train):
                self.eval()
                pred = self.model(coords, updated_params)
            preds.append(pred)

        self.train(meta_train)
        preds = torch.stack(preds)
        return preds
