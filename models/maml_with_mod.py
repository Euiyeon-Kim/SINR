from collections import OrderedDict

import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import functional as F


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


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


class ModulationLayer(nn.Module):
    def __init__(self, in_f, out_f, is_last=False):
        super(ModulationLayer, self).__init__()
        self.in_f = in_f
        self.out_f = out_f
        self.linear = Linear(in_f, out_f)
        self.is_last = is_last

    def forward(self, x, params=None):
        x = self.linear(x, params)
        return x if self.is_last else torch.relu(x)


class Modulator(nn.Module):
    def __init__(self, in_f, hidden_node=256, depth=5):
        super(Modulator, self).__init__()
        layers = [ModulationLayer(in_f=in_f, out_f=hidden_node)]
        for i in range(depth):
            layers.append(ModulationLayer(in_f=hidden_node+in_f, out_f=hidden_node))
        self.layers = nn.Sequential(*layers)

    def forward(self, z, params=None):
        x = z
        hiddens = []
        for idx, layer in enumerate(self.layers):
            layer_param = OrderedDict()
            layer_param['weight'] = params.get(f'modulator.layers.{idx}.linear.weight')
            layer_param['bias'] = params.get(f'modulator.layers.{idx}.linear.bias')
            x = layer(x, layer_param)
            hiddens.append(x)
            x = torch.cat((x, z))
        return tuple(hiddens)


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


class ModulatedSirenModel(nn.Module):
    def __init__(self, coord_dim, num_c, w0=200, hidden_node=256, depth=5, latent_dim=256):
        super(ModulatedSirenModel, self).__init__()
        self.depth = 5
        self.modulator = Modulator(in_f=latent_dim, hidden_node=hidden_node, depth=depth-1)
        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        self.layers = nn.Sequential(*layers)
        self.last_layer = SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True)

    def forward(self, latents, coords, params):
        x = coords
        mods = self.modulator(latents, params)
        mods = cast_tuple(mods, self.depth)
        for idx, (layer, mod) in enumerate(zip(self.layers, mods)):
            layer_param = OrderedDict()
            layer_param['weight'] = params.get(f'layers.{idx}.linear.weight')
            layer_param['bias'] = params.get(f'layers.{idx}.linear.bias')
            x = layer(x, layer_param)
            x *= rearrange(mod, 'd -> () d')
        return self.last_layer(x)


class MAML(nn.Module):
    def __init__(self, coord_dim, num_c, inner_steps=3, inner_lr=1e-2, w0=200, hidden_node=256, depth=5, latent_dim=256):
        super().__init__()
        self.latent_dim = hidden_node
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.model = ModulatedSirenModel(coord_dim, num_c, w0, hidden_node, depth, latent_dim)

    def _inner_iter(self, z, coords, img, params, detach):
        with torch.enable_grad():
            # forward pass
            pred = self.model(z, coords, params)
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

    def _adapt(self, z, coords, img, params, meta_train):
        for step in range(self.inner_steps):
            params = self._inner_iter(z, coords, img, params, not meta_train)
        return params

    def forward(self, z, coords, data, meta_train):
        # a dictionary of parameters that will be updated in the inner loop
        params = OrderedDict(self.model.named_parameters())

        preds = []
        for ep in range(data.size(0)):
            # inner-loop training
            self.train()
            updated_params = self._adapt(z[ep], coords, data[ep], params, meta_train)

            with torch.set_grad_enabled(meta_train):
                self.eval()
                z = torch.normal(mean=0.0, std=1.0, size=(1, self.latent_dim)).cuda()
                pred = self.model(z, coords, updated_params)
            preds.append(pred)

        self.train(meta_train)
        preds = torch.stack(preds)
        return preds
