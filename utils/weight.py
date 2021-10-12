import torch

outputs = []


def hook(module, input, output):
    outputs.append(output)


model.layers[0].register_forward_hook(hook)
model.layers[1].register_forward_hook(hook)
model.layers[2].register_forward_hook(hook)
model.layers[3].register_forward_hook(hook)
model.layers[4].register_forward_hook(hook)
