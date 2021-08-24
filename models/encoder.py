import torch
import torch.nn as nn
from torchvision.models import vgg16


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=pad)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_c=3, latent_dim=256):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, latent_dim, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.mean(dim=(-2, -1))


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.layers = vgg16(pretrained=True).features[:15]
        self.linears = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.mean(dim=(-2, -1))
        return self.linears(x)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    e = VGGEncoder().to(device)
    from torchsummary import summary
    summary(e, (3, 32, 32))
    exit()
