from torch import nn
from models.siren import SirenLayer


class MappingNet(nn.Module):
    def __init__(self, in_f, out_f, hidden_node=256, depth=5):
        super(MappingNet, self).__init__()
        layers = [nn.Linear(in_f, hidden_node), nn.LeakyReLU(0.2)]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(hidden_node, hidden_node))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_node, out_f))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MappingConv(nn.Module):
    def __init__(self, in_c=2, out_c=3, nfc=32, min_nfc=32, num_layers=5):
        super(MappingConv, self).__init__()

        N = nfc
        self.head = ConvBlock(in_c, N, 3, 1, 0)

        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), 3, 1, 0)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Conv2d(max(N, min_nfc), out_c, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return nn.Sigmoid()(x)


class MappingSIREN(nn.Module):
    def __init__(self, coord_dim, num_c, w0=200, hidden_node=256, depth=5):
        super(MappingSIREN, self).__init__()
        layers = [SirenLayer(in_f=coord_dim, out_f=hidden_node, w0=w0, is_first=True)]
        for _ in range(1, depth - 1):
            layers.append(SirenLayer(in_f=hidden_node, out_f=hidden_node, w0=w0))
        layers.append(SirenLayer(in_f=hidden_node, out_f=num_c, is_last=True))
        # layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        x = self.layers(coords)
        return x


class FMappingConv(nn.Module):
    def __init__(self, in_c=1, out_c=512, nfc=32, min_nfc=32, num_layers=5):
        super(FMappingConv, self).__init__()

        N = nfc
        self.head = ConvBlock(in_c, N, 3, 1, 1)

        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Conv2d(max(N, min_nfc), out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return nn.Tanh()(x)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=pad)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


class Discriminator(nn.Module):
    def __init__(self, in_c=3, nfc=32, min_nfc=32, num_layers=5):
        super(Discriminator, self).__init__()

        N = nfc
        self.head = ConvBlock(in_c, N, 3, 1, 0)

        self.body = nn.Sequential()
        for i in range(num_layers - 2):
            N = int(nfc / pow(2, (i + 1)))
            block = ConvBlock(max(2 * N, min_nfc), max(N, min_nfc), 3, 1, 0)
            self.body.add_module('block%d' % (i + 1), block)

        # WGAN-GP discriminator has no activation at last layer
        self.tail = nn.Conv2d(max(N, min_nfc), 1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = MappingConv().to(device)
    print(d)
    summary(d, (2, 256, 160))
