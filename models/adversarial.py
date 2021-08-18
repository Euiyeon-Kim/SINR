from torch import nn


class MappingNet(nn.Module):
    def __init__(self, in_f, out_f, hidden_node=128, depth=3):
        super(MappingNet, self).__init__()
        layers = [nn.Linear(in_f, hidden_node), nn.LeakyReLU(0.2)]
        for _ in range(1, depth - 1):
            layers.append(nn.Linear(hidden_node, hidden_node))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_node, out_f))
        layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
    d = Discriminator().to(device)
    summary(d, (3, 256, 256))
    print(d)