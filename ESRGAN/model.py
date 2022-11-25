import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_act=True, **kwargs) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=True, **kwargs),
            nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.convblock(x)

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, scale_factor=2) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.upsample(x)

class LinearResBlock(nn.Module):

    def __init__(self, in_channels, channels=32, residual_beta=0.2) -> None:
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <=3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <=3 else False,
                )
            )

    def forward(self, x):
        new_input = x                                      #C=64
        for block in self.blocks:
            out = block(new_input)                         #C: 32, 32, 32, 32, 64
            new_input = torch.cat([new_input, out], dim=1) #C: 64,96,128,160,192
        return self.residual_beta * out + x                #0.2 * C=64 + C=64

class BasicResBlock(nn.Module):
    
    def __init__(self, in_channels, residual_beta=0.2) -> None:
        super().__init__()
        self.residual_beta = residual_beta
        self.resblock = nn.Sequential(*[LinearResBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.resblock(x) * self.residual_beta + x

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channesl):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channesl, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channesl),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channesl, out_channesl, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channesl),
            nn.LeakyReLU(0.2, inplace=True),            
        )
    
    def forward(self, x):
        return self.doubleconv(x)


class Generator(nn.Module):

    def __init__(self, img_channels=3, channels=64, num_blocks=23) -> None:
        super().__init__()

        self.initial = nn.Conv2d(img_channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.res_blocks = nn.Sequential(*[BasicResBlock(channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.upsample = nn.Sequential(UpsampleBlock(channels), UpsampleBlock(channels))
        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, img_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.res_blocks(initial)
        x = self.conv(x) + initial
        x = self.upsample(x)
        return self.final(x)

class Discriminator(nn.Module):

    def __init__(self, img_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(0.2, inplace=True), 
        )
        self.block = nn.Sequential(
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        )
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block(x)
        return self.classification(x)

def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


def test_1():
    gen = Generator()
    disc = Discriminator()
    low_res = 24
    x = torch.randn((5, 3, low_res, low_res))
    gen_out = gen(x)
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)

def test_2():
    model = Generator()
    summary(model, (3, 24, 24))

def test_3():
    model = Discriminator()
    summary(model, (3, 96, 96), depth=5)

if __name__ == "__main__":
    test_3()

