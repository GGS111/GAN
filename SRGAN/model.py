import torch
import torch.nn as nn

from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(num_parameters=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),            
        )

    def forward(self, x):
        return self.res_block(x) + x

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.up_block = nn.Sequential(
            nn.Conv2d(in_channels, 
                      in_channels * scale_factor ** 2, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU(num_parameters=in_channels)
        )

    def forward(self, x):
        return self.up_block(x)

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

    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, stride=1, padding=4),
            nn.PReLU(num_parameters=num_channels)
        )
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
            )
        self.upsamples = nn.Sequential(
            UpsampleBlock(num_channels, 2),
            UpsampleBlock(num_channels, 2)
            )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))

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

def test_1():
    low_resolution = 24  # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
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
        