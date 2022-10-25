import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, use_act=True, down=True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding_mode='reflect',**kwargs) if down
            else nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):

    def __init__(self, in_channels=3, num_features=64, num_res_block=9):
        super().__init__()

        self.initial = ConvBlock(in_channels, num_features, kernel_size=7, stride=1, padding=3)

        self.down_block = nn.Sequential(
            ConvBlock(num_features*1, num_features*2, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
        )

        self.res_block = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_res_block)]
        )

        self.up_block = nn.Sequential(
            ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_block = ConvBlock(num_features, in_channels, down=False, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.res_block(x)
        x = self.up_block(x)
        x = self.final_block(x)
        return torch.tanh(x)

def test():
    model = Generator(in_channels=3)
    summary(model, input_data=(3, 256, 256), depth=5, col_names=['input_size', "output_size", "kernel_size", "num_params"],)

if __name__ == '__main__':
    test()