import torch
import torch.nn as nn
from torchsummary import summary

class Block_down(nn.Module):
    
    def __init__(self, in_channels, out_channels, down=False):
        super(Block_down, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                padding_mode='reflect'
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.block(x)

class Block_up(nn.Module):

    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(Block_up, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    
    def __init__(self, in_channels=3, features=64):
        super(Generator, self).__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(                         #128
                in_channels,
                features,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
            ),
            nn.LeakyReLU(0.2)
        )

        self.down1 = Block_down(features, features*2) #64
        self.down2 = Block_down(features*2, features*4) #32
        self.down3 = Block_down(features*4, features*8) #16
        self.down4 = Block_down(features*8, features*8) #8
        self.down5 = Block_down(features*8, features*8) #4
        self.down6 = Block_down(features*8, features*8) #2

        self.bottlenect = nn.Sequential(                #1
            nn.Conv2d(                       
                features*8,
                features*8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU()
        )


        self.up1 = Block_up(features*8, features*8, use_dropout=True) #2
        self.up2 = Block_up(features*8*2, features*8, use_dropout=True) #4
        self.up3 = Block_up(features*8*2, features*8, use_dropout=True) #8
        self.up4 = Block_up(features*8*2, features*8, use_dropout=False) #16
        self.up5 = Block_up(features*8*2, features*4, use_dropout=False) #32
        self.up6 = Block_up(features*4*2, features*2, use_dropout=False) #64
        self.up7 = Block_up(features*2*2, features, use_dropout=False) #128

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features*2,
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh()
        )
    
    def forward(self, x):
        d1_init = self.initial_down(x)
        d1 = self.down1(d1_init)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        bottle = self.bottlenect(d6)
        up1 = self.up1(bottle)
        up2 = self.up2(torch.cat([up1, d6], dim=1))
        up3 = self.up3(torch.cat([up2, d5], dim=1))
        up4 = self.up4(torch.cat([up3, d4], dim=1))
        up5 = self.up5(torch.cat([up4, d3], dim=1))
        up6 = self.up6(torch.cat([up5, d2], dim=1))
        up7 = self.up7(torch.cat([up6, d1], dim=1))
        return self.final_up(torch.cat([up7, d1_init], dim=1))
        
def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

def test2():
    model = Generator(in_channels=3, features=64)
    summary(model, (3, 256, 256))


if __name__ == "__main__":
    test2()

