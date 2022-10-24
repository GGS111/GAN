import torch
import torch.nn as nn
from torchsummary import summary


class Discriminator(nn.Module):
    
    def __init__(self, channels, feature_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels, feature_d, kernel_size=4, stride=2, padding=1),
            self._block(feature_d, feature_d*2, 4, 2, 1),
            self._block(feature_d*2, feature_d*4, 4, 2, 1),
            self._block(feature_d*4, feature_d*8, 4, 2, 1),
            nn.Conv2d(feature_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
    

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0), # N x 1024 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # N x 512 x 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # N x 256 x 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # N x 128 x 32 x 32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1), # N x 3 x 64 x 64
            nn.Tanh(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(0.2)
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# model = Discriminator(3, 64)
# summary(model, (3, 64, 64))

# model = Generator(100, 3, 64)
# summary(model, (100, 1, 1))