import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=True,
                                )
        self.scale = (gain / (in_channels * kernel_size * kernel_size)) ** 0.5
        self.bias = self.conv.bias #Copy
        self.conv.bias = None 

        #Initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixel_norm

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x    
        return x   

class Generator(nn.Module):

    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )
 
        self.last_rgb_layer = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_block, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.last_rgb_layer])

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])

            self.prog_block.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x) #init 1x1 --> 4x4

        if steps == 0:
            return self.last_rgb_layer(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='nearest') #upscale x2
            out = self.prog_block[step](upscaled) #Use ConvBlock from ModuleList

        #Block for fade_in
        final_upscaled = self.rgb_layers[steps - 1](upscaled) 
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):

    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0) #1st layer
        self.rgb_layers.append(self.initial_rgb) #add to last block
        self.average_pool = nn.AvgPool2d(kernel_size=2, stride=2) #Downsapmling

        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),#4x4 --> 1x1
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps #Take current step to use initial rgb layer
        out = self.leaky(self.rgb_layers[cur_step](x)) #use initial rgb layer

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1) #from 4 dimensions to 2 dim

        #fade in block
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.average_pool(x)))
        out = self.average_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        #if a lot of steps we downsampling images with 2xConv+Avg_pool
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out) #2x Conv
            out = self.average_pool(out) #Downsampling

        out = self.minibatch_std(out)
        return self.final_block(out).view(x.shape[0], -1)

def test_1():
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=3)
    critic = Discriminator(IN_CHANNELS, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4)) #Take log2 for (2 ** img_size / 4) --> 0, 1, 2, 3, 4, 5, 6, 7, 8
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, alpha=0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img size: {img_size}")

def test_2():
    model = Generator(100, 512, img_channels=3)
    summary(model, (100, 1, 1), 0.5, 8)

def test_3():
    model = Discriminator(512, img_channels=3)
    summary(model, (3, 1024, 1024), 0.5, 8)

if __name__ == "__main__":
    test_1()