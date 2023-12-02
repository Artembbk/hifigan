from typing import Any
from torch import nn
import torch
from torch.nn.utils import weight_norm

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlock, self).__init__()

        self.dilations = dilations

        self.leaky_relu = nn.LeakyReLU()

        self.convs = []
        for m, _ in enumerate(dilations):
            for l, _ in enumerate(dilations[m]):
                padding = ((kernel_size - 1) * dilations[m][l]) // 2
                self.convs.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilations[m][l], padding=padding)))

        self.convs = nn.ModuleList(self.convs)

    def forward(self, x):
        for m, _ in enumerate(self.dilations):
            out = x
            for l, _ in enumerate(self.dilations[m]):
                out = self.leaky_relu(out)
                out = self.convs[len(self.dilations[m])*m + l](out)
            x = x + out
        return x

class MRF(nn.Module):
    def __init__(self, channels, kr, dilations):
        super(MRF, self).__init__()

        self.res_blocks = []

        for i, kernel_size in enumerate(kr):
            self.res_blocks.append(ResBlock(channels, kernel_size, dilations[i]))

        self.res_blocks = nn.ModuleList(self.res_blocks)

    def forward(self, x):
        out = torch.empty_like(x).to(x.device)
        for res_block in self.res_blocks:
            x = res_block(x)
            out = out + x
        
        return out

class Generator(nn.Module):
    def __init__(self, n_mels, h_init, ku, kr, dilations):
        super(Generator, self).__init__()

        self.init_conv = weight_norm(nn.Conv1d(n_mels, h_init, 7, dilation=1, padding=3))
        self.leaky_relu = nn.LeakyReLU()
        self.layers = []
        in_channels = h_init
        for i, kernel_size in enumerate(ku):
            self.layers.append(self.leaky_relu)
            out_channels = in_channels // 2
            self.layers.append(weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                                    stride=kernel_size//2, padding=kernel_size//4)))
            in_channels = out_channels

            self.layers.append(MRF(in_channels, kr, dilations))

        self.layers = nn.Sequential(*self.layers)

        self.out_conv = weight_norm(nn.Conv1d(in_channels, 1, 7, padding=3))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layers(x)
        x = self.leaky_relu(x)
        x = self.out_conv(x)
        x = self.tanh(x)
        return x




