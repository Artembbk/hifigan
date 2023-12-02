from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm

class SubMPD(nn.Module):
    def __init__(self, p):
        super(SubMPD, self).__init__()
        self.p = p

        self.leaky_relu = nn.LeakyReLU()
        self.layers = []
        channels = 1
        for l in range(1, 5):
            self.layers.append(weight_norm(nn.Conv2d(channels, 2**(5+l), stride=(3,1), kernel_size=(5, 1))))
            channels = 2**(5+l)

        self.post_convs = []
        self.post_convs.append(weight_norm(nn.Conv2d(2**(5+l), 1024, kernel_size=(5, 1))))
        self.post_convs.append(weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1))))

        self.layers = nn.ModuleList(self.layers)
        self.post_convs = nn.ModuleList(self.post_convs)

    def forward(self, x):
        feature_maps = []
        b, c, t = x.shape
        if t % self.p != 0:
            n_pad = self.p - (t % self.p)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.p, self.p)

        for layer in self.layers:
            x = self.leaky_relu(layer(x))
            feature_maps.append(x)

        for post_conv in self.post_convs:
            x = self.leaky_relu(post_conv(x))
            feature_maps.append(x)

        x = torch.flatten(x, 1, -1)

        return x, feature_maps

class MPD(nn.Module):
    def __init__(self, ps):
        super(MPD, self).__init__()

        self.ps = ps
        self.mpds = []
        for p in ps:
            self.mpds.append(SubMPD(p))

        self.mpds = nn.ModuleList(self.mpds)

    def forward(self, x):
        outs = []
        feature_maps = []
        for discriminator in self.mpds:
            x, sub_feature_maps = discriminator(x)
            outs.append(x)
            feature_maps.extend(sub_feature_maps)
        return outs, feature_maps



