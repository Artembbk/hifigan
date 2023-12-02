from torch import nn
import torch.nn.functional as F
import torch

class SubMPD(nn.Module):
    def __init__(self, p):
        super(SubMPD, self).__init__()
        self.p = p

        self.leaky_relu = nn.LeakyReLU()
        self.layers = []
        channels = 1
        for l in range(1, 5):
            self.layers.append(nn.Conv2d(channels, 2**(5+l), stride=(3,1), kernel_size=(5, 1)))
            self.layers.append(self.leaky_relu)
            channels = 2**(5+l)

        self.layers.append(nn.Conv2d(2**(5+l), 1024, kernel_size=(5, 1)))
        self.layers.append(self.leaky_relu)
        self.layers.append(nn.Conv2d(1024, 1, kernel_size=(3, 1)))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):

        b, c, t = x.shape
        if t % self.p != 0:
            n_pad = self.p - (t % self.p)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.p, self.p)

        x = self.layers(x)

        x = torch.flatten(x, 1, -1)

        return x

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
        for discriminator in self.mpds:
            outs.append(discriminator(x))
        return outs



