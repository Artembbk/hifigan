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

    def forward(self, x_real, x_gen):
        fmap_loss = 0

        b, c, t = x_real.shape
        if t % self.p != 0:
            n_pad = self.p - (t % self.p)
            x_real = F.pad(x_real, (0, n_pad), "reflect")
            x_gen = F.pad(x_gen, (0, n_pad), "reflect")
            t = t + n_pad
        x_real = x_real.view(b, c, t // self.p, self.p)
        x_gen = x_gen.view(b, c, t // self.p, self.p)

        for layer in self.layers:
            x_real = self.leaky_relu(layer(x_real))
            x_gen = self.leaky_relu(layer(x_gen))
            fmap_loss += torch.mean(torch.abs(x_real - x_gen))
            

        for post_conv in self.post_convs:
            x_real = self.leaky_relu(post_conv(x_real))
            x_gen = self.leaky_relu(post_conv(x_gen))
            fmap_loss += torch.mean(torch.abs(x_real - x_gen))

        x_real = torch.flatten(x_real, 1, -1)
        x_gen = torch.flatten(x_gen, 1, -1)

        real_loss = torch.mean((1 - x_real)**2)
        generated_loss = torch.mean(x_gen**2)
        gan_loss += real_loss + generated_loss

        return fmap_loss, gan_loss

class MPD(nn.Module):
    def __init__(self, ps):
        super(MPD, self).__init__()

        self.ps = ps
        self.mpds = []
        for p in ps:
            self.mpds.append(SubMPD(p))

        self.mpds = nn.ModuleList(self.mpds)

    def forward(self, x_real, x_gen):
        fmap_loss = 0
        gan_loss = 0
        for discriminator in self.mpds:
            sub_fmap_loss, sub_gan_loss = discriminator(x_real, x_gen)
            fmap_loss += sub_fmap_loss
            gan_loss += sub_gan_loss
        return fmap_loss, gan_loss



