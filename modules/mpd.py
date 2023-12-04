from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
from utils import get_padding

class SubMPD(nn.Module):
    def __init__(self, p):
        super(SubMPD, self).__init__()
        self.p = p

        self.leaky_relu = nn.LeakyReLU()
        self.layers = []
        channels = 1
        for l in range(1, 5):
            self.layers.append(weight_norm(nn.Conv2d(channels, 2**(5+l), stride=(3,1), kernel_size=(5, 1))), padding=(get_padding(5, 1), 0))
            channels = 2**(5+l)

        self.post_convs = []
        self.post_convs.append(weight_norm(nn.Conv2d(2**(5+l), 1024, kernel_size=(5, 1))), padding=(2, 0))
        self.post_convs.append(weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1))), padding=(1,0))

        self.layers = nn.ModuleList(self.layers)
        self.post_convs = nn.ModuleList(self.post_convs)

    def forward(self, x_real, x_gen):
        fmap_loss = 0
        disc_loss = 0
        gen_loss = 0

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
        disc_loss += real_loss + generated_loss

        generated_loss_1 = torch.mean((1 - x_gen)**2)
        real_loss_1 = torch.mean(x_real**2)
        gen_loss += real_loss_1 + generated_loss_1

        return fmap_loss, disc_loss, gen_loss

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
        disc_loss = 0
        gen_loss = 0
        for discriminator in self.mpds:
            sub_fmap_loss, sub_disc_loss, sub_gen_loss = discriminator(x_real, x_gen)
            fmap_loss += sub_fmap_loss
            gen_loss += sub_gen_loss
            disc_loss += sub_disc_loss

        return fmap_loss, disc_loss, gen_loss



