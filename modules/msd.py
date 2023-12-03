from torch import nn
import torch
from torch.nn.utils import weight_norm

class SubMSD(nn.Module):
    def __init__(self):
        super(SubMSD, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.last_conv = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        

    def forward(self, x_real, x_gen):
        fmap_loss = 0
        disc_loss = 0
        gen_loss = 0
        for conv in self.convs:
            x_real = self.leaky_relu(conv(x_real))
            x_gen = self.leaky_relu(conv(x_gen))
            fmap_loss += torch.mean(torch.abs(x_real - x_gen))

        x_real = self.last_conv(x_real)
        x_gen = self.last_conv(x_gen)
        fmap_loss += torch.mean(torch.abs(x_real - x_gen))

        x_real = torch.flatten(x_real, 1, -1)
        x_gen = torch.flatten(x_gen, 1, -1)

        generated_loss_1 = torch.mean((1 - x_gen)**2)
        real_loss_1 = torch.mean(x_real**2)
        gen_loss += real_loss_1 + generated_loss_1

        real_loss = torch.mean((1 - x_real)**2)
        generated_loss = torch.mean(x_gen**2)
        disc_loss += real_loss + generated_loss

        return fmap_loss, disc_loss, gen_loss


class MSD(torch.nn.Module):
    def __init__(self):
        super(MSD, self).__init__()
        self.discriminators = nn.ModuleList([
            SubMSD(),
            SubMSD(),
            SubMSD(),
        ])
        self.avgpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x_real, x_gen):
        fmap_loss = 0
        gen_loss = 0
        disc_loss = 0
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x_real = self.avgpools[i-1](x_real)
                x_gen = self.avgpools[i-1](x_gen)
            sub_fmap_loss, sub_disc_loss, sub_gen_loss = d(x_real, x_gen)
            fmap_loss += sub_fmap_loss
            gen_loss += sub_gen_loss
            disc_loss += sub_disc_loss

        return fmap_loss, disc_loss, gen_loss

