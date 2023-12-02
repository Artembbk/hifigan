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
        

    def forward(self, x):
        feature_maps = []
        for conv in self.convs:
            x = self.leaky_relu(conv(x))
            feature_maps.append(x)

        x = self.last_conv(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps


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
        gan_loss = 0
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x_real = self.avgpools[i-1](x_real)
                x_gen = self.avgpools[i-1](x_gen)
            x_out_real, sub_feature_maps_real = d(x_real)
            x_out_gen, sub_feature_maps_gen = d(x_gen)
            
            fmap_loss += torch.mean(torch.abs(sub_feature_maps_gen - sub_feature_maps_real))
            real_loss += torch.mean((1 - x_out_real)**2)
            generated_loss += torch.mean(x_out_gen**2)
            gan_loss += real_loss + generated_loss

        return fmap_loss, gan_loss

