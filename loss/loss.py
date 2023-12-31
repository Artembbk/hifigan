import torch
import torch.nn.functional as F

def gan_loss(r_disc_outs, g_disc_outs):
    real_loss = 0
    generated_loss = 0
    loss = 0
    for r_disc_out, g_disc_out in zip(r_disc_outs, g_disc_outs):
        real_loss += torch.mean((1 - r_disc_out)**2)
        generated_loss += torch.mean(g_disc_out**2)
        loss += real_loss + generated_loss

    return loss
    

def mel_loss(real_mel, generated_mel):
    return F.l1_loss(real_mel, generated_mel)

def feature_matching_loss(feature_maps_generated, feature_maps_real):
    loss = 0
    for fmap_gen, fmap_real in zip(feature_maps_generated, feature_maps_real):
        loss += torch.mean(torch.abs(fmap_gen - fmap_real))

    return loss

class GeneratedLoss():
    def __init__(self) -> None:
        pass

    def __call__(self, adv_loss, fm_loss, mel_loss):
        return adv_loss + 2*fm_loss + 45*mel_loss


