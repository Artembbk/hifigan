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
    return F.l1_loss(real_mel - generated_mel)

