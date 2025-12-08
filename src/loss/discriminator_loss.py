from typing import List

import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    """
    HiFiGAN discriminator loss function from original paper, e.g.:
    Discriminator Loss = GAN discriminator Loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        mpd: torch.Tensor,
        mpd_fake: torch.Tensor,
        msd: torch.Tensor,
        msd_fake: torch.Tensor,
        **batch
    ):
        """
        Args:
            mpd (Tensor): mpd output tensor for real audio.
            mpd_fake (Tensor): mpd output tensor for generated(fake) audio.
            msd (Tensor): mpd output tensor for real audio.
            msd_fake (Tensor): msd output tensor for generated(fake) audio.
        Returns:
            losses (dict): dict containing Discriminator Loss.
        """

        # MPD
        loss_mpd = 0
        for real, fake in zip(mpd, mpd_fake):
            loss_mpd += torch.mean((1 - real) ** 2) + torch.mean(fake**2)

        # MSD
        loss_msd = 0
        for real, fake in zip(msd, msd_fake):
            loss_msd += torch.mean((1 - real) ** 2) + torch.mean(fake**2)

        return {"loss_d": loss_mpd + loss_msd}
