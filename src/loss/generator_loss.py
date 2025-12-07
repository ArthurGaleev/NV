import torch
from torch import nn
from typing import List
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    """
    HiFiGAN generator loss function from original paper, e.g.:
    Generator Loss = GAN generator Loss + 2 * Mel-Spectrogram Loss + 45 * Feature Matching Loss
    """

    def __init__(self):
        super().__init__()

    def forward(
            self, 
            mel_spectrogram: torch.Tensor,
            mel_spectrogram_fake: torch.Tensor,
            mpd_features: List[torch.Tensor],
            mpd_fake: torch.Tensor,
            mpd_features_fake: List[torch.Tensor],
            msd_features: List[torch.Tensor],
            msd_fake: List[torch.Tensor],
            msd_features_fake: List[torch.Tensor],
            **batch
        ):
        """
        Args:
            mel_spectrogram (Tensor): mel spectrogram for real audio
            mel_spectrogram_fake (Tensor): mel spectrogram for generated(fake) audio.
            mpd (Tensor): mpd output tensor for real audio.
            mpd_features (List[Tensor]): mpd output features for each layer for real audio.
            mpd_fake (Tensor): mpd output tensor for generated(fake) audio.
            mpd_features_fake (List[Tensor]): mpd output features for each layer for generated(fake) audio.
            msd (Tensor): mpd output tensor for real audio.
            msd_features (List[Tensor]): msd output features for each layer for real audio.
            msd_fake (Tensor): msd output tensor for generated(fake) audio.
            msd_features_fake (List[Tensor]): msd output features for each layer for generated(fake) audio.
        Returns:
            losses (dict): dict containing Generator Loss, and GAN, Mel-Spectrogram, Feature Matching generator Loss.
        """
        # GAN Loss

        # MPD
        loss_gan_mpd = 0
        for fake in mpd_fake:
            loss_gan_mpd += torch.mean((1 - fake)**2)
        
        # MSD
        loss_gan_msd = 0
        for fake in msd_fake:
            loss_gan_msd += torch.mean((1 - fake)**2)
        
        loss_gan = loss_gan_mpd + loss_gan_msd


        # Mel-Spectrogram Loss
        loss_mel_spec = F.l1_loss(mel_spectrogram, mel_spectrogram_fake)


        # Feature Matching Loss

        # MPD
        loss_ftr_match_mpd = 0
        for reals, fakes in zip(mpd_features, mpd_features_fake):
            for real, fake in zip(reals, fakes):
                loss_ftr_match_mpd += torch.mean(torch.abs(real - fake))

        # MSD
        loss_ftr_match_msd = 0
        for reals, fakes in zip(msd_features, msd_features_fake):
            for real, fake in zip(reals, fakes):
                loss_ftr_match_msd += torch.mean(torch.abs(real - fake))

        loss_ftr_match = loss_ftr_match_mpd + loss_ftr_match_msd


        # Generator Loss
        loss = loss_gan + 45 * loss_mel_spec + 2 * loss_ftr_match

        return {
            "loss": loss,
            "loss_gan_g": loss_gan,
            "loss_mel_spec_g": loss_mel_spec,
            "loss_ftr_match_g": loss_ftr_match
        }
