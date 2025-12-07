import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from typing import Tuple, List, Dict, Union
from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class Resblock1(nn.Module):
    def __init__(self, channels: int, k_r: int, dilation_r: List[int]):
        super().__init__()

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Conv1d(channels, channels, kernel_size=k_r, dilation=d_r, padding=(d_r*(k_r-1))//2),
                    nn.LeakyReLU(),
                    nn.Conv1d(channels, channels, kernel_size=k_r, dilation=1, padding=(k_r-1)//2)
                )
                for d_r in dilation_r
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x += conv_block(x)
        return x


class MRF(nn.Module):
    def __init__(
            self, 
            channels: int, 
            kernel_r: List[int],
            dilation_r: List[int]
        ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                Resblock1(channels, k_r, dilation_r)
                for k_r in kernel_r
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x += res_block(x)
        return x

class Generator(nn.Module):
    def __init__(
            self, 
            input_dim: int = 80,
            hidden_u: int = 512,
            kernel_u: List[int] = [16, 16, 4, 4],
            kernel_r: List[int] = [3, 7, 11],
            dilation_r: List[int] = [1, 3, 5]
        ):
        super().__init__()

        # 7x1 Conv
        self.conv_start = nn.Conv1d(input_dim, hidden_u, kernel_size=7, stride=1)
        
        self.convt_mrf_blokcs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(
                        in_channels=hidden_u//(2**l), 
                        out_channels=hidden_u//(2**(l+1)), 
                        kernel_size=k_u, 
                        stride=k_u//2
                    ),
                    MRF(hidden_u//(2**(l+1)), kernel_r, dilation_r)
                )
                for l, k_u in enumerate(kernel_u)
            ]
        )

        # 7x1 Conv
        self.conv_end = nn.Conv1d(hidden_u//(2**(len(kernel_u))), 1, kernel_size=7, stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_start(x)
        x = self.convt_mrf_blokcs(x)
        x = F.tanh(self.conv_end(F.leaky_relu(x)))
        return x


class SubMPD(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        norm_f = nn.utils.parametrizations.weight_norm

        # 5x1 Convs
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1))),
                norm_f(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1))),
                norm_f(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1))),
                norm_f(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1))),
                norm_f(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=1))
            ]
        )
        
        # 3x1 Conv
        self.conv_end = norm_f(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # pad and reshape
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            t = t + n_pad
            x = F.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, t // self.period, self.period)

        layer_features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x))
            layer_features.append(x)

        x = self.conv_end(x)
        layer_features.append(x)

        return torch.flatten(x, 1, -1), layer_features


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SubMPD(p)
                for p in [2, 3, 5, 7, 11]
            ]
        )

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        x_rs = []
        x_fs = []
        layer_features_rs = []
        layer_features_fs = []
        for discriminator in self.discriminators:
            x_r, layer_feature_r = discriminator(x_real)
            x_rs.append(x_r)
            layer_features_rs.append(layer_feature_r)

            x_f, layer_feature_f = discriminator(x_fake)
            x_fs.append(x_f)
            layer_features_fs.append(layer_feature_f)

        return x_rs, layer_features_rs, x_fs, layer_features_fs


class SubMSD(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_f = nn.utils.parametrizations.weight_norm if use_spectral_norm == False else nn.utils.parametrizations.spectral_norm

        # 15x1 Conv + 41x1 Convs + 5x1 Conv from MelGAN paper https://arxiv.org/abs/1910.06711
        # with increased layers amount and decreased stride as  mentioned in HiFi-GAN paper
        # padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                norm_f(nn.Conv1d(16, 32, kernel_size=41, stride=2, groups=4, padding=20)),
                norm_f(nn.Conv1d(32, 64, kernel_size=41, stride=2, groups=8, padding=20)),
                norm_f(nn.Conv1d(64, 128, kernel_size=41, stride=2, groups=16, padding=20)),
                norm_f(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=32, padding=20)),
                norm_f(nn.Conv1d(256, 512, kernel_size=41, stride=2, groups=64, padding=20)),
                norm_f(nn.Conv1d(512, 1024, kernel_size=41, stride=2, groups=128, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, kernel_size=41, stride=2, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            ]
        )
        
        self.conv_end = norm_f(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        layer_features = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x)
            layer_features.append(x)

        x = self.conv_end(x)
        layer_features.append(x)

        return torch.flatten(x, 1, -1), layer_features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SubMSD(use_spectral_norm=True),
                SubMSD(),
                SubMSD(),
            ]
        )

        # x2, then x4 average pools
        self.avg_pools = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2)
            ]
        )

    def forward(self, x_real: torch.Tensor, x_fake: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        x_rs = []
        x_fs = []
        layer_features_rs = []
        layer_features_fs = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                x_real = self.avg_pools[i-1](x_real)
                x_fake = self.avg_pools[i-1](x_fake)

            x_r, layer_feature_r = discriminator(x_real)
            x_rs.append(x_r)
            layer_features_rs.append(layer_feature_r)

            x_f, layer_feature_f = discriminator(x_fake)
            x_fs.append(x_f)
            layer_features_fs.append(layer_feature_f)

        return x_rs, layer_features_rs, x_fs, layer_features_fs


class HiFiGAN(nn.Module):
    """
    HiFi-GAN implementation from https://arxiv.org/abs/2010.05646
    """

    def __init__(
            self,
            hidden_u: int = 512,
            kernel_u: List[int] = [16, 16, 4, 4],
            kernel_r: List[int] = [3, 7, 11],
            dilation_r: List[int] = [1, 3, 5]
        ):
        """
        Args:
            hidden_u (int): hidden size of generator.
            kernel_u (List[int]): list of kernel sizes of the transposed convolutions in generator.
            kernel_r (List[int]): list of kernel sizes of the convolutions in ResBlocks in MRF's generator.
            dilation_r (List[int]): list of dilation rates of the convolutions in ResBlocks in MRF's generator.
        """
        super().__init__()

        # Generator
        self.generator = Generator(
            input_dim=80, 
            hidden_u=hidden_u,
            kernel_u=kernel_u,
            kernel_r=kernel_r,
            dilation_r=dilation_r
        )

        # Discriminators
        self.mpd = MPD()
        self.msd = MSD()

        self.get_mel_spectrogram = MelSpectrogram(MelSpectrogramConfig)

    def forward(
            self, 
            audio_real: torch.Tensor, 
            mel_spectrogram_real: torch.Tensor
        ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Model forward method.

        Args:
            audio_real (Tensor): real audio.
            mel_spectrogram_real (Tensor): its real mel spectrogram.
        Returns:
            output (dict): output dict containing new generated audio, its mel spectrogram, mpd and msd output with layer features.
        """

        # Generator
        audio_fake = self.generator(mel_spectrogram_real)
        mel_spectrogram_fake = self.get_mel_spectrogram(audio_fake.squeeze(1))

        # Discriminators
        audio_real = audio_real.unsqueeze(1)
        # MPD
        x_real_mpd, ftrs_real_mpd, x_fake_mpd, ftrs_fake_mpd = self.mpd(audio_real, audio_fake)

        # MPD
        x_real_msd, ftrs_real_msd, x_fake_msd, ftrs_fake_msd = self.msd(audio_real, audio_fake)

        return {
            "audio_new": audio_fake,
            "mel_spectrogram_new": mel_spectrogram_fake,
            "mpd": x_real_mpd,
            "mpd_features": ftrs_real_mpd,
            "msd": x_real_msd,
            "msd_features": ftrs_real_msd,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
