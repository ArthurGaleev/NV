from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class Resblock1(nn.Module):
    def __init__(self, input_dim: int, k_r: int, dilation_r: List[int]):
        super().__init__()

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        input_dim,
                        input_dim,
                        kernel_size=k_r,
                        dilation=d_r,
                        padding=(d_r * (k_r - 1)) // 2,
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        input_dim,
                        input_dim,
                        kernel_size=k_r,
                        dilation=1,
                        padding=(k_r - 1) // 2,
                    ),
                )
                for d_r in dilation_r
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = x + conv_block(x)
        return x


class MRF(nn.Module):
    def __init__(self, input_dim: int, kernel_r: List[int], dilation_r: List[int]):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [Resblock1(input_dim, k_r, dilation_r) for k_r in kernel_r]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = x + res_block(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        hidden_u: int = 512,
        kernel_u: List[int] = [16, 16, 4, 4],
        kernel_r: List[int] = [3, 7, 11],
        dilation_r: List[int] = [1, 3, 5],
    ):
        super().__init__()

        # 7x1 Conv
        self.conv_start = nn.Conv1d(
            input_dim, hidden_u, kernel_size=7, stride=1, padding=3
        )

        self.convt_mrf_blokcs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        in_channels=hidden_u // (2**l),
                        out_channels=hidden_u // (2 ** (l + 1)),
                        kernel_size=k_u,
                        stride=k_u // 2,
                        padding=(k_u - (k_u // 2)) // 2,
                    ),
                    MRF(hidden_u // (2 ** (l + 1)), kernel_r, dilation_r),
                )
                for l, k_u in enumerate(kernel_u)
            ]
        )

        # 7x1 Conv
        self.conv_end = nn.Conv1d(
            hidden_u // (2 ** (len(kernel_u))), 1, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_start(x)
        x = self.convt_mrf_blokcs(x)
        x = F.tanh(self.conv_end(F.leaky_relu(x, 0.1)))
        return x


class SubMPD(nn.Module):
    def __init__(self, period: int, norm_func: nn.Module = weight_norm):
        super().__init__()
        self.period = period

        # 5x1 Convs
        self.convs = nn.ModuleList(
            [
                norm_func(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1))),
                norm_func(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1))),
                norm_func(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1))),
                norm_func(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1))),
                norm_func(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=1)),
            ]
        )

        # 3x1 Conv
        self.conv_end = norm_func(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # pad and reshape
        b, c, t = x.shape
        if t % self.period != 0:
            pad_amount = self.period - (t % self.period)
            t = t + pad_amount
            x = F.pad(x, (0, pad_amount), "constant")
        x = x.view(b, c, t // self.period, self.period)

        layer_features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            layer_features.append(x)

        x = self.conv_end(x)
        layer_features.append(x)

        return torch.flatten(x, 1, -1), layer_features


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([SubMPD(p) for p in [2, 3, 5, 7, 11]])

    def forward(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
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
    def __init__(self, norm_func: nn.Module = weight_norm):
        super().__init__()

        # 15x1 Conv + 41x1 Convs + 5x1 Conv from MelGAN paper https://arxiv.org/abs/1910.06711
        # with increased layers amount and decreased stride as  mentioned in HiFi-GAN paper
        # padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [
                norm_func(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                norm_func(
                    nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)
                ),
                # norm_func(
                #     nn.Conv1d(32, 64, kernel_size=41, stride=2, groups=8, padding=20)
                # ),
                norm_func(
                    nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)
                ),
                # norm_func(
                #     nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=32, padding=20)
                # ),
                norm_func(
                    nn.Conv1d(
                        256, 1024, kernel_size=41, stride=4, groups=64, padding=20
                    )
                ),
                # norm_func(
                #     nn.Conv1d(
                #         512, 1024, kernel_size=41, stride=2, groups=128, padding=20
                #     )
                # ),
                norm_func(
                    nn.Conv1d(
                        1024, 1024, kernel_size=41, stride=4, groups=256, padding=20
                    )
                ),
                norm_func(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
            ]
        )

        self.conv_end = norm_func(
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        layer_features = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            layer_features.append(x)

        x = self.conv_end(x)
        layer_features.append(x)

        return torch.flatten(x, 1, -1), layer_features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                SubMSD(norm_func=spectral_norm),
                SubMSD(),
                SubMSD(),
            ]
        )

        # x2, then x4 average pools
        self.avg_pools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        x_rs = []
        x_fs = []
        layer_features_rs = []
        layer_features_fs = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                x_real = self.avg_pools[i - 1](x_real)
                x_fake = self.avg_pools[i - 1](x_fake)

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
        dilation_r: List[int] = [1, 3, 5],
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
            dilation_r=dilation_r,
        )

        # Discriminators
        self.mpd = MPD()
        self.msd = MSD()

        self.get_mel_spectrogram = MelSpectrogram(MelSpectrogramConfig)

    def forward(
        self,
        mel_spectrogram_real: torch.Tensor,
        first_stage: bool = None,
        audio_real: torch.Tensor = None,
        audio_fake: torch.Tensor = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Model forward method.

        Args:
            audio_real (Tensor): real audio.
            mel_spectrogram_real (Tensor): its real mel spectrogram.
            first_stage (bool): determines the stage of the training, e.g. None stage (run generator), first stage (fix G, upd D), second stage (upd G).
            audio_fake (Tensor): fake audio for the first&second stage, need to pass from the None stage.
        Returns:
            output (dict): output dict containing new generated(fake) audio, its mel spectrogram, mpd and msd output with layer features (real/fake).
        """

        # runs Generator
        if first_stage is None:
            # Generator
            # if audio_real is not None:
            #     audio_fake = self.generator(mel_spectrogram_real).squeeze(1)[
            #         :, : audio_real.shape[-1]
            #     ]  # adjust fake to real length, this would only do smth if real audio wasn't of the power of 2, e.g. in train with cropped to 8192 audio the fake audio would already have the same 8192 length
            # else:
            #     # if audio_real is None, just return the full audio_fake
            audio_fake = self.generator(mel_spectrogram_real).squeeze(1)

            mel_spectrogram_fake = self.get_mel_spectrogram(audio_fake)

            return {
                "audio_fake": audio_fake,
                "mel_spectrogram_fake": mel_spectrogram_fake,
            }

        audio_real = audio_real.unsqueeze(1)
        audio_fake = audio_fake.unsqueeze(1)

        # runs Discriminator with fixed generated(fake) audio
        if first_stage:
            # Discriminators

            # MPD
            x_mpd_real, ftrs_mpd_real, x_mpd_fake, ftrs_mpd_fake = self.mpd(
                audio_real, audio_fake
            )

            # MPD
            x_msd_real, ftrs_msd_real, x_msd_fake, ftrs_msd_fake = self.msd(
                audio_real, audio_fake
            )

            return {
                "mpd": x_mpd_real,
                "mpd_fake": x_mpd_fake,
                "msd": x_msd_real,
                "msd_fake": x_msd_fake,
            }

        # otherwise runs only Discriminator
        # Discriminators

        # MPD
        x_mpd_real, ftrs_mpd_real, x_mpd_fake, ftrs_mpd_fake = self.mpd(
            audio_real, audio_fake
        )

        # MPD
        x_msd_real, ftrs_msd_real, x_msd_fake, ftrs_msd_fake = self.msd(
            audio_real, audio_fake
        )

        return {
            "mpd": x_mpd_real,
            "mpd_features": ftrs_mpd_real,
            "mpd_fake": x_mpd_fake,
            "mpd_features_fake": ftrs_mpd_fake,
            "msd": x_msd_real,
            "msd_features": ftrs_msd_real,
            "msd_fake": x_msd_fake,
            "msd_features_fake": ftrs_msd_fake,
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
