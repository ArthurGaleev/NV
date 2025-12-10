import math
from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # audio_len_for_mel_spec: int = 8960


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=True,
            # center=False
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        # make the spectrogram length equal to nearest value in the power of 2
        # e.g. for 8192 audio length and mel config above, mel_frames=31 =>
        # => output spectrogram length would be 32, we will pad the audio to handle this
        # mel_frames = (audio.shape[-1] - self.config.n_fft) // self.config.hop_length + 1
        # mel_frames = 2 ** math.ceil(math.log2(mel_frames))
        # pad_amount = ((mel_frames - 1) * self.config.hop_length) - (
        #     audio.shape[-1] - self.config.n_fft
        # )

        # # pad to 8960, so mel spec size would be 32
        # if pad_amount:
        #     audio = F.pad(audio, (0, pad_amount), mode="constant")

        # mel = self.mel_spectrogram.to(audio.device)(audio)

        audio = F.pad(
            audio, 
            (
                (self.config.n_fft - self.config.hop_length) // 2,
                (self.config.n_fft - self.config.hop_length) // 2
            ),
            mode="reflect"
        )

        mel = self.mel_spectrogram.to(audio.device)(audio).clamp_(min=1e-5).log_()
        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
