import torch

from src.metrics.base_metric import BaseMetric
from src.utils.wvmos import get_wvmos
import torchaudio
from src.transforms.mel_spectrogram import MelSpectrogramConfig


class MOS(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        MOS score metric.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_wvmos(cuda = (device=="cuda"))

    def __call__(self, audio, **batch):
        """
        Calculate mos score by fine-tuned wav2vec2.0.

        Args:
            audio (Tensor): single/batch audio for calculating the mos score.
        Returns:
            metric (float): calculated (mean) mos score.
        """
        if audio.dim == 3:
            return self.model.get_mos_score_batch(audio)
        return self.model.get_mos_score(audio)
