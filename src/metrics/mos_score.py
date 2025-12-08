import torch

from src.metrics.base_metric import BaseMetric
from src.utils.wvmos import get_wvmos


class MOS(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        MOS score metric.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_wvmos(cuda=(device == "cuda"))

    def __call__(self, audio, **batch):
        """
        Calculate mos score by fine-tuned wav2vec2.0.

        Args:
            audio (Tensor): single/batch audio for calculating the mos score.
        Returns:
            metric (float): calculated (mean) mos score.
        """
        # convert to batch format if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mos_scores = []
        for i in range(audio.shape[0]):
            mos_scores.append(self.model.get_mos_score(audio[i]))

        return torch.tensor(mos_scores).mean().item()
