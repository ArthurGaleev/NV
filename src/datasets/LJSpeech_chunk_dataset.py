import os

import kaggle
import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LJSpeechChunkDataset(BaseDataset):
    """
    LJSpeechDataset chunked by auidio_len.
    """

    def __init__(self, audio_len=8192, sample_rate=22050, *args, **kwargs):
        """
        Args:
            audio_len (int): chunks length.
            sample_rate (int): audio sample rate.
        """
        path_dir = ROOT_PATH / "data" / "datasets"
        data_dir = path_dir / "LJSpeech-1.1"
        audio_dir = data_dir / "wavs"
        audio_crop_dir = data_dir / f"wavs_crop-{audio_len}"

        if not data_dir.exists():
            torchaudio.datasets.LJSPEECH(root=path_dir, download=True)

        data = []
        for audio_path in list(audio_dir.iterdir()):
            data.append(audio_path)

        if not audio_crop_dir.exists():
            audio_crop_dir.mkdir(exist_ok=True, parents=True)

            for audio_path in tqdm(
                data, total=len(data), desc="Making audio crops for the dataset"
            ):
                audio, sr = torchaudio.load(audio_path)
                for i, audio_start in enumerate(
                    range(0, audio.shape[1] - audio_len + 1, audio_len)
                ):
                    audio_crop_path = audio_crop_dir / (
                        audio_path.stem + f"-{i}" + ".wav"
                    )
                    torchaudio.save(
                        audio_crop_path,
                        audio[:, audio_start : audio_start + audio_len],
                        sample_rate,
                    )

        data = []
        for audio_path in tqdm(
            list(audio_crop_dir.iterdir()), desc="Creating train dataset"
        ):
            data.append({"audio_path": str(audio_path)})

        super().__init__(data, *args, **kwargs)
