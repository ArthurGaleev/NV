import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
import os
import kaggle


class LJSpeechDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, audio_len=8192, sample_rate=22050, *args, **kwargs
    ):
        """
        Args:
            name (str): partition name
        """
        path_dir = ROOT_PATH / "data" / "datasets"
        data_dir = path_dir / "LJSpeech-1.1"
        audio_dir = data_dir / "wavs"
        audio_crop_dir = data_dir / f"wavs_crop-{audio_len}"

        if not data_dir.exists():
            kaggle.api.dataset_download_files('mathurinache/the-lj-speech-dataset', path=path_dir, quiet=False, unzip=True)

        data = []
        for audio_path in list(audio_dir.iterdir()):
            data.append(audio_path)

        if not audio_crop_dir.exists():
            audio_crop_dir.mkdir(exist_ok=True, parents=True)

            for audio_path in tqdm(data, total=len(data), desc="Making audio crops for the dataset"):
                audio, sr = torchaudio.load(audio_path)
                for i, audio_start in enumerate(range(0, audio.shape[1] - audio_len + 1, audio_len)):
                    audio_crop_path = audio_crop_dir / (audio_path.stem + f"-{i}" + ".wav")
                    torchaudio.save(audio_crop_path, audio[:, audio_start:audio_start+audio_len], sample_rate)

        data = []
        for audio_path in tqdm(list(audio_crop_dir.iterdir()), desc="Creating train dataset"):
            data.append({"audio_path": str(audio_path)})

        super().__init__(data, *args, **kwargs)
