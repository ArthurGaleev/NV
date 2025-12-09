import os

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LJSpeechDataset(BaseDataset):
    """
    LJSpeechDataset.
    """

    def __init__(self, *args, **kwargs):
        path_dir = ROOT_PATH / "data" / "datasets"
        data_dir = path_dir / "LJSpeech-1.1"
        audio_dir = data_dir / "wavs"

        if not data_dir.exists():
            path_dir.mkdir(exist_ok=True, parents=True)
            torchaudio.datasets.LJSPEECH(root=path_dir, download=True)

        data = []
        for audio_path in tqdm(list(audio_dir.iterdir()), desc="Creating dataset"):
            data.append({"audio_path": str(audio_path)})

        super().__init__(data, *args, **kwargs)
