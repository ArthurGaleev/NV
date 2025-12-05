import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
import os


class LJSpeechDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        wavs_dir = ROOT_PATH / "data" / "datasets" / "LJSpeech" / "wavs"

        assert wavs_dir.exists(), "to use LJSpeech as dataset, place it's wavs into wavs folder in ./data/datasets/LJSpeech"

        data = []
        for wav_path in tqdm(list(wavs_dir.iterdir()), desc="Creating dataset"):
            data.append({"audio_path": str(wav_path)})

        super().__init__(data, *args, **kwargs)
