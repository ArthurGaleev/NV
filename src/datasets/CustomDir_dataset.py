from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from typing import List, Union


class CustomDirDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        data: List[dict] = None,
        *args,
        **kwargs,
    ):
        path = Path(path)
        assert path.exists(), "CustomDirDataset path is not valid"
        ts_path = path / "transcriptions"

        if data == None:
            data = []
            for transcription_path in tqdm(list((ts_path / "transcriptions").iterdir()), desc="Creating custom dataset"):
                data.append({"transcription": transcription_path.read_text()})
        else:
            assert "audio_path" in data[0].keys(), "Invalid data passed to CustomDirDataset"
            for i in tqdm(range(len(data)), desc="Creating test dataset"):
                data[i]["transcription"] = ts_path / (Path(data[i]["audio_path"]).stem + ".txt")

        super().__init__(data, *args, **kwargs)