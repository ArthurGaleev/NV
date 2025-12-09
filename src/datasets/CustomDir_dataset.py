import csv
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchaudio
from speechbrain.inference.TTS import Tacotron2
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        data: List[dict] = None,
        use_pretrained_text2mel=False,
        limit=None,
        shuffle_index=False,
        *args,
        **kwargs,
    ):
        path = Path(path)
        assert path.exists(), "CustomDirDataset path is not valid"
        transcription_dir = path / "transcriptions"

        if data is None or not data:
            assert (
                transcription_dir.exists() or (path / "metadata.csv").exists()
            ), "nor audio, nor transcription dir, nor metadata.csv file found in dataset, you need to add at least one of them for synthesize"

        if use_pretrained_text2mel:
            if not transcription_dir.exists():
                if (path / "metadata.csv").exists():
                    print("Parsing metadata.csv into transcriptions folder...")
                    transcription_dir.mkdir(parents=True, exist_ok=True)
                    with open(path / "metadata.csv", mode="r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.split("|")
                            if len(parts) == 3:
                                output_file_path = transcription_dir / (
                                    parts[0] + ".txt"
                                )
                                output_file_path.write_text(parts[2], encoding="utf-8")
                else:
                    raise RuntimeError(
                        "nor transcriptions folder, nor metadata.csv file found, add at least one of them for use_text2mel=True"
                    )

            # Intialize TTS (tacotron2)
            tacotron2 = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech"
            )

            transcription_paths = list(transcription_dir.iterdir())

            if data is None or not data:
                if shuffle_index:
                    random.seed(42)
                    random.shuffle(transcription_paths)
                if limit is not None:
                    transcription_paths = transcription_paths[:limit]

                data = []
                for transcription_path in tqdm(
                    list(transcription_dir.iterdir())[:limit],
                    desc="Creating custom dataset with pre-trained text2mel",
                ):
                    # Running the TTS
                    mel_output, mel_length, alignment = tacotron2.encode_text(
                        transcription_path.read_text()
                    )
                    data.append(
                        {
                            "mel_spectrogram": mel_output,
                            "text_path": str(transcription_path),
                        }
                    )
            else:
                if shuffle_index:
                    random.seed(42)
                    random.shuffle(data)
                if limit is not None:
                    data = data[:limit]

                for i in tqdm(
                    list(range(len(data))),
                    desc="Creating custom dataset with pre-trained text2mel",
                ):
                    # Running the TTS
                    mel_output, mel_length, alignment = tacotron2.encode_text(
                        (
                            transcription_dir
                            / (Path(data[i]["audio_path"]).stem + ".txt")
                        ).read_text()
                    )
                    data[i]["mel_spectrogram"] = mel_output

        super().__init__(
            data,
            use_pretrained_text2mel=use_pretrained_text2mel,
            limit=limit,
            shuffle_index=shuffle_index,
            *args,
            **kwargs,
        )
