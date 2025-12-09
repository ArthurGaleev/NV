import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import requests
from tqdm.auto import tqdm

from src.datasets.CustomDir_dataset import CustomDirDataset
from src.utils.io_utils import ROOT_PATH

YANDEX_URL = {
    "test_data": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    }
}


class YandexDownloadDataset(CustomDirDataset):
    def __init__(
        self,
        download_name="test_data",
        use_pretrained_text2mel=False,
        *args,
        **kwargs,
    ):
        """
        Args:
            download_name (str): dataset name.
        """
        data_dir = ROOT_PATH / "data" / "datasets"
        if not (data_dir / download_name).exists():
            download_info = YANDEX_URL[download_name]
            assert download_info[
                "public_key"
            ], "YANDEX_DISK_URL env var is not specified"

            data_dir.mkdir(exist_ok=True, parents=True)
            final_url = download_info["base_url"] + urlencode(
                dict(public_key=download_info["public_key"])
            )
            response = requests.get(final_url)
            download_url = response.json()["href"]
            print("Downloading test data...")
            download_response = requests.get(download_url)
            print("Successfully downloaded")
            zip = zipfile.ZipFile(io.BytesIO(download_response.content))
            zip.extractall(data_dir)

        data = []
        if (data_dir / download_name / "gt_audio").exists():
            audio_dir_name = "gt_audio"
        elif (data_dir / download_name / "wavs").exists():
            audio_dir_name = "wavs"
        else:
            audio_dir_name = None

        if audio_dir_name is not None:
            for audio_path in list(
                (data_dir / download_name / audio_dir_name).iterdir()
            ):
                data.append({"audio_path": str(audio_path)})

        super().__init__(
            data=data,
            path=data_dir / download_name,
            use_pretrained_text2mel=use_pretrained_text2mel,
            *args,
            **kwargs,
        )
