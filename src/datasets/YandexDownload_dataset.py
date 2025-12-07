import io
import os
import zipfile
from urllib.parse import urlencode
from tqdm.auto import tqdm

import requests

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
        *args,
        **kwargs,
    ):
        data_dir = ROOT_PATH / "data" / "datasets"
        if not (data_dir / download_name).exists():
            data_dir.mkdir(exist_ok=True, parents=True)
            download_info = YANDEX_URL[download_name]
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
        for audio_path in list((data_dir / download_name / "gt_audio").iterdir()):
            data.append({"audio_path": str(audio_path)})

        super().__init__(data=data, path=data_dir/download_name, *args, **kwargs)