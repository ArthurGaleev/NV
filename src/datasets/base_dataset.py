import logging
import random
from typing import List

import torch
import torchaudio
from torch.utils.data import Dataset

from src.transforms.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        use_pretrained_text2mel=False,
        audio_len=None
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.get_mel_spectrogram = MelSpectrogram(MelSpectrogramConfig)

        self.instance_transforms = instance_transforms
        self.use_pretrained_text2mel = use_pretrained_text2mel
        self.audio_len = audio_len

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """

        data_dict = self._index[ind]

        if self.use_pretrained_text2mel:
            if "audio_path" in data_dict.keys():
                return {
                    "mel_spectrogram": data_dict["mel_spectrogram"],
                    "audio_path": data_dict["audio_path"],
                }
            return {
                "mel_spectrogram": data_dict["mel_spectrogram"],
                "text_path": data_dict["text_path"],
            }

        audio_path = data_dict["audio_path"]

        audio = self.load_audio(audio_path)

        mel_spectrogram = self.get_mel_spectrogram(audio)

        if self.audio_len:
            correct_mel_len = self.audio_len // 256
            mel_rnd_start = random.randint(0, mel_spectrogram.shape(1) - correct_mel_len - 1)
            mel_spectrogram = mel_spectrogram[:, mel_rnd_start:mel_rnd_start + correct_mel_len]

        instance_data = {
            "audio": audio,
            "mel_spectrogram": mel_spectrogram,
            "audio_path": audio_path,
        }

        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        target_sr = MelSpectrogramConfig.sr
        if sr != MelSpectrogramConfig.sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
        # Filter logic
        pass

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "audio" or "mel_spectrogram" in entry, (
                "Each dataset item should include field 'audio' 'mel_spectrogram'"
                " - pnly mel spectrogram if there were no gt_audio, audio transcriptions instead."
            )

    @staticmethod
    def _sort_index(index):
        """
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
