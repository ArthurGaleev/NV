import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    if "audio" in dataset_items[0].keys():
        result_batch["audio"] = torch.vstack([elem["audio"] for elem in dataset_items])
    if "audio_path" in dataset_items[0].keys():
        result_batch["audio_path"] = [elem["audio_path"] for elem in dataset_items]
    if "text_path" in dataset_items[0].keys():
        result_batch["text_path"] = [elem["text_path"] for elem in dataset_items]

    result_batch["mel_spectrogram"] = torch.vstack(
        [elem["mel_spectrogram"] for elem in dataset_items]
    )

    return result_batch
