import os
import shutil
import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    if config.inferencer.get("texts_query", ""):
        if not config.inferencer["use_text2mel"]:
            print(
                "inferencer.use_text2mel is manually set to True, because for specified texts_query it is obligatory"
            )
            config.inferencer["use_text2mel"] = True
            config.datasets.test["use_pretrained_text2mel"] = True

        config.inferencer["dataset_name"] = "query_data"

        print("Creating dataset with transcriptions from texts_query")
        transcriptions_dir = (
            ROOT_PATH / "data" / "datasets" / "query_data" / "transcriptions"
        )

        if transcriptions_dir.exists():
            shutil.rmtree(transcriptions_dir)
        transcriptions_dir.mkdir(exist_ok=True, parents=True)

        texts = config.inferencer["texts_query"].splitlines()
        for index, text in enumerate(texts):
            filepath = transcriptions_dir / f"query-{index + 1}.txt"
            filepath.write_text(text, encoding="utf-8")

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture
    model = instantiate(config.model).to(device)
    # print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = (
        ROOT_PATH
        / "data"
        / "saved"
        / "synthesized"
        / (
            config.inferencer.save_path
            + ("_text2mel" if config.inferencer["use_text2mel"] else "")
        )
    )
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
