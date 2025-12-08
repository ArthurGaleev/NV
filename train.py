import warnings
from itertools import chain

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifi_gan_coslr")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed, config.trainer.get("save_reproducibility", True))

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    if config.trainer.get("parallel", False):
        model = torch.nn.DataParallel(model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function_d = instantiate(config.loss_function_d).to(device)
    loss_function_g = instantiate(config.loss_function_g).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params_mpd = filter(lambda p: p.requires_grad, model.mpd.parameters())
    trainable_params_msd = filter(lambda p: p.requires_grad, model.msd.parameters())
    trainable_params_generator = filter(
        lambda p: p.requires_grad, model.generator.parameters()
    )
    optimizer_d = instantiate(
        config.optimizer_d, params=chain(trainable_params_mpd, trainable_params_msd)
    )
    optimizer_g = instantiate(config.optimizer_g, params=trainable_params_generator)
    lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)
    lr_scheduler_g = instantiate(config.lr_scheduler_g, optimizer=optimizer_g)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion_d=loss_function_d,
        criterion_g=loss_function_g,
        metrics=metrics,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        lr_scheduler_d=lr_scheduler_d,
        lr_scheduler_g=lr_scheduler_g,
        config=config,
        device=device,
        dtype=config.trainer.get("dtype", "float32"),
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
