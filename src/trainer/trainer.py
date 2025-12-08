import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.mel_spectrogram import MelSpectrogramConfig
from pathlib import Path


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # Generator stage
        outputs = self.model(batch["audio"], batch["mel_spectrogram"], first_stage=None)
        batch.update(outputs)

        # Fix Generator, update Discriminator stage
        if self.is_train:
            self.optimizer_d.zero_grad()
        outputs = self.model(
            batch["audio"],
            batch["mel_spectrogram"],
            first_stage=True,
            audio_fake=batch["audio_fake"].detach(),
        )
        batch.update(outputs)

        losses_d = self.criterion_d(**batch)
        batch.update(losses_d)

        if self.is_train:
            batch["loss_d"].backward()
            self._clip_grad_norm()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

        # Update Generator stage
        if self.is_train:
            self.optimizer_g.zero_grad()
        outputs = self.model(
            batch["audio"],
            batch["mel_spectrogram"],
            first_stage=False,
            audio_fake=batch["audio_fake"],
        )
        batch.update(outputs)

        losses_g = self.criterion_g(**batch)
        batch.update(losses_g)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer_g.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(batch["audio_fake"]))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            self.log_spectrogram(
                batch["mel_spectrogram_fake"][0],
                spectrogram_name=f"{Path(batch['audio_path'][0]).stem}_mel_spectrogram",
            )
            self.log_audio(batch["audio_fake"][0], audio_name=f"{Path(batch['audio_path'][0]).stem}_audio")

    def log_spectrogram(self, spectrogram, spectrogram_name="mel_spectrogram"):
        spectrogram_for_plot = spectrogram.detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot, self.config)
        self.writer.add_image(spectrogram_name, image)

    def log_audio(self, audio, audio_name="audio"):
        audio = audio.detach().cpu()
        self.writer.add_audio(audio_name, audio, sample_rate=MelSpectrogramConfig.sr)
