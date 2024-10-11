from pathlib import Path

import numpy as np
import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


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
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
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
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, **batch):
        self.writer.add_audio("audio", audio[0], 16000)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        metrics_to_log = self.evaluation_metrics.keys()

        rows = {}
        rows["Target"] = [
            self.text_encoder.normalize_text(target)
            for target in text[:examples_to_log]
        ]

        probs_cut = [
            prob[:length] for prob, length in zip(log_probs, log_probs_length.numpy())
        ]
        if "CER_(LM-BeamSearch)" in metrics_to_log:
            bs_lm_preds = [
                self.text_encoder.ctc_beamsearch(prob, type="lm")[0]["hypothesis"]
                for prob in probs_cut[:examples_to_log]
            ]
            rows["BS_LM_predictions"] = bs_lm_preds
            metrics = np.array(
                [
                    np.array(
                        [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                    )
                    for target, pred in zip(rows["Target"], bs_lm_preds)
                ]
            )

            rows["BS_LM_CER"] = metrics[:, 0]
            rows["BS_LM_WER"] = metrics[:, 1]

        if "CER_(BeamSearch)" in metrics_to_log:
            bs_nolm_preds = [
                self.text_encoder.ctc_beamsearch(prob, type="nolm")[0]["hypothesis"]
                for prob in probs_cut[:examples_to_log]
            ]
            rows["BS_predictions"] = bs_nolm_preds

            metrics = np.array(
                [
                    np.array(
                        [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                    )
                    for target, pred in zip(rows["Target"], bs_nolm_preds)
                ]
            )
            rows["BS_CER"] = metrics[:, 0]
            rows["BS_WER"] = metrics[:, 1]

        if "CER_(BeamSearch-custom)" in metrics_to_log:
            bs_preds = [
                self.text_encoder.ctc_beamsearch(prob, type="custom")[0]["hypothesis"]
                for prob in probs_cut[:examples_to_log]
            ]
            rows["BS_custom_predictions"] = bs_preds

            metrics = np.array(
                [
                    np.array(
                        [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                    )
                    for target, pred in zip(rows["Target"], bs_preds)
                ]
            )
            rows["BS_CER"] = metrics[:, 0]
            rows["BS_WER"] = metrics[:, 1]

        if "CER_(Argmax)" in metrics_to_log:
            argmax_inds = log_probs.cpu().argmax(-1).numpy()
            argmax_inds = [
                inds[: int(ind_len)]
                for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
            ]
            argmax_texts_raw = [
                self.text_encoder.decode(inds) for inds in argmax_inds[:examples_to_log]
            ]
            argmax_texts = [
                self.text_encoder.ctc_decode(inds)
                for inds in argmax_inds[:examples_to_log]
            ]

            rows["Argmax_predicitons_raw"] = argmax_texts_raw
            rows["Argmax_predicitons"] = argmax_texts

            metrics = np.array(
                [
                    np.array(
                        [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                    )
                    for target, pred in zip(rows["Target"], argmax_texts)
                ]
            )
            rows["Argmax_CER"] = metrics[:, 0]
            rows["Argmax_WER"] = metrics[:, 1]

        df = pd.DataFrame.from_dict(rows)
        df.index = [Path(path).name for path in audio_path[:examples_to_log]]

        self.writer.add_table("predictions", df)
