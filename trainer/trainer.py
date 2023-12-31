import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchaudio

from base import BaseTrainer
from base.base_text_encoder import BaseTextEncoder
from logger.utils import plot_spectrogram_to_buf
from utils import inf_loop, MetricTracker
from melspec import *
from loss import mel_loss
import itertools

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            msd,
            mpd,
            criterion,
            metrics,
            optimizer_g,
            optimizer_d,
            config,
            device,
            dataloaders,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, msd, mpd, criterion, metrics, optimizer_g, optimizer_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.mel_specer = MelSpectrogram(MelSpectrogramConfig())
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm gen", "mel_loss", "fmap_loss", "gan_loss", *[m.name for m in self.metrics["train"]], writer=self.writer
        )

        self.evaluation_metrics = MetricTracker(
            "loss", "mel_loss", "fmap_loss", "gan_loss", *[m.name for m in self.metrics["val"]], writer=self.writer
        )


    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.generator.parameters()), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                filter(lambda p: p.requires_grad, itertools.chain(self.msd.parameters(), self.mpd.parameters())), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.msd.train()
        self.mpd.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm gen", self.get_grad_norm(self.generator))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )
                # self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                
                for i in range(1, 4):
                    audio_wave, sr = torchaudio.load(f"test/audio_{i}.wav")
                    audio_wave = audio_wave[0:1, :]  # remove all channels but the first
                    target_sr = 22050
                    if sr != target_sr:
                        audio_wave = torchaudio.functional.resample(audio_wave, sr, target_sr)
                    new_length = (audio_wave.shape[1] // 256) * 256
                    audio_wave = audio_wave[:, :new_length].unsqueeze(1)
                    mel = self.mel_specer(audio_wave)[..., :-1].squeeze(1).to(self.device)
                    generated_audio = self.generator(mel)
                    self.writer.add_audio(f"real_{i}", audio_wave.squeeze(1), 22050)
                    self.writer.add_audio(f"generated_{i}", generated_audio.cpu().squeeze(1), 22050)



                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()

        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()
        
        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        # batch = self.move_batch_to_device(batch, self.device)

        spec = batch['spectrogram'][..., :-1]
        wav = batch['audio'].unsqueeze(1)
        wav_mel = self.mel_specer(wav)[..., :-1]

        spec = torch.autograd.Variable(spec.to(self.device, non_blocking=True))
        wav = torch.autograd.Variable(wav.to(self.device, non_blocking=True))
        wav_mel = torch.autograd.Variable(wav_mel.to(self.device, non_blocking=True))

        generated_wav = self.generator(spec)
        mel_from_gen = self.mel_specer(generated_wav.cpu())[..., :-1].to(self.device)

        if is_train:
            self.optimizer_d.zero_grad()

        # MPD
        _, disc_loss_mpd, _ = self.mpd(wav, generated_wav.detach())

        # MSD
        _, disc_loss_msd, _ = self.msd(wav, generated_wav.detach())

        loss_disc_all = disc_loss_msd + disc_loss_mpd

        if is_train:
            loss_disc_all.backward()
            # self._clip_grad_norm()
            self.optimizer_d.step()
            
        # # Generator
        self.optimizer_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = mel_loss(mel_from_gen, wav_mel)

        fmap_loss_mpd, _, gen_loss_mpd_g = self.mpd(wav, generated_wav)
        fmap_loss_msd, _, gen_loss_msd_g = self.msd(wav, generated_wav)

        fmap_loss = fmap_loss_msd + fmap_loss_mpd
        gan_loss = gen_loss_msd_g + gen_loss_mpd_g
        
        batch["fmap_loss"] = fmap_loss
        batch["gan_loss"] = gan_loss
        batch["mel_loss"] = loss_mel
        batch["loss"] = self.criterion(gan_loss, fmap_loss, loss_mel)

        if is_train:
            batch["loss"].backward()
            # self._clip_grad_norm()
            self.optimizer_g.step()

        metrics.update("loss", batch["loss"].item())
        metrics.update("fmap_loss", batch["fmap_loss"].item())
        metrics.update("mel_loss", batch["mel_loss"].item())
        metrics.update("gan_loss", batch["gan_loss"].item())
        for met in self.metrics["train" if is_train else "val"]:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.msd.eval()
        self.mpd.eval()
        self.generator.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            # self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.generator.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        for name, p in self.msd.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        for name, p in self.mpd.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    # def _log_predictions(
    #         self,
    #         text,
    #         log_probs,
    #         log_probs_length,
    #         audio_path,
    #         examples_to_log=10,
    #         *args,
    #         **kwargs,
    # ):
    #     if self.writer is None:
    #         return
    #     argmax_inds = log_probs.cpu().argmax(-1).numpy()
    #     argmax_inds = [
    #         inds[: int(ind_len)]
    #         for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
    #     ]
    #     argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
    #     argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
    #     tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
    #     shuffle(tuples)
    #     rows = {}
    #     for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
    #         target = BaseTextEncoder.normalize_text(target)
    #         wer = calc_wer(target, pred) * 100
    #         cer = calc_cer(target, pred) * 100

    #         rows[Path(audio_path).name] = {
    #             "target": target,
    #             "raw prediction": raw_pred,
    #             "predictions": pred,
    #             "wer": wer,
    #             "cer": cer
    #         }
    #     self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
