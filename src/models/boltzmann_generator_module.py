import logging
import math
from typing import Any, Dict, Optional, Tuple
import time

import hydra
import matplotlib.pyplot as plt
import ot as pot
import torch
import torchmetrics
from bgflow import MeanFreeNormalDistribution, NormalDistribution
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
    energy_distances,
)
from src.models.components.ema import EMA
from src.models.components.jarzynski_sampler import JarzynskiSampler
from src.models.components.utils import RunningMedian, resample
from src.utils.tbg_utils import sampling_efficiency

logger = logging.getLogger(__name__)


class BoltzmannGeneratorLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        jarzynski_sampler: JarzynskiSampler,
        sampling_config,
        ema_decay: float,
        compile: bool,
        mean_free_prior: bool = True,
        stabilize_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `FlowMatchLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("datamodule"))
        if args or kwargs:
            logger.warning(f"Unexpected arguments: {args}, {kwargs}")

        self.net = net

        if self.hparams.ema_decay > 0:
            self.net = EMA(net, decay=self.hparams.ema_decay)

        self.datamodule = datamodule

        self.jarzynski_sampler = jarzynski_sampler(
            source_energy=self.proposal_energy,
            target_energy=self.datamodule.energy,
        )

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches

        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        if self.hparams.mean_free_prior:
            self.prior = MeanFreeNormalDistribution(
                self.datamodule.dim, self.datamodule.n_particles, two_event_dims=False
            )
        else:
            self.prior = NormalDistribution(self.datamodule.dim)
        if self.hparams.stabilize_training:
            self.gradient_history = RunningMedian(100)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)
        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}

    def predict_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of samples.

        :param batch: A batch of (dummy) data.
        :return: A tuple containing the generated samples, the log probability, and the prior
            samples.
        """
        samples, log_p, prior_samples = self.batched_generate_samples(batch.shape[0])
        return samples, log_p, prior_samples

    def batched_generate_samples(
        self, total_size: int, batch_size: Optional[int] = None, dummy_ll: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.hparams.sampling_config.batch_size
        samples = []
        log_ps = []
        prior_samples = []
        for _ in tqdm(range(total_size // batch_size)):
            s, lp, ps = self.generate_samples(batch_size, dummy_ll=dummy_ll)
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        if total_size % batch_size > 0:
            s, lp, ps = self.generate_samples(total_size % batch_size, dummy_ll=dummy_ll)
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        samples = torch.cat(samples, dim=0)
        log_ps = torch.cat(log_ps, dim=0)
        prior_samples = torch.cat(prior_samples, dim=0)
        return samples, log_ps, prior_samples

    def generate_samples(
        self, batch_size: int, n_timesteps: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """
        raise NotImplementedError

    @torch.no_grad()
    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
        return
        loss = self.model_step(batch)
        if prefix == "val":
            self.val_metrics.update(loss)
        elif prefix == "test":
            self.test_metrics.update(loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="test")

    def on_eval_epoch_end(self, metrics, prefix: str = "val") -> None:
        self.log_dict(metrics.compute())
        metrics.reset()
        #try:
        if self.hparams.ema_decay > 0:
            if self.hparams.eval_ema:
                self.net.backup()
                self.net.copy_to_model()
                self.evaluate(prefix)
                self.net.restore_to_model()
            if self.hparams.eval_non_ema:
                self.evaluate(prefix + "/non_ema")
        else:
            self.evaluate(prefix)
        plt.close("all")
        #except Exception as e:
        #    logger.warning("Skipping evaluation due to exception")
        #    logger.warning(e)

    def evaluate(self, prefix: str = "val", generator=None, output_dir=None) -> None:
        logging.info("Eval epoch end")
        if generator is None:
            generator = self.batched_generate_samples
        if prefix.startswith("val") or prefix.startswith("base"):
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples
            true_data = self.datamodule.data_val
        elif prefix.startswith("test"):
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
            true_data = self.datamodule.data_test

        
        torch.cuda.synchronize()
        start_time = time.time()

        samples, log_p, prior_samples = generator(num_proposal_samples)

        torch.cuda.synchronize()
        time_duration = time.time() - start_time
        self.log(f"{prefix}/samples_walltime", time_duration)
        self.log(f"{prefix}/samples_per_second", len(samples) / time_duration)

        samples_dict = {
            "samples": samples,
            "log_p": log_p,
            "prior_samples": prior_samples,
        }
        if output_dir is None:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logging.info(f"Saving {len(samples)} samples to {output_dir}/{prefix}_samples.pt")
        torch.save(samples_dict, f"{output_dir}/{prefix}_samples.pt")

        # compute energy
        sample_target_energy = self.datamodule.energy(samples)

        if self.hparams.sampling_config.energy_cutoff is not None:
            filter_array = sample_target_energy < self.hparams.sampling_config.energy_cutoff
            samples = samples[filter_array]
            log_p = log_p[filter_array]
            sample_target_energy = sample_target_energy[filter_array]
            self.log(f"{prefix}/filtered_samples", torch.sum(~filter_array).float())
        else:
            samples = samples
            self.log(f"{prefix}/filtered_samples", 0.0)

        self.log(f"{prefix}/mean_energy", sample_target_energy.mean(), sync_dist=True)
        target_target_energy = self.datamodule.energy(true_data)

        # compute weights + resample
        assert log_p.shape == sample_target_energy.shape
        logits = -sample_target_energy - log_p
        ess = sampling_efficiency(logits)
        self.log(f"{prefix}/effective_sample_size", ess, sync_dist=True)
        resampled_samples = resample(samples, logits)
        num_eval_samples = min(
            self.hparams.sampling_config.num_eval_samples, len(samples), len(true_data)
        )

        # compute dist metrics
        dist_metrics = compute_distribution_distances_with_prefix(
            self.datamodule.unnormalize(samples[:num_eval_samples]).cpu(),
            self.datamodule.unnormalize(true_data[:num_eval_samples]).cpu(),
            prefix=prefix,
        )
        dist_metrics[f"{prefix}/num_eval_samples"] = num_eval_samples
        self.log_dict(dist_metrics)
            
        # compute resampled dist metrics
        resampled_dist_metrics = compute_distribution_distances_with_prefix(
            self.datamodule.unnormalize(resampled_samples[:num_eval_samples]).cpu(),
            self.datamodule.unnormalize(true_data[:num_eval_samples]).cpu(),
            prefix=prefix + "/resampled",
        )
        self.log_dict(resampled_dist_metrics)

        # compute energy metrics
        energy_metrics = energy_distances(sample_target_energy, target_target_energy, prefix)
        self.log_dict(energy_metrics)

        # compute resampled energy metrics
        resampled_sample_target_energy = self.datamodule.energy(resampled_samples)
        self.log(f"{prefix}/resampled/mean_energy", resampled_sample_target_energy.mean(), sync_dist=True)
        resampled_energy_metrics = energy_distances(resampled_sample_target_energy, target_target_energy, prefix + "/resampled")
        self.log_dict(resampled_energy_metrics)

        jarzynski_samples, jarzynski_logits, jarzynski_energy_metrics = None, None, None
        if self.jarzynski_sampler is not None and self.jarzynski_sampler.enabled:

            self.jarzynski_sampler.wandb_logger = self.datamodule.get_wandb_logger(self.loggers)

            num_jarzynski_samples = min(
                self.hparams.sampling_config.num_jarzynski_samples, len(samples)
            )

            torch.cuda.synchronize()
            start_time = time.time()

            jarzynski_samples, jarzynski_logits = self.jarzynski_sampler.sample(
                samples[:num_jarzynski_samples]
            )

            torch.cuda.synchronize()
            time_duration = time.time() - start_time
            self.log(f"{prefix}/jarzynski/samples_walltime", time_duration)
            self.log(f"{prefix}/jarzynski/samples_per_second", len(jarzynski_samples) / time_duration)

            num_eval_samples = min(
                num_jarzynski_samples,
                self.hparams.sampling_config.num_eval_samples,
                len(true_data),
            )

            jarzynski_ess = sampling_efficiency(jarzynski_logits)
            self.log(f"{prefix}/jarzynski/effective_sample_size", jarzynski_ess, sync_dist=True)

            # compute jarzynski dist metrics
            jarzynski_dist_metrics = compute_distribution_distances_with_prefix(
                self.datamodule.unnormalize(jarzynski_samples[:num_eval_samples]),
                self.datamodule.unnormalize(true_data[:num_eval_samples]),
                prefix=prefix + "/jarzynski",
            )
            jarzynski_dist_metrics[f"{prefix}/jarzynski/num_eval_samples"] = num_eval_samples
            self.log_dict(jarzynski_dist_metrics)

            # compute jarzynski energy metrics
            sample_target_jarzynski_energy = self.datamodule.energy(jarzynski_samples)
            self.log(f"{prefix}/jarzynski/mean_energy", sample_target_jarzynski_energy.mean())
            jarzynski_energy_metrics = energy_distances(
                sample_target_jarzynski_energy, target_target_energy, prefix + "/jarzynski"
            )
            self.log_dict(jarzynski_energy_metrics)

        # log dataset metrics
        dataset_metrics = self.datamodule.log_on_epoch_end(
            samples,
            log_p,
            jarzynski_samples,
            jarzynski_logits,
            samples_test=true_data,
            loggers=self.loggers,
            prefix=prefix,
        )
        if dataset_metrics is not None:
            self.log_dict(dataset_metrics)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_train_epoch_start(self) -> None:
        logging.info("Train epoch start")
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        logging.info("Validation epoch start")
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        logging.info("Test epoch start")
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logging.info("Train epoch end")

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(self.val_metrics, "val")
        logging.info("Validation epoch end")

    def on_test_epoch_end(self) -> None:
        self.on_eval_epoch_end(self.test_metrics, "test")
        logging.info("Test epoch end")

    def on_after_backward(self) -> None:

        valid_gradients = True
        flat_grads = torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])
        global_norm = torch.norm(flat_grads, p=2)
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )

                if not valid_gradients:
                    break

        self.log("global_gradient_norm", global_norm, on_step=True, prog_bar=True)
        if not valid_gradients:
            logger.warning(
                "detected inf or nan values in gradients. not updating model parameters"
            )
            self.zero_grad()
            return

        if not self.hparams.stabilize_training:
            return
        self.gradient_history.update(global_norm.item())
        running_global_norm = self.gradient_history.compute()
        if global_norm > 20 * running_global_norm:
            logger.warning(
                f"detected large_gradient {global_norm} which is more than 20 times the running median {running_global_norm}. not updating model parameters"
            )
            self.zero_grad()
        elif global_norm > 5 * running_global_norm:
            logger.warning(
                f"detected large_gradient {global_norm} which is more than 5 "
                f"times the running median {running_global_norm}. clipping"
                "gradients"
            )
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 5 * running_global_norm)
            self.log("clipped_gradient_norm", norm, on_step=True, prog_bar=True)

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        raise NotImplementedError

    # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in self.trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.log_dict({"train/grad_norm": total_norm}, prog_bar=True)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.net, EMA):
            self.net.update_ema()


if __name__ == "__main__":
    _ = BoltzmannGeneratorLitModule(None, None, None, None)
