import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchmetrics
from bgflow import MeanFreeNormalDistribution, NormalDistribution
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.utilities import grad_norm
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.data.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
    energy_distances,
)
from src.models.components.ema import EMA
from src.models.components.jarzynski_sampler import JarzynskiSampler
from src.models.components.utils import RunningMedian, resample
from src.utils.tbg_utils import sampling_efficiency
from src.utils.data_types import SamplesData

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

        self.target_target_energy = None

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
        if "skip_eval_step" in self.hparams and not self.hparams.skip_eval_step:
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

    @torch.no_grad()
    def evaluate(self, prefix: str = "val", generator=None, output_dir=None) -> None:
        """Generates samples from the proposal and runs SMC if enabled.
        Also computes metrics, through the datamodule function "metrics_and_plots".
        """
        logging.info("Evaluating sampling")

        # Define proposal generator
        if generator is None:
            generator = self.batched_generate_samples
            if "dummy_ll" in self.hparams and self.hparams.dummy_ll:
                generator = lambda x: self.batched_generate_samples(x, dummy_ll=True)

        if prefix.startswith("test"):
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
            true_samples = self.datamodule.data_test
        else:
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples
            true_samples = self.datamodule.data_val

        true_data = SamplesData(
            true_samples,
            -self.energy(true_samples),
        )

        # Generate samples and record time
        torch.cuda.synchronize()
        start_time = time.time()
        proposal_samples, proposal_log_p, prior_samples = generator(num_proposal_samples)
        torch.cuda.synchronize()
        time_duration = time.time() - start_time
        self.log(f"{prefix}/samples_walltime", time_duration)
        self.log(f"{prefix}/samples_per_second", len(proposal_samples) / time_duration)

        # Compute proposal center of mass std - TODO should this just be 1 / sqrt(N) ?
        coms = proposal_samples.view(proposal_samples.shape[0], -1, 3).mean(dim=1)
        proposal_com_std = coms.std()
        self.log(f"{prefix}/proposal_com_std", proposal_com_std, sync_dist=True)
        self.datamodule.proposal_com_std = proposal_com_std

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            proposal_samples,
            proposal_samples_energy,
        )

        # Compute resampling index
        proposal_samples_energy = -self.energy(proposal_samples)
        resampling_logits = proposal_samples_energy - proposal_log_p # TODO CoM
        _, resampling_index = resample(resampling_logits, num_proposal_samples, return_index=True)

        # metrics.update(sampling_efficiency(resampling_logits), prefix))

        reweighted_data = SamplesData(
            proposal_samples[resampling_index],
            proposal_samples_energy[resampling_index],
        )

        # Filter proposal samples based on logit clipping
        if self.hparams.sampling_config.clip_logits:
            clipped_logits_mask = resampling_logits > torch.quantile(
                resampling_logits, 1 - float(self.hparams.sampling_config.clip_logits)
            )
            resampling_logits = resampling_logits[~clipped_logits_mask]
            proposal_samples = proposal_samples[~clipped_logits_mask]
            proposal_samples_energy = proposal_samples_energy[~clipped_logits_mask]
            logging.info("Clipped logits")

        # Filter samples based on target energy cutoff
        if self.hparams.sampling_config.energy_cutoff is not None:
            filter_array = proposal_samples_energy < self.hparams.sampling_config.energy_cutoff
            filtered_samples = proposal_samples[filter_array]
            self.log(f"{prefix}/filtered_samples", torch.sum(~filter_array).float())
            logging.info("Clipping energies")
        else:
            filtered_samples = proposal_samples
            self.log(f"{prefix}/filtered_samples", 0.0)

        if self.jarzynski_sampler is not None and self.jarzynski_sampler.enabled:

            # TODO remove this once refactored
            self.jarzynski_sampler.use_com_energy = self.hparams.use_com_energy

            logging.info("Jarzynski sampling enabled")

            # Hack to get wandb logger into jarzynski sampler for plotting
            self.jarzynski_sampler.wandb_logger = self.datamodule.get_wandb_logger(self.loggers)

            num_jarzynski_samples = min(
                self.hparams.sampling_config.num_jarzynski_samples, len(filtered_samples)
            )

            # Generate jarzynski samples and record time
            torch.cuda.synchronize()
            start_time = time.time()
            jarzynski_samples, jarzynski_logits = self.jarzynski_sampler.sample(
                filtered_samples[:num_jarzynski_samples]
            )
            torch.cuda.synchronize()
            time_duration = time.time() - start_time
            self.log(f"{prefix}/jarzynski/samples_walltime", time_duration)
            self.log(
                f"{prefix}/jarzynski/samples_per_second", len(jarzynski_samples) / time_duration
            )

            # Datatype for easier metrics and plotting
            jarzynski_data = SamplesData(
                jarzynski_samples,
                -self.energy(jarzynski_samples),
            )

        else:

            jarzynski_data = None

        # log dataset metrics
        metrics = self.datamodule.metrics_and_plots(
            true_data,
            proposal_data,
            reweighted_data,
            jarzynski_data,
            num_eval_samples=num_eval_samples,
            prefix=prefix,
        )
        self.log_dict(metrics)

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
