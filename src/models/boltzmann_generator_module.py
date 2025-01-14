import logging
import math
from typing import Any, Dict, Tuple

import ot as pot
import torch
import torchmetrics
from bgflow import MeanFreeNormalDistribution
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from src.models.components.distribution_distances import \
    compute_distribution_distances
from src.models.components.jarzynski_sampler import JarzynskiSampler
from src.utils.tbg_utils import kish_effective_sample_size
from torchmetrics import MeanMetric

logger = logging.getLogger(__name__)


class BoltzmannGeneratorLitModule(LightningModule):
    """

    TODO - Add a description.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        jarzynski_sampler: JarzynskiSampler,
        sampling_config,
        compile: bool,
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
        self.prior = MeanFreeNormalDistribution(
            self.datamodule.dim, self.datamodule.n_particles, two_event_dims=False
        )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

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
        self.wandb_logger = None
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger

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
        samples, log_p, prior_samples = self.generate_samples(batch.shape[0])
        return samples, log_p, prior_samples

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

    def eval_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
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
        if prefix == "val":
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples
            true_data = self.datamodule.data_val
        elif prefix == "test":
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
            true_data = self.datamodule.data_test
        samples, log_p, prior_samples = self.generate_samples(num_proposal_samples)
        jarzynski_samples, jarzynski_weights = None, None
        if self.jarzynski_sampler is not None and self.jarzynski_sampler.enabled:
            num_jarzynski_samples = self.hparams.sampling_config.num_jarzynski_samples
            assert num_jarzynski_samples <= num_proposal_samples
            jarzynski_samples, jarzynski_weights = self.jarzynski_sampler.sample(
                samples[:num_jarzynski_samples]
            )

        sample_target_energy = self.datamodule.energy(samples)
        target_target_energy = self.datamodule.energy(true_data)
        assert log_p.shape == sample_target_energy.shape
        logits = -sample_target_energy - log_p
        ess = kish_effective_sample_size(logits) / len(logits)
        self.log(f"{prefix}/effective_sample_size", ess, sync_dist=True)
        num_eval_samples = min(
            self.hparams.sampling_config.num_eval_samples, len(samples), len(true_data)
        )
        names, dists = compute_distribution_distances(
            self.datamodule.unnormalize(samples[:num_eval_samples]).cpu(),
            self.datamodule.unnormalize(true_data[:num_eval_samples]).cpu(),
        )
        energy_w2 = math.sqrt(
            pot.emd2_1d(target_target_energy.cpu().numpy(), sample_target_energy.cpu().numpy())
        )
        energy_w1 = pot.emd2_1d(
            target_target_energy.cpu().numpy(),
            sample_target_energy.cpu().numpy(),
            metric="euclidean",
        )
        names = [f"{prefix}/{name}" for name in names]
        dist_metrics = dict(zip(names, dists))
        dist_metrics[f"{prefix}/energy_w2"] = energy_w2
        dist_metrics[f"{prefix}/energy_w1"] = energy_w1
        dist_metrics[f"{prefix}/num_eval_samples"] = num_eval_samples
        self.log_dict(metrics.compute() | dist_metrics)
        self.datamodule.log_on_epoch_end(
            samples,
            log_p,
            jarzynski_samples,
            jarzynski_weights,
            wandb_logger=self.wandb_logger,
            prefix=prefix,
        )
        metrics.reset()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(self.val_metrics, "val")

    def on_test_epoch_end(self) -> None:
        self.on_eval_epoch_end(self.test_metrics, "test")

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        raise NotImplementedError
        x0, dlogp = self.flow(x, reverse=True)
        return -(-self.prior.energy(x0).view(-1) - dlogp.view(-1))


if __name__ == "__main__":
    _ = BoltzmannGeneratorLitModule(None, None, None, None)
