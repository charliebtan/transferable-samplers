import logging
import time
from typing import Any, Optional

import hydra
import matplotlib.pyplot as plt
import torch
import torchmetrics
from bgflow import NormalDistribution
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.data.components.data_types import SamplesData
from src.models.components.ema import EMA
from src.models.components.smc_sampler import SMCSampler
from src.models.components.utils import resample

logger = logging.getLogger(__name__)


class BoltzmannGeneratorLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        smc_sampler: SMCSampler,
        sampling_config: dict,
        ema_decay: float,
        compile: bool,
        use_com_adjustment: bool = False,
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

        self.smc_sampler = smc_sampler(
            source_energy=self.proposal_energy,
            target_energy=self.datamodule.energy,
            log_image_fn=self.log_image,
        )

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.prior = NormalDistribution(
            self.datamodule.hparams.dim  # for transferable this will be the dim of the largest peptide
        )

    def log_image(self, img: torch.Tensor, title: str = None) -> None:
        """Log an image to the logger.

        :param img: The image to log.
        :param title: The title of the image.
        """
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(title, [img])

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

    def configure_optimizers(self) -> dict[str, Any]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
        if "skip_eval_step" in self.hparams and not self.hparams.skip_eval_step:
            loss = self.model_step(batch)
            if prefix == "val":
                self.val_metrics.update(loss)
            elif prefix == "test":
                self.test_metrics.update(loss)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
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
    def evaluate(self, prefix: str = "val", proposal_generator=None, output_dir=None) -> None:
        """Generates samples from the proposal and runs SMC if enabled.
        Also computes metrics, through the datamodule function "metrics_and_plots".
        """
        logging.info("Evaluating sampling")

        # Define proposal generator
        if proposal_generator is None:
            proposal_generator = self.batched_generate_samples
            if "dummy_ll" in self.hparams and self.hparams.dummy_ll:
                proposal_generator = lambda x: self.batched_generate_samples(x, dummy_ll=True)

        if prefix.startswith("test"):
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
            true_samples = self.datamodule.data_test.data
            encodings = self.datamodule.data_test.encodings  # noqa: F841
        else:
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples
            true_samples = self.datamodule.data_val.data
            encodings = self.datamodule.data_val.encodings  # noqa: F841

        true_data = SamplesData(
            self.datamodule.as_pointcloud(self.datamodule.unnormalize(true_samples)),
            self.datamodule.energy(true_samples),
        )

        # Generate samples and record time
        torch.cuda.synchronize()
        start_time = time.time()
        proposal_samples, proposal_log_p, prior_samples = proposal_generator(num_proposal_samples)
        torch.cuda.synchronize()
        time_duration = time.time() - start_time
        self.log(f"{prefix}/samples_walltime", time_duration)
        self.log(f"{prefix}/samples_per_second", len(proposal_samples) / time_duration)

        # Save samples to disk
        samples_dict = {
            "prior_samples": prior_samples,
            "proposal_samples": proposal_samples,
            "proposal_log_p": proposal_log_p,
        }
        if output_dir is None:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logging.info(f"Saving {len(proposal_samples)} samples to {output_dir}/{prefix}_samples.pt")
        torch.save(samples_dict, f"{output_dir}/{prefix}_samples.pt")

        # Compute energy
        proposal_samples_energy = self.datamodule.energy(proposal_samples)

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples)),
            proposal_samples_energy,
        )

        # Compute proposal center of mass std - TODO should this just be 1 / sqrt(N) ?
        coms = self.datamodule.center_of_mass(proposal_samples).mean(dim=1)
        proposal_com_std = coms.std()
        self.proposal_com_std = proposal_com_std
        self.log(f"{prefix}/proposal_com_std", proposal_com_std, sync_dist=True)

        # Apply CoM adjustment to energy, this must be done here for compatibility with CNFs
        if self.hparams.sampling_config.use_com_adjustment:
            proposal_log_p = proposal_log_p - self.com_energy_adjustment(proposal_samples)

        # Compute resampling index
        resampling_logits = -proposal_samples_energy - proposal_log_p

        # Filter samples based on logit clipping - this affects both IS and SMC
        if self.hparams.sampling_config.clip_reweighting_logits:
            clipped_logits_mask = resampling_logits > torch.quantile(
                resampling_logits,
                1 - float(self.hparams.sampling_config.clip_reweighting_logits),
            )
            proposal_samples = proposal_samples[~clipped_logits_mask]
            proposal_samples_energy = proposal_samples_energy[~clipped_logits_mask]
            resampling_logits = resampling_logits[~clipped_logits_mask]
            logging.info("Clipped logits for resampling")

        _, resampling_index = resample(proposal_samples, resampling_logits, return_index=True)

        reweighted_data = SamplesData(
            self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples[resampling_index])),
            proposal_samples_energy[resampling_index],
            logits=resampling_logits,
        )

        if self.smc_sampler is not None and self.smc_sampler.enabled:
            logging.info("SMC sampling enabled")

            num_smc_samples = min(self.hparams.sampling_config.num_smc_samples, len(proposal_samples))

            # Generate smc samples and record time
            torch.cuda.synchronize()
            start_time = time.time()
            smc_samples, smc_logits = self.smc_sampler.sample(
                proposal_samples[:num_smc_samples]
            )  # already returned resampled
            torch.cuda.synchronize()
            time_duration = time.time() - start_time
            self.log(f"{prefix}/smc/samples_walltime", time_duration)
            self.log(f"{prefix}/smc/samples_per_second", len(smc_samples) / time_duration)

            # Save samples to disk
            smc_samples_dict = {
                "smc_samples": smc_samples,
                "smc_logits": smc_logits,
            }
            logging.info(f"Saving {len(smc_samples)} samples to {output_dir}/{prefix}_smc_samples.pt")
            torch.save(smc_samples_dict, f"{output_dir}/{prefix}_smc_samples.pt")

            # Datatype for easier metrics and plotting
            smc_data = SamplesData(
                self.datamodule.as_pointcloud(self.datamodule.unnormalize(smc_samples)),
                self.datamodule.energy(smc_samples),
                logits=smc_logits,
            )

        else:
            smc_data = None

        # log dataset metrics
        metrics = self.datamodule.metrics_and_plots(
            self.log_dict,
            self.log_image,
            true_data,
            proposal_data,
            reweighted_data,
            smc_data,
            prefix=prefix,
        )
        self.log_dict(metrics)

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
        for _name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())

                if not valid_gradients:
                    break

        self.log("global_gradient_norm", global_norm, on_step=True, prog_bar=True)
        if not valid_gradients:
            logger.warning("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()
            return

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

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        raise NotImplementedError


if __name__ == "__main__":
    _ = BoltzmannGeneratorLitModule(None, None, None, None)
