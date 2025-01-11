import math
from typing import Any, Dict, Tuple

import torch
import torchmetrics
from bgflow import MeanFreeNormalDistribution
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from src.utils.dw4_plots import TARGET
from src.utils.tbg_utils import kish_effective_sample_size
from torchmetrics import MeanMetric
from tqdm import tqdm


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
        compile: bool,
        jarzynski_batch_size: int = None,  # TODO bit weird this is here but main generation done by data module
    ) -> None:
        """Initialize a `FlowMatchLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.datamodule = datamodule

        # loss function
        self.criterion = torch.nn.MSELoss(reduce="mean")

        # metric objects for calculating and averaging accuracy across batches

        self.train_metrics = torchmetrics.MetricCollection(
            {"loss": MeanMetric()}, prefix="train/"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")
        # self.train_loss = MeanMetric()

        # TODO TODO I'm not sure this is the right place to have this

        self.prior = MeanFreeNormalDistribution(8, 4, two_event_dims=False)

        # torch.distributions.MultivariateNormal(
        #     torch.zeros(8), torch.eye(8)
        # )  # TODO is this the right place for this? # TODO MeanFreeNormal

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
        # self.log("train/loss", loss)
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
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return {"optimizer": optimizer}

    def predict_step(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="test")

    def on_eval_epoch_end(self, metrics, prefix: str = "val") -> None:
        batch_size = 100
        samples, log_p, prior_samples = self.generate_samples(batch_size)
        self.log_dict(metrics.compute())
        self.datamodule.log_on_epoch_end(
            samples, log_p, wandb_logger=self.wandb_logger, prefix=prefix
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
        x0, dlogp = self.flow(x, reverse=True)
        return -(-self.prior.energy(x0).view(-1) - dlogp.view(-1))

    def linear_energy_interpolation(self, x, t):
        energy = (1 - t) * self.proposal_energy(x) + t * TARGET.energy(x).view(-1)
        assert energy.shape == (
            x.shape[0],
        ), "Energy should be a flat vector, one value per sample"
        return energy

    def linear_energy_interpolation_gradients(self, x, t):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True

            et = self.linear_energy_interpolation(x, t)

            assert (
                et.requires_grad
            ), "et should require grad - check the energy function for no_grad"

            # this is a bit hacky but is fine as long as
            # the energy function is defined properly and
            # doesn't mix batch items

            x_grad, t_grad = torch.autograd.grad(et.sum(), (x, t))

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"
            assert t_grad.shape == t.shape, "t_grad should have the same shape as t"

        assert x_grad is not None, "x_grad should not be None"
        assert t_grad is not None, "t_grad should not be None"

        return x_grad, t_grad

    @torch.no_grad()
    def jarzyinski_process(self, samples_proposal):
        # TODO I think I should test with a simple energy function and make sure I am getting the correct energies etc

        X = samples_proposal.to(self.device)

        eps = 0.4
        num_timesteps = 1000  # TODO should default to 1000

        A = torch.zeros(X.shape[0], device=X.device)  # the jarzynski weights

        timesteps = torch.linspace(0, 1, num_timesteps + 1)
        dt = 1 / num_timesteps

        A_list = [A]
        ESS_list = []

        # slice into list of batches (tensors)
        X_batches = [
            X[i : i + self.hparams.jarzynski_batch_size]
            for i in range(0, X.shape[0], self.hparams.jarzynski_batch_size)
        ]
        A_batches = [
            A[i : i + self.hparams.jarzynski_batch_size]
            for i in range(0, A.shape[0], self.hparams.jarzynski_batch_size)
        ]

        for j, t in tqdm(enumerate(timesteps[:-1])):
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):
                # get the energy gradients
                energy_grad_x, energy_grad_t = (
                    self.linear_energy_interpolation_gradients(X_batch, t)
                )

                # compute the updates
                dX_t = -eps * energy_grad_x * dt + math.sqrt(
                    2 * eps * dt
                ) * torch.randn_like(X_batch)
                dA_t = -energy_grad_t * dt

                assert (
                    dX_t.shape == X_batch.shape
                ), "dX_t should have the same shape as X_batch"
                assert (
                    dA_t.shape == A_batch.shape
                ), "dA_t should have the same shape as A_batch"

                # apply the updates to the batch in the list
                X_batches[batch_idx] = X_batch + dX_t
                A_batches[batch_idx] = A_batch + dA_t

                if X_batches[batch_idx].isnan().any():
                    raise ValueError("X_batch has NaNs")
                if A_batches[batch_idx].isnan().any():
                    raise ValueError("A_batch has NaNs")

            # cat the batches to compute global statistics
            X = torch.cat(X_batches, dim=0)
            A = torch.cat(A_batches, dim=0)

            jarzynski_weights = torch.softmax(A, dim=-1)

            A_list.append(A)
            ESS = kish_effective_sample_size(A).item() / len(A)
            ESS_list.append(ESS)

            if ESS < 0.5:
                # qmc_rand = sampler.random(n=len(A))
                # cum_prob = torch.cumsum(torch.softmax(A, dim=-1), dim=0)
                # indexes = np.searchsorted(cum_prob, qmc_rand, side="left").flatten()
                indexes = torch.multinomial(jarzynski_weights, len(A), replacement=True)
                X = X[indexes]
                A = torch.zeros_like(A)
            if j % 1000 == 0:
                pass
                # print("energy", j, target_energy(X))

        jarzynski_samples = X
        jarzynski_weights = torch.softmax(A, dim=-1)

        return jarzynski_samples.cpu(), jarzynski_weights.cpu()


if __name__ == "__main__":
    _ = BoltzmannGeneratorLitModule(None, None, None, None)
