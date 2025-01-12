from typing import Any, Dict, Tuple

import torch

from lightning import LightningDataModule, LightningModule
from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule
from src.models.shortcut_module import ShortcutLitModule


class NormalizingFlowLitModule(BoltzmannGeneratorLitModule):
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
        jarzynski_batch_size: int,  # TODO bit weird this is here but main generation done by data module
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, datamodule, compile)

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = batch
        x0, dlogp = self.net(x1)
        loss = (0.5 * x0.pow(2)).mean() - dlogp.mean()
        #loss = self.prior.energy(x0).mean() - dlogp.mean()
        return loss

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
        prior_samples = self.prior.sample(batch_size).to(self.device)
        prior_log_p = -self.prior.energy(prior_samples)
        # This is a bit slow... but probably fine 2x calls
        x_pred = self.net.reverse(prior_samples)
        _, logdets = self.net(x_pred)
        log_p = prior_log_p.flatten() + logdets.flatten()
        return x_pred, log_p, torch.empty(0)


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
