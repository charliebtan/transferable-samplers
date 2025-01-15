from typing import Tuple

import torch

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule

from bgflow import NormalDistribution


class NormalizingFlowLitModule(BoltzmannGeneratorLitModule):

    def __init__(
        self,
        mean_free_prior: bool = False,
        force_gaussian_loss: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(*args, **kwargs)

        assert not (not self.hparams.mean_free_prior and self.hparams.force_gaussian_loss)

        if not self.hparams.mean_free_prior:
            # overwrites the MeanFreeNormalDistribution in BoltzmannGeneratorLitModule
            self.prior = NormalDistribution(self.datamodule.dim)

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = batch
        batch = batch.reshape(-1, 22, 3)
        x0, dlogp = self.net(x1)

        if self.hparams.force_gaussian_loss:
            loss = (0.5 * x0.pow(2)).mean() - dlogp.mean()
        else:
            loss = self.prior.energy(x0).mean() - dlogp.mean()
        return loss

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        x_pred, dlogp = self.net.forward(x)
        return -(-self.prior.energy(x).view(-1) - dlogp.view(-1))

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
        with torch.no_grad():
            x_pred = self.net.reverse(prior_samples)
            _, logdets = self.net(x_pred)
        log_p = prior_log_p.flatten() + logdets.flatten()
        return x_pred, log_p, torch.empty(0)


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
