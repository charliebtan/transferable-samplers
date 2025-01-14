from typing import Tuple

import torch

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule


class NormalizingFlowLitModule(BoltzmannGeneratorLitModule):
    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = batch
        x0, dlogp = self.net(x1)
        loss = (0.5 * x0.pow(2)).mean() - dlogp.mean()
        # loss = self.prior.energy(x0).mean() - dlogp.mean()
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
        x_pred = self.net.reverse(prior_samples)
        _, logdets = self.net(x_pred)
        log_p = prior_log_p.flatten() + logdets.flatten()
        return x_pred, log_p, torch.empty(0)


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
