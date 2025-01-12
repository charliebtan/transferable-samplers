from typing import Any, Dict, Tuple

import torch

from src.models.boltzmann_flow_module import BoltzmanFlowLitModule

# TODO this currently assumes you are using a zuko normalzing flow
# If you start using other normalizing flows you will need to make a zuko
# wrapper I think?


class NormalizingFlowLitModule(BoltzmanFlowLitModule):
    """

    TODO - Add a description.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net().log_prob(x)

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """
        loss = -self.forward(batch).mean()

        return loss

    def generate_samples(
        self, batch_size: int, n_timesteps: int = None, device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        samples = self.net().sample(batch_size)
        log_p = self.net().log_prob(samples)
        return samples, log_p.squeeze(), torch.empty(0)


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
