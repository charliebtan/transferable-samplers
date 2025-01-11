from typing import Any, Dict, Tuple

import torch

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
        compile: bool,
        jarzynski_batch_size: int,  # TODO bit weird this is here but main generation done by data module
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, compile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net(x)

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        v_pred, *_ = self.forward(xt)

        loss = self.criterion(v_pred, v_shortcuts)
        loss = self.criterion(v_pred, v_shortcuts)

        return loss

    def flow(self, x: torch.Tensor, reverse=False) -> torch.Tensor:
        if not reverse:
            v_pred, logdets = self.forward(x)
        else:
            v_pred, *_ = self.net.reverse(x)
            _, logdets = self.forward(x)

        # TODO I think you may need to handle this case different because this is the forward logdet

        samples = x + v_pred

        return samples, logdets[..., None]

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

        samples, dlogp = self.flow(prior_samples)

        log_p = prior_log_p + dlogp

        return samples, log_p.squeeze(), torch.empty(0)


if __name__ == "__main__":
    _ = InvertibleShortcutLitModule(None, None, None, None)
