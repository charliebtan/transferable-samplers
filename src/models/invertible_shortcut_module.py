from typing import Any, Dict, Tuple

import torch

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule
from src.models.shortcut_module import ShortcutLitModule


class InvertibleShortcutLitModule(BoltzmannGeneratorLitModule):
    """

    TODO - Add a description.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        base_flow_ckpt_path: str,
        d_base: int,
        jarzynski_batch_size: int,  # TODO bit weird this is here but main generation done by data module
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, compile)

        self.base_flow = ShortcutLitModule.load_from_checkpoint(
            base_flow_ckpt_path,  # TODO find a way to pass null in without breaking for sampling
        )

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
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        if not self.hparams.d_base == 0:
            raise NotImplementedError("Only d_base=0 is supported for now")
            # TODO you need to have either multiple networks or time conditioning

        # get a vector of d_base values (all the same)
        d_base = torch.ones(batch.shape[0]) * self.hparams.d_base

        # sample a t value for each sample in the batch
        # following the discretization of d_base
        t = self.base_flow.sample_t(d_base)

        # sample the prior and get xt
        batch_prior = self.prior.sample(batch.shape[0]).to(batch.device)
        xt = self.base_flow.get_xt(batch, batch_prior, t)

        # compute the shortcut vector field
        with torch.no_grad():
            v_shortcuts = self.base_flow.get_bootstrap_targets(xt, t, d_base)

        # apply shortcut vector field to xt
        # dt = 1.0 / 2**self.hparams.d_base
        # x_after_shortcuts = xt + v_shortcuts * dt

        # TODO
        # if we fix the final point to just be the target and we only have a single step
        # we can just regress x1 - x0 directly - no need for the shortcut model

        # boolean array to check if the t corresponds to the final step
        # is_final_step = t > 1.0 - (1.0 / 2 ** self.hparams.d_base + 1e-6)

        # replace the target with the shortcut target if the step is final
        # TODO you prob could save some compute by handling this before computing the shortcuts
        # breakpoint()
        # x_target = torch.where(is_final_step[..., None], batch, xt_after_shortcuts)

        v_pred, *_ = self.forward(xt)

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
