from typing import Any, Dict, Tuple

import torch
from lightning import LightningDataModule, LightningModule

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule
from src.models.normalizing_flow_module import NormalizingFlowLitModule
from src.models.shortcut_module import ShortcutLitModule


class InvertibleShortcutLitModule(NormalizingFlowLitModule):
    """

    TODO - Add a description.

    """

    def __init__(
        self,
        base_flow_ckpt_path: str,
        d_base: int,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        assert d_base == 0, "Only d_base=0 is supported for now"
        if base_flow_ckpt_path is None:
            raise ValueError("base_flow_ckpt_path must be provided")
        super().__init__(*args, **kwargs)
        self.base_flow = ShortcutLitModule.load_from_checkpoint(
            base_flow_ckpt_path,
            datamodule=self.datamodule,
            jarzynski_sampler=self.hparams.jarzynski_sampler,
            sampling_config=self.hparams.sampling_config,
        )

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
        d_base = torch.ones(batch.shape[0], device=batch.device) * self.hparams.d_base

        # sample a t value for each sample in the batch
        # following the discretization of d_base
        t = self.base_flow.sample_t(d_base).to(batch.device)

        # sample the prior and get xt
        batch_prior = self.prior.sample(batch.shape[0]).to(batch.device)

        # compute the shortcut vector field
        with torch.no_grad():
            v_shortcuts = self.base_flow.get_bootstrap_targets(batch_prior, t, d_base)
            batch_target = batch_prior + v_shortcuts

        x_pred, _ = self.net.forward(batch_prior.detach())
        loss = self.criterion(x_pred, batch_target)

        return loss

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.net.reverse(x)
        x_pred, dlogp = self.net.forward(x0)
        return -(-self.prior.energy(x0).view(-1) - dlogp.view(-1))

    def generate_samples(
        self,
        batch_size: int,
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
        with torch.no_grad():
            x_pred, logdets = self.net(prior_samples)
        log_p = prior_log_p.flatten() + logdets.flatten()

        return x_pred, log_p, torch.empty(0)


if __name__ == "__main__":
    _ = InvertibleShortcutLitModule(None, None, None, None)
