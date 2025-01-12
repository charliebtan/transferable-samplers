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
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
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
        super().__init__(net, optimizer, scheduler, datamodule, compile, jarzynski_batch_size)

        if base_flow_ckpt_path is None:
            raise ValueError("base_flow_ckpt_path must be provided")
        self.base_flow = ShortcutLitModule.load_from_checkpoint(
            base_flow_ckpt_path,  # TODO find a way to pass null in without breaking for sampling
            datamodule=datamodule
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

        if False:
            # (alex) This is like 5x faster on A100 cause of inplace ops in reverse
            x_pred, _ = self.net.forward(batch_target.detach())
            loss = self.criterion(x_pred, batch_prior)
        else:
            x_pred = self.net.reverse(batch_prior.detach())
            loss = self.criterion(x_pred, batch_target)

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
        x0_pred, logdets = self.net(x_pred)
        if False:
            # Unclear which option is better
            log_p = prior_log_p.flatten() + logdets.flatten()
        else:
            log_p = -self.prior.energy(x0_pred).flatten() - logdets.flatten()
        return x_pred, log_p, torch.empty(0)

if __name__ == "__main__":
    _ = InvertibleShortcutLitModule(None, None, None, None)
