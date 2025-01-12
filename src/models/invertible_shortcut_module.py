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

        v_pred, *_ = self.reverse(xt)

        loss = self.criterion(v_pred, v_shortcuts)

        return loss


if __name__ == "__main__":
    _ = InvertibleShortcutLitModule(None, None, None, None)
