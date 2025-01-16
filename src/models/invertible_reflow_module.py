from typing import Tuple

import torch

from src.models.components.rigid_align import weighted_rigid_align
from src.models.flow_matching_module import FlowMatchLitModule
from src.models.normalizing_flow_module import NormalizingFlowLitModule

class InvertibleReflowModule(NormalizingFlowLitModule):
    def __init__(
        self,
        base_flow_ckpt_path: str,
        aligned_loss_fn: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        if base_flow_ckpt_path is None:
            raise ValueError("base_flow_ckpt_path must be provided")
        super().__init__(*args, **kwargs)
        self.base_flow = FlowMatchLitModule.load_from_checkpoint(
            base_flow_ckpt_path,
            datamodule=self.datamodule,
            jarzynski_sampler=self.hparams.jarzynski_sampler,
            sampling_config=self.hparams.sampling_config,
        )
        self.samples = None
        self.prior_samples = None

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # sample the prior and get xt
        if self.samples is None:
            self.base_flow.net.backup()
            self.base_flow.net.copy_to_model()
            self.samples, self.prior_samples = self.base_flow.batched_generate_samples_no_ll(
                self.hparams.num_reflow_samples, batch_size=self.hparams.reflow_batch_size
            )
            self.evaluate(prefix="base_flow", generator=self.base_flow.batched_generate_samples)
            self.base_flow.net.restore_to_model()
        # Sample random indices from length of samples
        idx = torch.randint(0, self.samples.shape[0], (batch.shape[0],), device=self.device)
        batch_prior = self.prior_samples[idx]
        batch_target = self.samples[idx]
        x_pred, _ = self.net.forward(batch_prior)

        if self.hparams.aligned_loss_fn:
            batch_target = weighted_rigid_align(
                true_coords=batch_target,
                pred_coords=x_pred,
                n_particles=self.datamodule.n_particles,
                n_dimensions=self.datamodule.n_dimensions,
            )

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
    _ = InvertibleReflowModule(None, None, None, None)
