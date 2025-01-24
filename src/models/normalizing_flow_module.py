from typing import Tuple

import torch
from bgflow import NormalDistribution

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule


class NormalizingFlowLitModule(BoltzmannGeneratorLitModule):
    def __init__(
        self,
        force_gaussian_loss: bool = True,
        energy_kl_loss: bool = False,
        energy_kl_weight: float = 0.01,
        log_invertibility_error: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(*args, **kwargs)

        self.energy_kl_loss = energy_kl_loss
        self.energy_kl_weight = energy_kl_weight

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = batch
        x0, dlogp = self.net(x1)

        if self.hparams.force_gaussian_loss:
            loss = (0.5 * x0.pow(2)).mean() - dlogp.mean()
        else:
            loss = self.prior.energy(x0).mean() - dlogp.mean()

        if self.energy_kl_loss:
            samples, log_p, _ = self.generate_samples(x1.shape[0])
            energy_loss = self.energy_kl(samples, log_p).mean()
            loss = loss + self.energy_kl_weight * energy_loss
            self.log("Energy Loss", energy_loss.item(), prog_bar=True)
        return loss

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        x_pred, fwd_logdets = self.net(x)
        fwd_logdets = fwd_logdets * self.datamodule.dim # rescale from mean to sum
        return -(-self.prior.energy(x_pred).view(-1) + fwd_logdets.view(-1))

    def energy_kl(self, x: torch.Tensor, model_log_p: torch.Tensor) -> torch.Tensor:
        sample_target_energy = self.datamodule.energy(x)

        # Energy is -1* log_p
        energy_kl = sample_target_energy + model_log_p
        return energy_kl

    def generate_samples(
        self,
        batch_size: int,
        n_timesteps: int = None,
        dummy_ll=False,
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

            x_pred = self.net.reverse(prior_samples)
            x_recon, fwd_logdets = self.net(x_pred)
            fwd_logdets = fwd_logdets * self.datamodule.dim # rescale from mean to sum

            self.log("invert/mse", torch.mean((prior_samples - x_recon) ** 2))
            self.log('invert/max_abs', torch.max(abs(prior_samples - x_recon)))
            self.log('invert/mean_abs', torch.mean(abs(prior_samples - x_recon)))
            self.log('invert/median_abs', torch.median(abs(prior_samples - x_recon)))
            cutoff = 0.01
            self.log(f'invert/fail_count_{cutoff}',
                torch.sum(abs(prior_samples - x_recon) > cutoff).sum().float()
            )
            self.log(f'invert/fail_count_sample_{cutoff}',
                (torch.sum(abs(prior_samples - x_recon) > cutoff, dim=1) > 0).sum().float()
            )
            cutoff = 0.001
            self.log(f'invert/fail_count_{cutoff}',
                torch.sum(abs(prior_samples - x_recon) > cutoff).sum().float()
            )
            self.log(f'invert/fail_count_sample_{cutoff}',
                (torch.sum(abs(prior_samples - x_recon) > cutoff, dim=1) > 0).sum().float()
            )
        
        log_p = prior_log_p.flatten() + fwd_logdets.flatten()

        return x_pred, log_p, torch.empty(0)


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
