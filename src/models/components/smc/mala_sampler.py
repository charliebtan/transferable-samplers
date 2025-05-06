import math

import torch

from src.models.components.smc.base_sampler import SMCSampler


class SMCSamplerMALA(SMCSampler):
    def __init__(
        self,
        log_image_fn: callable,
        batch_size: int = 128,
        langevin_eps: float = 1e-7,
        num_timesteps: int = 100,
        ess_threshold: float = -1.0,
        warmup: float = 0.1,
        enabled: bool = False,
        do_energy_plots: bool = False,
        input_energy_cutoff: float = None,
        systematic_resampling: bool = False,
    ):
        super().__init__(
            log_image_fn=log_image_fn,
            batch_size=batch_size,
            langevin_eps=langevin_eps,
            num_timesteps=num_timesteps,
            ess_threshold=ess_threshold,
            warmup=warmup,
            enabled=enabled,
            do_energy_plots=do_energy_plots,
            input_energy_cutoff=input_energy_cutoff,
            systematic_resampling=systematic_resampling,
        )

    def metropolis_hastings(self, source_energy, target_energy, t, x, x_proposal, step_size):
        # get the energy gradients for x_proposal
        energy_grad_x, _ = self.linear_energy_interpolation_gradients(source_energy, target_energy, t, x)
        energy_grad_x_proposal, _ = self.linear_energy_interpolation_gradients(
            source_energy, target_energy, t, x_proposal
        )

        E_proposal = self.linear_energy_interpolation(t, x_proposal)
        E = self.linear_energy_interpolation(t, x)
        logp = -E_proposal + E
        logp += (
            -0.5 * torch.sum((x - x_proposal + step_size * energy_grad_x_proposal) ** 2, dim=(1, 2)) / (2 * step_size)
        )
        logp -= -0.5 * torch.sum((x_proposal - x + step_size * energy_grad_x) ** 2, dim=(1, 2)) / (2 * step_size)

        u = torch.rand_like(logp)
        mask = (logp > torch.log(u))[..., None, None]
        return mask

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(source_energy, target_energy, t, x)
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        x_proposal = x + dx

        mask = self.metropolis_hastings(source_energy, target_energy, t, x, x_proposal, eps)
        x = mask * x_proposal + (1 - mask) * x

        logw = logw - dt * energy_grad_t
        return x, logw
