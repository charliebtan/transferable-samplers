import math

import torch

from src.models.components.smc.base_sampler import SMCSampler


class SMCSamplerULA(SMCSampler):
    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(source_energy, target_energy, t, x)
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        dlogw = -energy_grad_t * dt

        x = x + dx
        logw = logw + dlogw
        return x, logw
