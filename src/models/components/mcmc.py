import math

import torch
import torch.nn as nn


class MCMC(nn.Module):
    AVAIL_MCMC = ["ula", "mala", "hmc"]

    def __init__(
        self,
        source_energy: callable,
        target_energy: callable,
        mcmc_type: str = "ula",
        warmup: float = 0.1,
        langevin_eps: float = 1e-7,
    ):
        super().__init__()
        assert mcmc_type in self.AVAIL_MCMC
        self.source_energy = source_energy
        self.target_energy = target_energy
        self.langevin_eps = langevin_eps
        self.warmup = warmup
        self.mcmc_type = mcmc_type

    def langevin_eps_fn(self, t):
        if t < self.warmup:
            return (self.langevin_eps * t) / self.warmup
        else:
            return self.langevin_eps

    def linear_energy_interpolation(self, t, x, encoding):
        source_energy = self.source_energy(x, encoding=encoding)
        target_energy = self.target_energy(x)
        target_energy = target_energy.reshape(-1)
        assert source_energy.shape == (x.shape[0],), f"Source energy should be a flat vector not {source_energy.shape}"
        assert target_energy.shape == (x.shape[0],), f"Target energy should be a flat vector, not {target_energy.shape}"
        energy = (1 - t) * source_energy + t * target_energy
        return energy

    def linear_energy_interpolation_gradients(self, t, x, encoding):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True

            et = self.linear_energy_interpolation(x, t, encoding=encoding)

            assert et.requires_grad, "et should require grad - check the energy function for no_grad"

            # this is a bit hacky but is fine as long as
            # the energy function is defined properly and
            # doesn't mix batch items
            x_grad, t_grad = torch.autograd.grad(et.sum(), (x, t))

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"
            assert t_grad.shape == t.shape, "t_grad should have the same shape as t"

        assert x_grad is not None, "x_grad should not be None"
        assert t_grad is not None, "t_grad should not be None"

        return x_grad, t_grad

    def ula(self, t, x, encoding, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(t, x, encoding=encoding)
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        dlogw = -energy_grad_t * dt

        x = x + dx
        logw = logw + dlogw
        return x, logw

    def mala(self, t, x, encoding, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # get the energy gradients
        energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(t, x, encoding=encoding)
        dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
        dlogw = -energy_grad_t * dt

        x_proposal = x + dx
        logw_proposal = logw + dlogw

        # get the energy gradients for x_proposal
        energy_grad_x_proposal, _ = self.linear_energy_interpolation_gradients(t, x_proposal, encoding=encoding)

        E_proposal = self.linear_energy_interpolation(t, x_proposal, encoding)
        E = self.linear_energy_interpolation(t, x, encoding)
        logp = -E_proposal + E
        logp += -0.5 * torch.sum((x - x_proposal + eps * energy_grad_x_proposal) ** 2, dim=(1, 2)) / (2 * eps)
        logp -= -0.5 * torch.sum((x_proposal - x + eps * energy_grad_x) ** 2, dim=(1, 2)) / (2 * eps)

        u = torch.rand_like(logp)
        mask = (logp > torch.log(u))[..., None, None]

        x = mask * x_proposal + (1 - mask) * x
        logw = mask * logw_proposal + (1 - mask) * logw

        return x, logw

    def hmc(self):
        NotImplementedError("HMC not implemented")

    def forward(self, t, x, encoding, logw, dt):
        if self.mcmc_type == "ula":
            return self.ula(t, x, encoding, logw, dt)
        elif self.mcmc_type == "mala":
            return self.mala(t, x, encoding, logw, dt)
        elif self.mcmc_type == "hmc":
            return self.hmc(t, x, encoding, logw, dt)
        else:
            raise ValueError(
                f"{self.mcmc_type} is not implemented. List of available mcmc algorthms: {self.AVAIL_MCMC}"
            )
