import torch

from src.models.components.smc.base_sampler import SMCSampler


class SMCSamplerHMC(SMCSampler):
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

    def metropolis_hastings(self, source_energy, target_energy, t, x, x_proposal, v, v_proposal):
        energy = lambda _x: self.linear_energy_interpolation(source_energy, target_energy, t, _x)
        norm = lambda _v: torch.sum(_v**2, axis=(-2, -1))
        logp = -0.5 * norm(v_proposal) + 0.5 * norm(v) - energy(x_proposal) + energy(x)
        u = torch.rand_like(logp)
        mask = (logp > torch.log(u))[..., None, None]

        return mask

    def leapfrog(self, source_energy, target_energy, t, x, v, dt):
        grad_energy = lambda _x: self.linear_energy_interpolation_gradients(source_energy, target_energy, t, _x)[0]
        v = v - 0.5 * dt * grad_energy(x)
        x = x + dt * v
        v = v - 0.5 * dt * grad_energy(x)
        return x, v

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        # get step size for langevin
        eps = self.langevin_eps_fn(t)

        # sample momentum from standard gaussian
        v = torch.randn_like(x)

        dlogw = self.target_energy(x) - self.source_energy(x)

        # update the samples
        x_proposal, v_proposal = self.leapfrog(source_energy, target_energy, t, x, v, eps)

        mask = self.metropolis_hastings(source_energy, target_energy, t, x, x_proposal, v, v_proposal)

        x = mask * x_proposal + (1 - mask) * x

        # update weights
        logw = logw - dt * dlogw
        return x, logw
