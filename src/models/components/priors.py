import torch
from torch.distributions import Normal


class NormalDistribution:
    def __init__(self, num_dimensions: int = 3, mean: float = 0.0, std: float = 1.0):
        self.num_dimensions = num_dimensions
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)

    def sample(self, num_samples: int, num_particles: int) -> torch.Tensor:
        return self.distribution.sample((num_samples, num_particles * self.num_dimensions))

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 2
        pointwise_energy = -self.distribution.log_prob(x)
        if mask is not None:
            pointwise_energy = pointwise_energy * torch.repeat_interleave(mask, self.num_dimensions, dim=-1)

        return pointwise_energy.reshape(x.shape[0], -1).sum(dim=-1)
