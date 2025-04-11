import torch
from torch.distributions import Normal


class NormalDistribution:
    def __init__(self, dim: int = 3, mean: float = 0.0, std: float = 1.0):
        self.dim = dim
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)

    def sample(self, num_samples: int, num_particles: int) -> torch.Tensor:
        return self.distribution.sample((num_samples, num_particles, self.dim)).reshape(num_samples, -1)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pointwise_energy = -self.distribution.log_prob(x)
        if mask is not None:
            pointwise_energy = pointwise_energy * mask

        return pointwise_energy.reshape(x.shape[0], -1).sum(dim=-1)
