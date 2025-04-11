import torch
from torch.distributions import Normal


class NormalDistribution:
    def __init__(self, dim: int = 3, mean: float = 0.0, std: float = 1.0, mean_free: bool = False):
        self.dim = dim
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)
        self.mean_free = mean_free

    def sample(self, num_samples: int, num_particles: int) -> torch.Tensor:
        x = self.distribution.sample((num_samples, num_particles, self.dim))
        if self.mean_free:
            x = x - x.mean(dim=-1, keepdim=True)

        return x.reshape(num_samples, -1)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mean_free:
            x = x - x.mean(dim=-1, keepdim=True)

        pointwise_energy = -self.distribution.log_prob(x)
        if mask is not None:
            pointwise_energy = pointwise_energy * mask

        return pointwise_energy.reshape(x.shape[0], -1).sum(dim=-1, keepdims=True)


if __name__ == "__main__":
    normal_dist = NormalDistribution(dim=3, mean=0.0, std=1.0, mean_free=True)
    samples = normal_dist.sample(num_samples=10, num_particles=5)
    print("Samples", samples)
    print("Samples shape", samples.shape)
    energy = normal_dist.energy(samples)
    print("Energy:", energy)
    print("Energy shape", energy.shape)
