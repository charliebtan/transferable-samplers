import torch
from torch.distributions import Normal


class NormalDistribution:
    def __init__(self, num_dimensions: int = 3, mean: float = 0.0, std: float = 1.0, mean_free: bool = False):
        self.num_dimensions = num_dimensions
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)
        self.mean_free = mean_free

    def sample(self, num_samples: int, num_particles: int) -> torch.Tensor:
        x = self.distribution.sample((num_samples, num_particles, self.num_dimensions))
        if self.mean_free:
            x = x - x.mean(dim=-1, keepdim=True)
        return x.reshape(num_samples, -1)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 2
        if self.mean_free:
            x = x.reshape(x.shape[0], -1, self.num_dimensions)
            x = x - x.mean(dim=-1, keepdim=True)
            x = x.reshape(x.shape[0], -1)

        pointwise_energy = -self.distribution.log_prob(x)

        if mask is not None:
            pointwise_energy = pointwise_energy * torch.repeat_interleave(mask, self.num_dimensions, dim=-1)

        return pointwise_energy.sum(dim=-1, keepdims=True)


if __name__ == "__main__":
    normal_dist = NormalDistribution(dim=3, mean=0.0, std=1.0, mean_free=True)
    samples = normal_dist.sample(num_samples=10, num_particles=5)
    print("Samples", samples)
    print("Samples shape", samples.shape)
    energy = normal_dist.energy(samples)
    print("Energy:", energy)
    print("Energy shape", energy.shape)
