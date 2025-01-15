import numpy as np
import torch


class CenterOfMassTransform(torch.nn.Module):
    def __init__(self, num_particles, dim, std):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim
        self.std = std

    def forward(self, data):
        assert len(data.shape) == 1, "only process single molecules"
        data = data.reshape(self.num_particles, self.dim)
        noise = torch.randn_like(data) * self.std
        data = data + noise
        data = data.reshape(self.num_particles * self.dim)

        return data

if __name__ == "__main__":
    pass