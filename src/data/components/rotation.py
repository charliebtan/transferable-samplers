import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def create_random_rotation_matrix(batch_size):
    # Generate random numbers from a normal distribution
    u = torch.randn((batch_size, 4))

    # Normalize to get unit quaternions
    norm_u = torch.norm(u, p=2, dim=1, keepdim=True)
    q = u / norm_u

    # Convert quaternions to rotation matrices
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    R[:, 0, 1] = 2 * qx * qy - 2 * qz * qw
    R[:, 0, 2] = 2 * qx * qz + 2 * qy * qw

    R[:, 1, 0] = 2 * qx * qy + 2 * qz * qw
    R[:, 1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    R[:, 1, 2] = 2 * qy * qz - 2 * qx * qw

    R[:, 2, 0] = 2 * qx * qz - 2 * qy * qw
    R[:, 2, 1] = 2 * qy * qz + 2 * qx * qw
    R[:, 2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    return R


class Random3DRotationTransform(torch.nn.Module):
    def __init__(self, num_particles, dim):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim

    def forward(self, data):
        data = data.reshape(-1, self.num_particles, self.dim)  # batch dimension needed for einsum
        # rot = create_random_rotation_matrix(len(data))
        rot = torch.tensor(R.random(len(data)).as_matrix()).to(data)
        data = torch.einsum("bij,bki->bkj", rot, data)
        data = data.reshape(self.num_particles * self.dim)  # don't want to return with batch dim
        return data


class Random2DRotationTransform(torch.nn.Module):
    def __init__(self, num_particles, dim):
        super().__init__()
        self.num_particles = num_particles
        self.dim = dim

    def forward(self, data):
        # rotation augmentation
        data = data.reshape(-1, self.num_particles, self.dim)  # batch dimension needed for einsum
        x = torch.rand(len(data)) * 2 * np.pi
        s = torch.sin(x)
        c = torch.cos(x)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute(2, 0, 1)
        data = torch.einsum("bij,bki->bkj", rot, data)
        data = data.reshape(self.num_particles * self.dim)  # don't want to return with batch dim
        return data


if __name__ == "__main__":
    # Example usage
    batch_size = 5
    rotation_matrices = create_random_rotation_matrix(batch_size)
    print(rotation_matrices.shape)  # Should print (5, 3, 3)
