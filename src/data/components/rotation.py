import torch


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


if __name__ == "__main__":
    # Example usage
    batch_size = 5
    rotation_matrices = create_random_rotation_matrix(batch_size)
    print(rotation_matrices.shape)  # Should print (5, 3, 3)
