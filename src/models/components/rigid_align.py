
   # started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from einops import einsum
import torch
import torch.nn.functional as F

def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights=None,
    mask=None,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """

    if weights is None:
        weights = torch.ones(true_coords.shape[0], device=true_coords)

    if mask is None:
        mask = torch.ones(true_coords.shape[0], device=true_coords)

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords
 