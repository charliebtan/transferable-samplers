import logging

import torch

# check if chirality is the same
# if not --> mirror
# if still not --> discard
def find_chirality_centers(
    adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2
) -> torch.Tensor:
    """
    Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reportet.
            Default is 2, because in this case the mirroring is a simple permutation.

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[
        0
    ]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2].long()
        adj_types = atom_types[bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list).long()


def compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """
    Compute indicator signs for a given configuration.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    # print(coords.shape, chirality_centers.shape, chirality_centers)
    direction_vectors = (
        coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    )
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)

def check_symmetry_change(
    coords: torch.Tensor, chirality_centers: torch.Tensor, reference_signs: torch.Tensor
) -> torch.Tensor:
    """
    Check for a batch if the chirality changed wrt to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers
        reference_signs: List of reference sign for the chirality_centers
    Returns:
        Mask, where changes are True
    """
    perm_sign = compute_chirality_sign(coords, chirality_centers)
    return (perm_sign != reference_signs.to(coords)).any(dim=-1)

def check_symmetry(true_samples, pred_samples, adj_list, atom_types, prefix=""):

    chirality_centers = find_chirality_centers(adj_list, atom_types)
    reference_signs = compute_chirality_sign(
        true_samples[[1]], chirality_centers
    )
    symmetry_change = check_symmetry_change(
        pred_samples, chirality_centers, reference_signs
    )
    pred_samples[symmetry_change] *= -1
    correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
    symmetry_change = check_symmetry_change(
        pred_samples, chirality_centers, reference_signs
    )
    uncorrectable_symmetry_rate = symmetry_change.sum() / len(symmetry_change)

    metrics = {
        prefix + "/correct_symmetry_rate": correct_symmetry_rate,
        prefix + "/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
    }

    if uncorrectable_symmetry_rate > 0.1:
        logging.warning(
            f"Uncorrectable symmetry rate is {uncorrectable_symmetry_rate:.2f}, "
        )

    return metrics, symmetry_change