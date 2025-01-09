from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from bgflow import MultiDoubleWellPotential
from bgflow.utils import distance_vectors, distances_from_vectors, remove_mean

# define system dimensionality and a target energy/distribution

DIM = 8
N_PARTICLES = 4
N_DIMENSIONS = DIM // N_PARTICLES

# DW parameters
A = 0.9
B = -4
C = 0
OFFSET = 4

TARGET = MultiDoubleWellPotential(DIM, N_PARTICLES, A, B, C, OFFSET, two_event_dims=False)

# define a MCMC sampler to sample from the target energy

dw4_data = np.load("/Users/chatan/fast-tbg/data/dw4-dataidx.npy", allow_pickle=True)
all_data = remove_mean(dw4_data[0], N_PARTICLES, N_DIMENSIONS)
idx = dw4_data[1]
DATA_HOLDOUT = all_data[idx[-500000:]]


def distance_fn(x):
    x = x.view(-1, N_PARTICLES, N_DIMENSIONS)
    return distances_from_vectors(distance_vectors(x)).reshape(-1)


def energy_histogram(
    samples_proposal: torch.Tensor,
    importance_weights: torch.Tensor,
    samples_jarzynski: torch.Tensor = None,
    jarzynski_weights: torch.Tensor = None,
    save_path: str = None,
) -> None:
    """I only made this for DW4 for now."""

    energies_data = TARGET.energy(DATA_HOLDOUT).detach().cpu().numpy()
    energies_proposal = TARGET.energy(samples_proposal).detach().cpu().numpy()

    min_energy = min(energies_data.min(), energies_proposal.min())

    plt.figure(figsize=(13, 8))

    plt.hist(
        energies_data,
        bins=100,
        density=True,
        range=(min_energy, 0),
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=4,
        label="True data",
    )
    plt.hist(
        energies_proposal,
        bins=100,
        density=True,
        range=(min_energy, 0),
        alpha=0.4,
        histtype="step",
        linewidth=4,
        color="r",
        label="Proposal",
    )
    plt.hist(
        energies_proposal,
        bins=100,
        density=True,
        range=(min_energy, 0),
        alpha=0.4,
        histtype="step",
        linewidth=4,
        color="b",
        label="Proposal (reweighted)",
        weights=importance_weights,
    )

    if samples_jarzynski is not None:
        energies_jarzynski = TARGET.energy(samples_jarzynski).detach().cpu().numpy()
        plt.hist(
            energies_jarzynski,
            bins=100,
            density=True,
            range=(min_energy, 0),
            alpha=0.4,
            histtype="step",
            linewidth=4,
            color="r",
            label="Jarzynski",
        )
        # plt.hist(
        #     energies_jarzynski,
        #     bins=100,
        #     density=True,
        #     range=(min_energy, 0),
        #     alpha=0.4,
        #     histtype="step",
        #     linewidth=4,
        #     color="b",
        #     label="Jarzynski (reweighted)",
        #     weights=jarzynski_weights,
        # )

    plt.xlabel("u(x)", fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.legend(fontsize=25)

    plt.savefig(save_path) if save_path else None


def distance_histogram(
    samples_proposal: torch.Tensor,
    importance_weights: torch.Tensor,
    samples_jarzynski: torch.Tensor = None,
    jarzynski_weights: torch.Tensor = None,
    save_path: str = None,
) -> None:
    """I only made this for DW4 for now."""

    distances_data = distance_fn(DATA_HOLDOUT).detach().cpu().numpy()
    distances_proposal = distance_fn(samples_proposal).detach().cpu().numpy()

    importance_weights = torch.repeat_interleave(
        importance_weights.flatten(), N_PARTICLES * (N_PARTICLES - 1)
    )

    # TODO maybe useful?
    # def distance_energy(d):
    #     d = d - offset
    #     return c * d**4 + b * d**2

    # d = torch.linspace(1, 7, 1000).view(-1, 1) + 1e-6
    # u = torch.exp(-(distance_energy(d).view(-1, 1) - offset)).sum(dim=-1, keepdim=True) * d.abs() ** (
    #     dim // n_particles - 1
    # )
    # Z = (u * 1 / (len(d) / (d.max() - d.min()))).sum()
    # e = u / Z  # * 1.1

    plt.figure(figsize=(16, 9))

    # plt.plot(d, e, label="Groundtruth", linewidth=4, alpha = 0.5)

    plt.hist(
        distances_data,
        bins=100,
        label="holdout samples",
        alpha=0.5,
        density=True,
        histtype="step",
        linewidth=4,
    )
    plt.hist(
        distances_proposal,
        bins=100,
        label="Proposal",
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=4,
        range=(0, 8),
    )
    plt.hist(
        distances_proposal,
        bins=100,
        label="Proposal (reweighted)",
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=4,
        weights=importance_weights,
        range=(0, 8),
    )
    if samples_jarzynski is not None:
        distances_jarzynski = distance_fn(samples_jarzynski).detach().cpu().numpy()

        jarzynski_weights = torch.repeat_interleave(
            jarzynski_weights.flatten(), N_PARTICLES * (N_PARTICLES - 1)
        )
        plt.hist(
            distances_jarzynski,
            bins=100,
            label="Jarzynski",
            alpha=0.7,
            density=True,
            histtype="step",
            linewidth=4,
            range=(0, 8),
        )
        # plt.hist(
        #     distances_jarzynski,
        #     bins=100,
        #     label="Jarzynski (reweighted)",
        #     alpha=0.7,
        #     density=True,
        #     histtype="step",
        #     linewidth=4,
        #     weights=jarzynski_weights,
        #     range=(0, 8),
        # )

    plt.xlim(0, 7)
    plt.legend(fontsize=25)
    plt.xlabel("Distance", fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.title("Distance distribution", fontsize=45)

    plt.savefig(save_path) if save_path else None
