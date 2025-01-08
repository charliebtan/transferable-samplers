from typing import Callable

import matplotlib.pyplot as plt
import torch


def energy_histogram(
    samples_data: torch.Tensor,
    samples_proposal: torch.Tensor,
    importance_weights: torch.Tensor,
    energy_fn: Callable,
) -> None:
    """I only made this for DW4 for now."""

    energies_data = energy_fn(samples_data).detach().cpu().numpy()
    energies_proposal = energy_fn(samples_proposal).detach().cpu().numpy()

    min_energy = min(energies_data.min(), energies_proposal.min())

    plt.figure(figsize=(13, 8))

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
        color="b",
        label="Proposal (reweighted)",
        weights=importance_weights,
    )

    plt.xlabel("u(x)", fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.legend(fontsize=25)


def distance_histogram(
    samples_data: torch.Tensor,
    samples_proposal: torch.Tensor,
    importance_weights: torch.Tensor,
    distance_fn: Callable,
) -> None:
    """I only made this for DW4 for now."""

    distances_data = distance_fn(samples_data).detach().cpu().numpy()
    distances_proposal = distance_fn(samples_proposal).detach().cpu().numpy()

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
        label="Proposal (rewieghted)",
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=4,
        weights=importance_weights,
        range=(0, 8),
    )
    plt.xlim(0, 7)
    plt.legend(fontsize=25)
    plt.xlabel("Dinstance", fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.title("Distance distribution", fontsize=45)
