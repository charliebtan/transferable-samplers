from lightning.pytorch.loggers import WandbLogger
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

def interatomic_dist(x):
    
    n_particles = x.shape[1]

    # Compute the pairwise interatomic distances
    # removes duplicates and diagonal
    distances = x[:, None, :, :] - x[:, :, None, :]
    distances = distances[
        :,
        torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1,
    ]
    dist = torch.linalg.norm(distances, dim=-1)

    return dist.flatten()

def plot_atom_distances(
    true_samples,
    proposal_samples,
    resampled_samples,
    jarzynski_samples,
    ylim=None,
    prefix="",
    wandb_logger: WandbLogger = None,
):

    true_samples_dist = interatomic_dist(true_samples).cpu()
    min_dist = true_samples_dist.min()
    max_dist = true_samples_dist.max()

    if proposal_samples is not None:
        proposal_samples_dist = interatomic_dist(proposal_samples).cpu()
        min_dist = min(min_dist, proposal_samples_dist.min())
        max_dist = max(max_dist, proposal_samples_dist.max())
    
    if resampled_samples is not None:
        resampled_samples_dist = interatomic_dist(resampled_samples).cpu()
        min_dist = min(min_dist, resampled_samples_dist.min())
        max_dist = max(max_dist, resampled_samples_dist.max())

    if jarzynski_samples is not None:
        jarzynski_samples_dist = interatomic_dist(jarzynski_samples).cpu()
        min_dist = min(min_dist, jarzynski_samples_dist.min())
        max_dist = max(max_dist, jarzynski_samples_dist.max())

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    bin_edges = np.linspace(min_dist, max_dist, 100)

    ax.hist(
        true_samples_dist,
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=3,
        label="True data",
    )
    if proposal_samples is not None:
        ax.hist(
            proposal_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=3,
            label="Proposal",
        )
    if resampled_samples is not None:
        ax.hist(
            resampled_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="b",
            label="Proposal (reweighted)",
        )
    if jarzynski_samples is not None:
        ax.hist(
            jarzynski_samples_dist,
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="orange",
            label="SBG",
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel("Interatomic Distance  ", labelpad=-2)  # , fontsize=35)
    plt.ylabel("Normalized Density")  # , fontsize=35)
    plt.legend()  # fontsize=30)

    fig.canvas.draw()

    wandb_logger.log_image(f"{prefix}generated_samples", [fig])
