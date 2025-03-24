from lightning.pytorch.loggers import WandbLogger
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

def plot_energies(
    test_samples_energy,
    proposal_samples_energy,
    resampled_samples_energy,
    jarzynski_samples_energy,
    ylim=None,
    min_energy=None,
    max_energy=None,
    prefix="",
    wandb_logger: WandbLogger = None,
):

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    energy_cropper = lambda x: torch.clamp(x, max=max_energy - 0.1) if max_energy else lambda x: x
    bin_edges = np.linspace(min_energy, max_energy, 100)

    ax.hist(
        energy_cropper(test_samples_energy.cpu()),
        bins=bin_edges,
        density=True,
        alpha=0.4,
        color="g",
        histtype="step",
        linewidth=3,
        label="True data",
    )
    if proposal_samples_energy is not None:
        ax.hist(
            energy_cropper(proposal_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=3,
            label="Proposal",
        )
    if resampled_samples_energy is not None:
        ax.hist(
            energy_cropper(resampled_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="b",
            label="Proposal (reweighted)",
        )
    if jarzynski_samples_energy is not None:
        ax.hist(
            energy_cropper(jarzynski_samples_energy.cpu()),
            bins=bin_edges,
            density=True,
            alpha=0.4,
            histtype="step",
            linewidth=3,
            color="orange",
            label="SBG",
        )

    xticks = list(ax.get_xticks())
    xticks = xticks[1:-1]
    new_tick = bin_edges[-1]
    custom_label = rf"$\geq {new_tick}$"
    xticks.append(new_tick)
    xtick_labels = [
        str(int(tick)) if tick != new_tick else custom_label for tick in xticks
    ]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.xlabel(r"$\mathcal{E}(x)$", labelpad=-5)  # , fontsize=35)
    plt.ylabel("Normalized Density")  # , fontsize=35)
    plt.legend()  # fontsize=30)

    fig.canvas.draw()

    wandb_logger.log_image(f"{prefix}generated_samples", [fig])



