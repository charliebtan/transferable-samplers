import logging
import pickle

import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
from matplotlib.colors import LogNorm

from src.evaluation.metrics.tica import tica_features

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

import numpy as np


def plot_tic01(ax, tics, tics_lims, cmap="viridis", p=""):
    bin_heights, x_edges, y_edges, *_ = ax.hist2d(
        tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap=cmap, rasterized=True
    )
    ax.set_xlabel("TIC0", fontsize=45)
    ax.set_ylabel("TIC1", fontsize=45)
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
    ax.set_xticks([])
    ax.set_yticks([])
    np.savez(
        f"{p}_tica_histogram.npz",
        bin_heights=bin_heights,
        x_edges=x_edges,
        y_edges=y_edges,
        tics=tics,
    )


def plot_tica(log_image_fn, samples, topology, tica_model_path, ref_samples=None, prefix=""):
    logging.info(f"Plotting TICA for {prefix}")
    with open(tica_model_path, "rb") as f:
        tica_model = pickle.load(f)  # noqa: S301
    pred_traj_samples = md.Trajectory(samples.cpu().numpy(), topology=topology)
    features = tica_features(pred_traj_samples)
    tics = tica_model.transform(features)

    if ref_samples is not None:
        ref_traj_samples = md.Trajectory(ref_samples.cpu().numpy(), topology=topology)
        ref_features = tica_features(ref_traj_samples)
        ref_tics = tica_model.transform(ref_features)
    else:
        ref_tics = tics

    fig, ax = plt.subplots()
    p = prefix.split("/")[2]
    ax = plot_tic01(ax, tics, tics_lims=ref_tics, p=p)

    log_image_fn(fig, f"{prefix}/tica/plot")
    plt.close()
