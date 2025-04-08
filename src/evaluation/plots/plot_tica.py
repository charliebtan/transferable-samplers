import matplotlib
import matplotlib.pyplot as plt
import mdtraj as md
from matplotlib.colors import LogNorm

from src.evaluation.metrics.tica import run_tica, tica_features

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


def plot_tic01(ax, tics, tics_lims, cmap="viridis"):
    _ = ax.hist2d(tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap=cmap, rasterized=True)
    ax.set_xlabel("TIC0", fontsize=45)
    ax.set_ylabel("TIC1", fontsize=45)
    ax.set_ylim(tics_lims[:, 1].min(), tics_lims[:, 1].max())
    ax.set_xlim(tics_lims[:, 0].min(), tics_lims[:, 0].max())
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_tica(log_image_fn, true_samples, pred_samples, topology, lagtime, prefix=""):
    true_traj_samples = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj_samples = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)
    tica_model = run_tica(true_traj_samples, lagtime=lagtime)
    features = tica_features(pred_traj_samples)
    tics = tica_model.transform(features)
    fig, ax = plt.subplots()
    ax = plot_tic01(ax, tics, tics_lims=tics)
    log_image_fn(fig, f"{prefix}/tica/plot")
    return fig
