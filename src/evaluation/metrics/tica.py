import pickle

import deeptime as dt
import mdtraj as md
import numpy as np
import torch

from src.evaluation.metrics.distribution_distances import distribution_distances

SELECTION = "symbol == C or symbol == N or symbol == S"


def compute_distances(xyz):
    distance_matrix_ca = np.linalg.norm(xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1)
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def wrap(array):
    return (np.sin(array), np.cos(array))


def tica_features(trajectory, use_dihedrals=True, use_distances=True, selection=SELECTION):
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    if use_dihedrals:
        _, phi = md.compute_phi(trajectory)
        _, psi = md.compute_psi(trajectory)
        _, omega = md.compute_omega(trajectory)
        dihedrals = np.concatenate([*wrap(phi), *wrap(psi), *wrap(omega)], axis=-1)
    if use_distances:
        distances = compute_distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([distances, dihedrals], axis=-1)
    elif use_distances:
        return distances
    elif use_dihedrals:
        return dihedrals
    else:
        return []


def run_tica(trajectory, lagtime=100, dim=2):
    ca_features = tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def tica_metric(true_samples, pred_samples, topology, tica_model_path, prefix=""):
    with open(tica_model_path, "rb") as f:
        tica_model = pickle.load(f)  # noqa: S301
    true_traj_samples = md.Trajectory(true_samples.cpu().numpy(), topology=topology)
    pred_traj_samples = md.Trajectory(pred_samples.cpu().numpy(), topology=topology)
    features_test = tica_features(true_traj_samples)
    features = tica_features(pred_traj_samples)
    n = min(len(features_test), len(features))
    tics_test = torch.Tensor(tica_model.transform(features_test))[:n, 0:2]
    tics = torch.Tensor(tica_model.transform(features))[:n, 0:2]
    return distribution_distances(tics_test, tics, prefix=prefix + "/tica")
