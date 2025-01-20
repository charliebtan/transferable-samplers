import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm
import torch
import torchvision
from bgflow import OpenMMBridge, OpenMMEnergy
from bgmol.datasets import AImplicitUnconstrained
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import LogNorm
from openmm import app

from src.data.base_datamodule import BaseDataModule
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.transform_dataset import TransformDataset
from src.data.components.utils import align_topology
from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
)
from src.models.components.optimal_transport import torus_wasserstein
from src.models.components.utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
)


class ALPDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/alanine/",
        data_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        filename: str = "AlaAlaAla_310K.npy",
        pdb_filename: str = "AlaAlaAla_310K.pdb",
        n_particles: int = 33,
        n_dimensions: int = 3,
        com_augmentation: bool = False,
        dim: int = 99,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 1.0,
        make_iid: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            filename=filename,
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            dim=dim,
        )

        # com is added once std known
        self.transforms = Random3DRotationTransform(self.n_particles, self.n_dimensions)

        self.scaling = scaling

        self.batch_size_per_device = batch_size
        # yes a hack but only way without changing bgmol

        self.adj_list = None
        self.atom_types = None
        assert dim == n_particles * n_dimensions

        self.pdb_path = f"{self.hparams.data_dir}/{pdb_filename}"
        self.topology = md.load_topology(self.pdb_path)
        self.pdb = app.PDBFile(self.pdb_path)
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

        system = forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        temperature = 300
        if n_particles == 42:
            temperature = 310
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )
        self.openmm_energy = OpenMMEnergy(
            bridge=OpenMMBridge(system, integrator, platform_name="CUDA")
        )
        self.potential = self.openmm_energy

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load the data + tensorize
        data = np.load(f"{self.hparams.data_dir}/{self.hparams.filename}", allow_pickle=True)
        data = data.reshape(-1, self.dim)
        data = torch.tensor(data).float() / self.scaling
        if self.hparams.make_iid:
            rand_idx = torch.randperm(data.shape[0])
            data = data[rand_idx]
        data = data[:300000]
        data = self.zero_center_of_mass(data)

        test_data = data[-100000:]
        train_data = data[:-100000]

        # compute std on only train data
        self.std = train_data.std()

        if self.hparams.com_augmentation:
            self.transforms = torchvision.transforms.Compose(
                [
                    self.transforms,
                    CenterOfMassTransform(self.n_particles, self.n_dimensions, self.std),
                ]
            )

        # standardize the data
        train_data = self.normalize(train_data)
        print("train_data shape", train_data.shape)
        test_data = self.normalize(test_data)

        # split the data
        self.data_train = TransformDataset(train_data[:-100000], transform=self.transforms)
        self.data_val = train_data[-100000:]
        self.data_test = test_data

    def log_on_epoch_end(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        resampled_samples: torch.Tensor = None,
        loggers=None,
        prefix: str = "",
    ) -> None:
        wandb_logger = self.get_wandb_logger(loggers)
        super().log_on_epoch_end(
            samples,
            log_p_samples,
            samples_jarzynski,
            jarzynski_log_p,
            resampled_samples,
            loggers=loggers,
            prefix=prefix,
        )
        samples = self.unnormalize(samples).cpu()
        sample_metrics = self.get_ramachandran_metrics(samples[:5000], prefix=prefix + "/rama")
        if resampled_samples is not None:
            resampled_samples = self.unnormalize(resampled_samples).cpu()
            resampled_sample_metrics = self.get_ramachandran_metrics(
                samples[:5000], prefix=prefix + "/resampled/rama"
            )
            sample_metrics.update(resampled_sample_metrics)
        if samples_jarzynski is not None:
            sample_jarzynski_metrics = self.get_ramachandran_metrics(
                samples_jarzynski[:5000], prefix=prefix + "/rama_jarzynski"
            )
            sample_metrics.update(sample_jarzynski_metrics)
        return sample_metrics

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples)
        x_true = self.get_phi_psi_vectors(self.unnormalize(self.data_test)[: x_pred.shape[0]])

        metrics = compute_distribution_distances_with_prefix(x_true, x_pred, prefix=prefix)
        metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)
        return metrics

    def get_phi_psi_vectors(self, samples):
        samples = samples.reshape(-1, self.n_particles, self.n_dimensions)
        traj_samples = md.Trajectory(samples, topology=self.topology)
        phis = md.compute_phi(traj_samples)[1]
        psis = md.compute_psi(traj_samples)[1]
        x = torch.cat([torch.from_numpy(phis), torch.from_numpy(psis)], dim=1)
        return x

    def plot_ramachandran(self, samples, prefix: str = "", wandb_logger: WandbLogger = None):
        samples = samples.reshape(-1, self.n_particles, self.n_dimensions)
        traj_samples = md.Trajectory(samples, topology=self.topology)
        phis = md.compute_phi(traj_samples)[1]
        psis = md.compute_psi(traj_samples)[1]
        fig, ax = plt.subplots()
        plot_range = [-np.pi, np.pi]
        h, x_bins, y_bins, im = ax.hist2d(
            phis, psis, 100, norm=LogNorm(), range=[plot_range, plot_range], rasterized=True
        )
        ticks = np.array(
            [np.exp(-6) * h.max(), np.exp(-4.0) * h.max(), np.exp(-2) * h.max(), h.max()]
        )
        ax.set_xlabel(r"$\varphi$", fontsize=45)
        # ax.set_title("Boltzmann Generator", fontsize=45)
        ax.set_ylabel(r"$\psi$", fontsize=45)
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_ticks([])
        cbar = fig.colorbar(im, ticks=ticks)
        # cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
        cbar.ax.set_yticklabels([6.0, 4.0, 2.0, 0.0], fontsize=25)

        cbar.ax.invert_yaxis()
        cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
        if wandb_logger is not None:
            wandb_logger.log_image(f"{prefix}/ramachandran", [fig])

        return fig

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        min_energy=-20,
        max_energy=80,
        ylim=(0, 0.1),
    ):
        if self.n_particles == 42:
            min_energy = -20
            max_energy = 80
            ylim = (0, 0.1)
        if self.n_particles == 33:
            min_energy = -200
            max_energy = -100
            ylim = (0, 0.1)
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            jarzynski_log_p,
            min_energy,
            max_energy,
            ylim=ylim,
        )
