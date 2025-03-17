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
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import LogNorm
from openmm import app

from src.data.base_datamodule import BaseDataModule
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.transform_dataset import TransformDataset
from src.data.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
)
from src.data.components.optimal_transport import torus_wasserstein
from src.models.components.utils import resample


class ChignolinDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/chignolin/",
        data_url: str = "https://osf.io/download/rk2ht/?view_only=af935a79a5e645b7aab5d37bc5eb3faa",
        filename: str = "chignolin-0_all.h5",
        n_particles: int = 166,
        n_dimensions: int = 3,
        com_augmentation: bool = False,
        dim: int = 498,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 1.0,
    ):
        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            filename=filename,
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            dim=dim,
        )

        assert dim == n_particles * n_dimensions

        # com is added once std known
        self.transforms = Random3DRotationTransform(self.n_particles, self.n_dimensions)

        self.scaling = scaling

        self.batch_size_per_device = batch_size

        self.adj_list = None
        self.atom_types = None

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        traj_samples = md.load(f"{self.hparams.data_dir}/{self.hparams.filename}")
        self.topology = traj_samples.topology

        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

        system = forcefield.createSystem(
            self.topology.to_openmm(),
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )

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

        data = torch.tensor(traj_samples.xyz).float()
        data = data.reshape(-1, self.dim)

        self.topology = traj_samples.topology

        data = self.zero_center_of_mass(data)

        train_data = data[:100000]
        test_data = data[100000:]

        # compute std on only train data
        self.std = train_data.std()

        if self.hparams.com_augmentation:
            self.transforms = torchvision.transforms.Compose(
                [
                    self.transforms,
                    CenterOfMassTransform(self.n_particles, self.n_dimensions, 0.2),
                ]
            )

        # standardize the data
        train_data = self.normalize(train_data)
        test_data = self.normalize(test_data)

        # split the data
        self.data_train = TransformDataset(train_data, transform=self.transforms)

        self.data_val, self.data_test = test_data[:20_000], test_data[20_000:]

        val_rng = np.random.default_rng(0)
        self.data_val = torch.tensor(val_rng.permutation(self.data_val))

        test_rng = np.random.default_rng(1)
        self.data_test = torch.tensor(test_rng.permutation(self.data_test))[:100_000]

        if self.topology is None:
            raise RuntimeError("Topology not set.")

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        use_com_energy: bool = False,
        min_energy=-500,
        max_energy=500,
        ylim=(0, 0.04),
    ):
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            use_com_energy,
            min_energy,
            max_energy,
            ylim=ylim,
        )

    def log_on_epoch_end(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        num_eval_samples: int = 5000,
        use_com_energy: bool = False,
        loggers=None,
        prefix: str = "",
    ) -> None:
        wandb_logger = self.get_wandb_logger(loggers)
        super().log_on_epoch_end(
            samples,
            log_p_samples,
            samples_jarzynski,
            use_com_energy=use_com_energy,
            loggers=loggers,
            prefix=prefix,
        )
        logging.info("Base plots done")

        metrics = {}
        resampled_samples = resample(samples, -self.energy(samples, use_com_energy=use_com_energy) - log_p_samples)
        samples = self.unnormalize(samples).cpu()
        samples_metrics = self.get_ramachandran_metrics(
            samples[:num_eval_samples],
            prefix=prefix + "/rama"
            )

        logging.info("Ramachandran metrics computed")

        self.plot_ramachandran(
            samples, prefix=prefix + "/rama", wandb_logger=wandb_logger
        )
        self.plot_ramachandran(samples, prefix=prefix + "/rama", wandb_logger=wandb_logger)
        metrics.update(samples_metrics)

        resampled_samples = self.unnormalize(resampled_samples.cpu())
        resampled_metrics = self.get_ramachandran_metrics(
            resampled_samples[:num_eval_samples], prefix=prefix + "/resampled/rama"
        )
        logging.info("Ramachandran metrics computed (resampled)")
        self.plot_ramachandran(
            resampled_samples, prefix=prefix + "/resampled/rama", wandb_logger=wandb_logger
        )
        metrics.update(resampled_metrics)

        if samples_jarzynski is not None:
            samples_jarzynski = self.unnormalize(samples_jarzynski).cpu()
            samples_jarzynski_metrics = self.get_ramachandran_metrics(
                samples_jarzynski[:num_eval_samples],
                prefix=prefix + "/jarzynski/rama"
            )
            logging.info("Ramachandran metrics computed (jarzynski)")
            self.plot_ramachandran(
                samples_jarzynski,
                prefix=prefix + "/jarzynski/rama",
                wandb_logger=wandb_logger,
            )
            metrics.update(samples_jarzynski_metrics)

        if "val" in prefix:
            self.plot_ramachandran(
                self.data_val, prefix=prefix + "/ground_truth/rama", wandb_logger=wandb_logger
            )
        elif "test" in prefix:
            self.plot_ramachandran(
                self.data_test, prefix=prefix + "/ground_truth/rama", wandb_logger=wandb_logger
            )

        return metrics

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples)

        if "val" in prefix:
            eval_samples = self.data_val[: x_pred.shape[0]]
        elif "test" in prefix:
            eval_samples = self.data_test[: x_pred.shape[0]]

        x_true = self.get_phi_psi_vectors(self.unnormalize(eval_samples))

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
        for i in range(phis.shape[1]):
            print(f"Plotting Ramachandran {i} out of {phis.shape[1]}")
            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
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
                wandb_logger.log_image(f"{prefix}/ramachandran/{i}", [fig])

            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
            )
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im) #, ticks=ticks)
            im.set_clim(vmax=samples.shape[0] // 20)
            cbar.ax.set_ylabel(f"Count, max = {int(h.max())}", fontsize=18)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran_simple/{i}", [fig])

        return fig