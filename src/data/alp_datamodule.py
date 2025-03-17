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
from src.models.components.utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
    resample,
)

logger = logging.getLogger(__name__)


class ALPDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/alanine/",
        data_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        filename: str = "AlaAlaAla_310K.npy",
        pdb_filename: str = "AlaAlaAla_310K.pdb",
        atom_encoding_filename: str = "atom_types_ecoding.npy",
        n_particles: int = 33,
        n_dimensions: int = 3,
        com_augmentation: bool = False,
        dim: int = 99,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 1.0,
        make_iid: bool = False,
        repeat_factor: int = 1,
    ):
        super().__init__(
            data_dir=data_dir,
            data_url=data_url,
            filename=filename,
            n_particles=n_particles,
            n_dimensions=n_dimensions,
            dim=dim,
            repeat_factor=repeat_factor,
        )
        assert dim == n_particles * n_dimensions

        # com is added once std known
        self.transforms = Random3DRotationTransform(self.n_particles, self.n_dimensions)

        self.scaling = scaling

        self.batch_size_per_device = batch_size

        self.adj_list = None
        self.atom_types = None
        self.atom_encoding_filename = atom_encoding_filename
        self.atom_types_encoding = np.load(
            f"{self.hparams.data_dir}/{self.hparams.atom_encoding_filename}", allow_pickle=True
        ).item()

        self.pdb_path = f"{self.hparams.data_dir}/{pdb_filename}"
        logger.info(f"Loading pdb file from {self.pdb_path}")
        self.topology = md.load_topology(self.pdb_path)
        self.pdb = app.PDBFile(self.pdb_path)

        self.adj_list, self.atom_types = self.compute_adj_list_and_atom_types()

        if n_particles != 42:

            forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

            system = forcefield.createSystem(
                self.pdb.topology,
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

        else:

            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.NoCutoff,
                nonbondedCutoff=0.9 * openmm.unit.nanometer,
                constraints=None,
            )
            temperature = 300
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

        if self.n_particles == 42:
            data = data[:700000]
        elif self.n_particles == 33:
            data = data[:300000]

        data = self.zero_center_of_mass(data)

        train_data = data[:100000]
        test_data = data[100000:]
        self.original_test_data = data[120_000:]

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
        test_data = self.normalize(test_data)

        # split the data
        self.data_train = TransformDataset(
            train_data.repeat(self.repeat_factor, 1), transform=self.transforms
        )

        self.data_val, self.data_test = test_data[:20_000], test_data[20_000:]

        val_rng = np.random.default_rng(0)
        self.data_val = torch.tensor(val_rng.permutation(self.data_val))

        test_rng = np.random.default_rng(1)
        self.data_test = torch.tensor(test_rng.permutation(self.data_test))[:100_000]

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        use_com_energy: bool = False,
        min_energy=-20,
        max_energy=80,
        ylim=(0, 0.1),
    ):
        if self.n_particles == 63:
            min_energy = -130
            max_energy = -30
            ylim = (0, 0.1)
        if self.n_particles == 53:
            min_energy = -150
            max_energy = -60
            ylim = (0, 0.1)
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
            use_com_energy,
            min_energy,
            max_energy,
            ylim=ylim,
        )

    def compute_adj_list_and_atom_types(self):
        atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
        atom_types = []
        for atom_name in self.topology.atoms:
            atom_types.append(atom_name.name[0])
        atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
        n_particles = len(atom_types)
        adj_list = torch.from_numpy(
            np.array([(b.atom1.index, b.atom2.index) for b in self.topology.bonds], dtype=np.int32)
        )
        return adj_list, atom_types

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
            samples[:num_eval_samples], prefix=prefix + "/rama"
        )
        metrics.update(samples_metrics)
        reference_samples = self.data_test
        chirality_centers = find_chirality_centers(self.adj_list, self.atom_types)
        reference_signs = compute_chirality_sign(
            reference_samples.reshape(-1, self.n_particles, 3)[[1]], chirality_centers
        )
        resampled_samples = resampled_samples.reshape(-1, self.n_particles, 3)
        symmetry_change = check_symmetry_change(
            resampled_samples, chirality_centers, reference_signs
        )
        print("Symmetry change frac:", (symmetry_change).float().mean())
        resampled_samples[symmetry_change] *= -1
        correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
        symmetry_change = check_symmetry_change(
            resampled_samples, chirality_centers, reference_signs
        )
        resampled_samples = resampled_samples[~symmetry_change]
        uncorrectable_symmetry_rate = symmetry_change.sum() / len(symmetry_change)
        resampled_samples = resampled_samples.reshape(-1, self.n_particles * 3)

        metrics.update(
            {
                prefix + "/correct_symmetry_rate": correct_symmetry_rate,
                prefix + "/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
            }
        )

        try:
            self.plot_ramachandran(samples, prefix=prefix + "/rama", wandb_logger=wandb_logger)

            resampled_samples = self.unnormalize(resampled_samples.cpu())
            resampled_metrics = self.get_ramachandran_metrics(
                resampled_samples[:num_eval_samples], prefix=prefix + "/resampled/rama"
            )
            metrics.update(resampled_metrics)
            logging.info("Ramachandran metrics computed (resampled)")
            self.plot_ramachandran(
                resampled_samples, prefix=prefix + "/resampled/rama", wandb_logger=wandb_logger
            )
        except ValueError as e:
            logging.error(f"Error in plotting Ramachandran: {e}")

        if samples_jarzynski is not None:
            samples_jarzynski = self.unnormalize(samples_jarzynski).cpu()
            samples_jarzynski_metrics = self.get_ramachandran_metrics(
                samples_jarzynski[:num_eval_samples], prefix=prefix + "/jarzynski/rama"
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

        logging.info("Ramachandran metrics computed")

        return metrics

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples)

        if "val" in prefix:
            eval_samples = self.data_val[: x_pred.shape[0]]
        elif "test" in prefix:
            eval_samples = self.data_test[: x_pred.shape[0]]
        else:
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
            cbar = fig.colorbar(im)  # , ticks=ticks)
            im.set_clim(vmax=samples.shape[0] // 20)
            cbar.ax.set_ylabel(f"Count, max = {int(h.max())}", fontsize=18)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran_simple/{i}", [fig])

        return fig
