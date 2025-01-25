import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import torch
import torchvision
from bgmol.datasets import AImplicitUnconstrained
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import LogNorm

from src.data.base_datamodule import BaseDataModule
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.transform_dataset import TransformDataset
from src.data.components.utils import align_topology
from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
    energy_distances,
)
from src.models.components.optimal_transport import torus_wasserstein
from src.models.components.utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
    resample,
)


class ALDPDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/AD2/",
        data_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        filename: str = "AD2_weighted.npy",
        n_particles: int = 22,
        n_dimensions: int = 3,
        com_augmentation: bool = False,
        dim: int = 66,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 10.0,
    ) -> None:
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
        # yes a hack but only way without changing bgmol
        self.bgmol_dataset = AImplicitUnconstrained(
            read=True, download=True if "AImplicitUnconstrained" not in os.listdir() else False
        )
        self.potential = self.bgmol_dataset.get_energy_model()
        self.adj_list = None
        self.atom_types = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load the data + tensorize
        train_data = np.load(f"{self.hparams.data_dir}/{self.hparams.filename}", allow_pickle=True)
        test_data = torch.tensor(self.bgmol_dataset.xyz).view(-1, self.dim)

        # data is 10 times larger in bgflow dataset than in numpy
        train_data = torch.tensor(train_data) / self.scaling

        # zero center of mass
        train_data = self.zero_center_of_mass(train_data)
        test_data = self.zero_center_of_mass(test_data)

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
        self.data_train = TransformDataset(train_data, transform=self.transforms)
        self.data_val = test_data[5::10]
        self.data_test = test_data[::10]

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        min_energy=-50,
        max_energy=100,
        ylim=(0, 0.2),
    ):
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            jarzynski_log_p,
            min_energy,
            max_energy,
            ylim=ylim,
        )

    def log_on_epoch_end(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        samples_test: torch.Tensor = None,
        loggers=None,
        prefix: str = "",
    ) -> None:
        wandb_logger = self.get_wandb_logger(loggers)
        super().log_on_epoch_end(
            samples,
            log_p_samples,
            samples_jarzynski,
            jarzynski_log_p,
            loggers=loggers,
            prefix=prefix,
        )
        metrics = {}
        samples_metrics = self.align_and_compute_metrics(
            samples, prefix=prefix + "/rama", wandb_logger=wandb_logger
        )
        metrics.update(samples_metrics)

        resampled_samples = resample(samples, -self.energy(samples) - log_p_samples)
        resampled_metrics = self.align_and_compute_metrics(
            resampled_samples, prefix=prefix + "/resampled/rama", wandb_logger=wandb_logger
        )
        metrics.update(resampled_metrics)

        if samples_jarzynski is not None:
            samples_jarzynski_metrics = self.align_and_compute_metrics(
                samples_jarzynski, prefix=prefix + "/jarzynski/rama", wandb_logger=wandb_logger
            )
            metrics.update(samples_jarzynski_metrics)
        if samples_test is not None:
            self.plot_ramachandran(
                samples_test, prefix=prefix + "/ground_truth/rama", wandb_logger=wandb_logger
            )

        return metrics

    def align_and_compute_metrics(
        self, samples, prefix: str = "", wandb_logger: WandbLogger = None
    ):
        samples = self.unnormalize(samples).cpu()
        aligned_samples, aligned_idxs = self.align_samples(samples)
        correct_config_rate = len(aligned_samples) / len(samples)
        if len(aligned_samples) == 0:
            return {
                f"{prefix}/correct_config_rate": 0,
                f"{prefix}/correct_symmetry_rate": 0,
            }
        symmetry_change = self.get_symmetry_change(aligned_samples)
        correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
        try:
            aligned_symmetric_samples = aligned_samples[~symmetry_change]
            self.plot_ramachandran(
                aligned_symmetric_samples, prefix=prefix, wandb_logger=wandb_logger
            )
            metrics = self.get_ramachandran_metrics(
                aligned_symmetric_samples[:5000], prefix=prefix
            )
            metrics.update(
                {
                    f"{prefix}/correct_config_rate": correct_config_rate,
                    f"{prefix}/correct_symmetry_rate": correct_symmetry_rate,
                }
            )
            return metrics
        except Exception as e:
            logging.warning(
                "Aligned samples:",
                aligned_samples.shape,
                "Symmetry change:",
                symmetry_change.shape,
            )
            logging.warning(e)
            return {
                f"{prefix}/correct_config_rate": -1.0,
                f"{prefix}/correct_symmetry_rate": -1.0,
            }

    def get_symmetry_change(self, aligned_samples):
        aligned_samples = aligned_samples.reshape(-1, self.n_particles, self.n_dimensions)
        topology = self.bgmol_dataset.system.mdtraj_topology
        traj_samples = md.Trajectory(aligned_samples, topology=topology)
        model_samples = torch.from_numpy(traj_samples.xyz)
        adj_list = self.get_adj_list()
        atom_types = self.get_atom_types()
        chirality_centers = find_chirality_centers(torch.from_numpy(adj_list), atom_types)
        reference_signs = compute_chirality_sign(
            self.unnormalize(self.data_test[:1]).reshape(-1, self.n_particles, self.n_dimensions),
            chirality_centers,
        )
        symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
        model_samples[symmetry_change] *= -1
        symmetry_change = check_symmetry_change(model_samples, chirality_centers, reference_signs)
        return symmetry_change

    def get_ramachandran_metrics(self, samples, prefix: str = ""):
        x_pred = self.get_phi_psi_vectors(samples)
        eval_samples = self.data_test[torch.randperm(x_pred.shape[0])]
        x_true = self.get_phi_psi_vectors(self.unnormalize(eval_samples))

        metrics = compute_distribution_distances_with_prefix(x_true, x_pred, prefix=prefix)
        metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)
        return metrics

    def get_phi_psi_vectors(self, samples):
        samples = samples.reshape(-1, self.n_particles, self.n_dimensions)
        traj_samples = md.Trajectory(samples, topology=self.bgmol_dataset.system.mdtraj_topology)
        phis = md.compute_phi(traj_samples)[1].flatten()
        psis = md.compute_psi(traj_samples)[1].flatten()
        x = torch.stack([torch.from_numpy(phis), torch.from_numpy(psis)], dim=1)
        return x

    def plot_ramachandran(self, samples, prefix: str = "", wandb_logger: WandbLogger = None):
        samples = samples.reshape(-1, self.n_particles, self.n_dimensions)
        traj_samples = md.Trajectory(samples, topology=self.bgmol_dataset.system.mdtraj_topology)
        phis = md.compute_phi(traj_samples)[1].flatten()
        psis = md.compute_psi(traj_samples)[1].flatten()
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

    def get_atom_types(self):
        if self.atom_types is not None:
            return self.atom_types
        atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3}
        topology = self.bgmol_dataset.system.mdtraj_topology
        atom_types = []
        for atom_name in topology.atoms:
            atom_types.append(atom_name.name[0])
        atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
        self.atom_types = atom_types
        return self.atom_types

    def get_adj_list(self):
        if self.adj_list is not None:
            return self.adj_list
        topology = self.bgmol_dataset.system.mdtraj_topology
        adj_list = np.array(
            [(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32
        )
        self.adj_list = adj_list
        return self.adj_list

    def align_samples(self, samples):
        aligned_samples = []
        aligned_idxs = []
        adj_list = self.get_adj_list()

        for i, sample in enumerate(samples.reshape(-1, self.n_particles, self.n_dimensions)):
            aligned_sample, is_isomorphic = align_topology(
                sample, adj_list.tolist(), self.get_atom_types()
            )
            if is_isomorphic:
                aligned_samples.append(aligned_sample)
                aligned_idxs.append(i)
        aligned_samples = np.array(aligned_samples)
        return aligned_samples, aligned_idxs
        print(f"Correct configuration rate {len(aligned_samples)/len(samples)}")


if __name__ == "__main__":
    _ = ALDPDataModule()
