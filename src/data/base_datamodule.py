import math
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from lightning import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        dim: int,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `BaseDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.dim = dim

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.current_epoch = 0

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def zero_center_of_mass(self, x):
        assert x.shape[-1] == self.dim
        x = x.view(-1, self.n_particles, self.n_dimensions)
        x = x - x.mean(axis=1, keepdims=True)
        x = x.view(-1, self.dim)
        return x

    def normalize(self, x):
        assert x.shape[-1] == self.dim
        assert self.std is not None, "Standard deviation should be computed first"
        assert self.std.numel() == 1, "Standard deviation should be scalar"
        x = x.view(-1, self.n_particles, self.n_dimensions)
        x = x - x.mean(axis=1, keepdims=True)
        x = x / self.std
        x = x.view(-1, self.dim)
        return x

    def unnormalize(self, x):
        assert x.shape[-1] == self.dim
        assert self.std is not None, "Standard deviation should be computed first"
        assert self.std.numel() == 1, "Standard deviation should be scalar"
        x = x * self.std.to(x)
        return x

    def energy(self, x, use_com_energy=False):

        if use_com_energy:

            # logging.info("Using CoM energy")

            sigma = self.proposal_com_std

            # self.std is the std dev of com augmentation in normalised scale
            com = x.view(-1, self.n_particles, self.n_dimensions).mean(axis=1)
            com_norm = com.norm(dim=-1)
            com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
                com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
            )

        x = self.unnormalize(x)
        energy = self.potential.energy(x).flatten()

        if use_com_energy:
            energy = energy + com_energy

        return energy

    def get_loggers(self, logger):
        for l in logger:
            if isinstance(l, WandbLogger):
                self.wandb_logger = l
                break

    def plot_energies(
        self,
        test_samples_energy,
        proposal_samples_energy,
        resampled_samples_energy,
        jarzynski_samples_energy,
        ylim=None,
        prefix="",
    ):

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300, constrained_layout=True)
        fig.patch.set_facecolor("white")

        energy_cropper = lambda x: torch.clamp(x, max=self.hparams.hist_max_energy - 0.1) if self.hparams.hist_max_energy else lambda x: x
        bin_edges = np.linspace(self.hparams.hist_min_energy, self.hparams.hist_max_energy, 100)

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
        self.wandb_logger.log_image(f"{prefix}generated_samples", [fig])

    def interatomic_dist(self, x):

        x = self.unnormalize(x)

        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.n_particles, self.n_dimensions)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)

        return dist.flatten()

    def plot_atom_distances(
        self,
        true_samples,
        proposal_samples,
        resampled_samples,
        jarzynski_samples,
        ylim=None,
        prefix="",
    ):

        true_samples_dist = self.interatomic_dist(true_samples).cpu()
        min_dist = true_samples_dist.min()
        max_dist = true_samples_dist.max()

        if proposal_samples is not None:
            proposal_samples_dist = self.interatomic_dist(proposal_samples).cpu()
            min_dist = min(min_dist, proposal_samples_dist.min())
            max_dist = max(max_dist, proposal_samples_dist.max())
        
        if resampled_samples is not None:
            resampled_samples_dist = self.interatomic_dist(resampled_samples).cpu()
            min_dist = min(min_dist, resampled_samples_dist.min())
            max_dist = max(max_dist, resampled_samples_dist.max())

        if jarzynski_samples is not None:
            jarzynski_samples_dist = self.interatomic_dist(jarzynski_samples).cpu()
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
        self.wandb_logger.log_image(f"{prefix}generated_samples", [fig])

if __name__ == "__main__":
    _ = BaseDataModule()
