import os
from typing import Any, Dict, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from bgflow import MultiDoubleWellPotential
from bgflow.utils import distance_vectors, distances_from_vectors
from lightning import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        data_url: str,
        filename: str,
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

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        if not os.path.exists(self.hparams.data_dir + self.hparams.filename):
            print(f"Downloading file from {self.hparams.data_url}")
            response = requests.get(self.hparams.data_url, timeout=300)

            # Save the file in binary (write) mode
            with open(self.hparams.data_dir + self.hparams.filename, "wb") as f:
                f.write(response.content)

            print(f"File downloaded and saved as: {self.hparams.filename}")

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

    def energy(self, x):
        x = self.unnormalize(x)
        return self.potential.energy(x).flatten()

    def log_on_epoch_end(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        wandb_logger: WandbLogger = None,
        prefix: str = "",
    ) -> None:
        if samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        samples_fig = self.get_dataset_fig(
            samples, log_p_samples, samples_jarzynski, jarzynski_log_p
        )
        wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])
        self.current_epoch += 1

    def interatomic_dist(self, x):
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
        return dist

    def sample_test_set(self, n):
        return self.data_test[torch.randint(0, len(self.data_test), (n,))]

    def get_dataset_fig(
        self,
        samples,
        log_p_samples: torch.Tensor,
        samples_jarzynski: torch.Tensor = None,
        jarzynski_log_p: torch.Tensor = None,
        min_energy=-26,
        max_energy=0,
        ylim=(0, 0.2),
    ):
        test_data_smaller = self.sample_test_set(5000)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        dist_samples = self.interatomic_dist(self.unnormalize(samples)).detach().cpu()
        dist_test = self.interatomic_dist(self.unnormalize(test_data_smaller)).detach().cpu()

        axs[0].hist(
            dist_test.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
            label="True data",
            color="g",
        )
        axs[0].hist(
            dist_samples.view(-1),
            bins=100,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
            label="Proposal",
            color="r",
        )
        if samples_jarzynski is not None:
            dist_samples_jarzynski = (
                self.interatomic_dist(self.unnormalize(samples_jarzynski)).detach().cpu()
            )
            axs[0].hist(
                dist_samples_jarzynski.view(-1),
                bins=100,
                alpha=0.5,
                density=True,
                histtype="step",
                linewidth=4,
                label="Jarzynski",
                color="orange",
            )

        axs[0].set_xlabel("Interatomic distance")

        energy_samples = self.energy(samples)
        logits = -energy_samples.flatten() - log_p_samples.flatten()
        importance_weights = torch.nn.functional.softmax(logits, dim=0).detach().cpu()
        energy_samples = energy_samples.detach().cpu()
        energy_test = self.energy(test_data_smaller).detach().cpu()

        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="g",
            histtype="step",
            linewidth=4,
            label="True data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            range=(min_energy, max_energy),
            color="r",
            histtype="step",
            linewidth=4,
            label="Proposal",
        )
        axs[1].hist(
            energy_samples,
            bins=100,
            density=True,
            range=(min_energy, max_energy),
            alpha=0.4,
            histtype="step",
            linewidth=4,
            color="b",
            label="Proposal (reweighted)",
            weights=importance_weights,
        )
        if samples_jarzynski is not None:
            energies_jarzynski = self.energy(samples_jarzynski)
            jarzynski_logits = -energies_jarzynski.flatten() - jarzynski_log_p.flatten()
            jarzynski_weights = torch.nn.functional.softmax(jarzynski_logits, dim=0).detach().cpu()
            energies_jarzynski = energies_jarzynski.detach().cpu().numpy()

            axs[1].hist(
                energies_jarzynski,
                bins=100,
                density=True,
                range=(min_energy, max_energy),
                alpha=0.4,
                histtype="step",
                linewidth=4,
                color="orange",
                label="Jarzynski",
            )
            axs[1].hist(
                energies_jarzynski,
                bins=100,
                density=True,
                range=(min_energy, max_energy),
                alpha=0.4,
                histtype="step",
                linewidth=4,
                color="grey",
                label="Jarzynski (reweighted)",
                weights=jarzynski_weights,
            )
        axs[1].set_xlabel("u(x)")
        axs[1].legend()
        axs[1].set_ylim(ylim)

        fig.canvas.draw()

        return fig


if __name__ == "__main__":
    _ = BaseDataModule()
