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
from src.models.components.distribution_distances import (
    compute_distribution_distances_with_prefix,
)
from src.models.components.optimal_transport import torus_wasserstein
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
        min_energy=-500,
        max_energy=-250,
        ylim=(0, 0.04),
    ):
        return super().get_dataset_fig(
            samples,
            log_p_samples,
            samples_jarzynski,
            min_energy,
            max_energy,
            ylim=ylim,
        )
