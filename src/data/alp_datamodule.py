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
from src.models.components.utils import (check_symmetry_change,
                                         compute_chirality_sign,
                                         find_chirality_centers)


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

        pdb = app.PDBFile(f"{self.hparams.data_dir}/{pdb_filename}")
        forcefield = app.ForceField("amber14-all.xml", "implicit/obc1.xml")

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            300 * openmm.unit.kelvin, 0.3 / openmm.unit.picosecond, 1.0 * openmm.unit.femtosecond
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
        data = data[:300000]
        data = torch.tensor(data).float() / self.scaling
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
