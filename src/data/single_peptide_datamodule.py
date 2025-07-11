import logging
import math
from typing import Optional

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
import torchvision

from src.data.base_datamodule import BaseDataModule
from src.data.components.openmm import OpenMMBridge, OpenMMEnergy
from src.data.components.preprocess.preprocess import prepare_tica_models
from src.data.dataset.numpy_dataset import NumpyDataset
from src.data.components.transforms.center_of_mass import CenterOfMassTransform
from src.data.components.transforms.rotation import Random3DRotationTransform
from src.data.components.transforms.standardize import StandardizeTransform


class SinglePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        sequence: str,
        temperature: float,
        num_dimensions: int,
        num_atoms: int,
        com_augmentation: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.train_data_path = f"{data_dir}/{sequence}_{temperature}K_train.npy"
        self.val_data_path = f"{data_dir}/{sequence}_{temperature}K_val.npy"
        self.test_data_path = f"{data_dir}/{sequence}_{temperature}K_test.npy"

        self.pdb_path = f"{self.hparams.data_dir}/{sequence}_{temperature}K.pdb"

        self.tica_models_path = f"{data_dir}/tica_models"

    def prepare_data(self):
        prepare_tica_models(
            {self.hparams.sequence: self.test_data_path},
            {self.hparams.sequence: self.pdb_path},
            dir=self.tica_models_path,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number "
                    "of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load the data
        train_data = np.load(self.train_data_path, allow_pickle=True)
        val_data = np.load(self.val_data_path, allow_pickle=True)
        test_data = np.load(self.test_data_path, allow_pickle=True)

        # Reshape and tensorize the data
        train_data = torch.from_numpy(train_data)
        val_data = torch.from_numpy(val_data)
        test_data = torch.from_numpy(test_data)

        self.topology_dict = {self.hparams.sequence: md.load_topology(self.pdb_path)}
        self.tica_model_paths = {self.hparams.sequence: f"{self.tica_models_path}/{self.hparams.sequence}-tica.pkl"}

        # Standarize data
        train_data = self.zero_center_of_mass(train_data)
        val_data = self.zero_center_of_mass(val_data)
        test_data = self.zero_center_of_mass(test_data)

        self.std = train_data.std()

        # Subsample the evaluation subsets data
        val_data = val_data[:: val_data.shape[0] // self.hparams.num_eval_samples]
        test_data = test_data[:: test_data.shape[0] // self.hparams.num_eval_samples]

        transform_list = [
            StandardizeTransform(self.std, self.hparams.num_dimensions),
            Random3DRotationTransform(self.hparams.num_dimensions),
        ]
        if self.hparams.com_augmentation:
            transform_list.append(
                CenterOfMassTransform(
                    1 / math.sqrt(self.hparams.num_particles),
                    self.hparams.num_dimensions,
                )
            )
        transforms = torchvision.transforms.Compose(transform_list)

        self.data_train = NumpyDataset(
            npy_array=train_data,
            num_dimensions=self.hparams.num_dimensions,
            transform=transforms,
        )

        logging.info(f"Train dataset size: {len(self.data_train)}")
        logging.info(f"Validation dataset size: {len(self.data_val)}")
        logging.info(f"Test dataset size: {len(self.data_test)}")


    def setup_potential(self):
        if self.hparams.sequence in ["Ace-A-Nme", "Ace-AAA-Nme"]:
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
                0.3 / openmm.unit.picosecond
                if self.hparams.sequence == "Ace-AAA-Nme"
                else 1.0 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))
        else:

            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
            temperature = 310

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.CutoffNonPeriodic,
                nonbondedCutoff=2.0 * openmm.unit.nanometer,
                constraints=None,
            )
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * openmm.unit.kelvin,
                0.3 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )

            # Initialize potential
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

        return potential

    def prepare_eval(self, prefix: str):

        if prefix == "test":
            true_samples = self.data_test
        elif prefix == "val":
            true_samples = self.data_val
        else:
            raise ValueError(f"Unknown prefix: {prefix}. Use 'val' or 'test'.")


        true_samples = self.normalize(true_samples)
        tica_model = self.tica_model

        permutations = self.permutations
        encodings = None
        potential = self.setup_potential()
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model
