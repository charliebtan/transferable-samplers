import logging
import os
import pickle
from typing import Optional
import numpy as np

import openmm
import openmm.app
import torch
import torchvision

from src.data.base_datamodule import BaseDataModule
from src.data.energy.openmm import OpenMMBridge, OpenMMEnergy
from src.data.datasets.webdataset import build_webdataset
from src.data.transforms.add_encodings import AddencodingsTransform
from src.data.transforms.add_permutations import AddPermutationsTransform
from src.data.transforms.center_of_mass import CenterOfMassTransform
from src.data.transforms.padding import PaddingTransform
from src.data.transforms.rotation import Random3DRotationTransform
from src.data.transforms.standardize import StandardizeTransform
from src.data.preprocessing.preprocessing import (
    prepare_and_cache_pdb_dict,
    prepare_and_cache_topology_dict,
    prepare_and_cache_encodings_dict,
    prepare_and_cache_permutations_dict,
)

class TransferablePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        num_aa_max: int,
        num_dimensions: int,
        num_atoms: int,
        standardization_factor: float,
        com_augmentation: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.pdb_dir = os.path.join(data_dir, "pdbs")
        self.cache_dir = os.path.join(data_dir, "preprocessing_cache")

        self.pdb_dict_pkl_path = os.path.join(self.cache_dir, "pdb_dict.pkl")
        self.topology_dict_pkl_path = os.path.join(self.cache_dir, "topology_dict.pkl")
        self.encodings_dict_pkl_path = os.path.join(self.cache_dir, "encodings_dict.pkl")
        self.permutations_dict_pkl_path = os.path.join(self.cache_dir, "permutations_dict.pkl")

        self.val_data_path = os.path.join(data_dir, "subsampled_trajectories/val")
        self.test_data_path = os.path.join(data_dir, "subsampled_trajectories/test")

    def prepare_data(self) -> None:
        """Download + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.cache_path, exist_ok=True)
        # TODO - download

        # Get all pdb files for training, validation, and test sets.
        pdb_paths = []
        for subset in ["train", "validation", "test"]:
            # Get the paths to the training pdb files
            pdb_dir = os.path.join(self.pdb_dir, subset)
            pdb_paths.extend([
                os.path.join(pdb_dir, filename) for filename in os.listdir(pdb_dir) if filename.endswith(".pdb")
            ])

        # Do data preprocessinging here and cache the results - to be loaded by workers later.
        pdb_dict = prepare_and_cache_pdb_dict(pdb_paths, self.pdb_dict_pkl_path)
        topology_dict = prepare_and_cache_topology_dict(pdb_dict, self.topology_dict_pkl_path)
        _ =  prepare_and_cache_encodings_dict(topology_dict, self.encodings_dict_pkl_path)
        _ = prepare_and_cache_permutations_dict(topology_dict, self.permutations_dict_pkl_path)

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
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    "the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load cached data preprocessinging dict
        with open(self.pdb_dict_pkl_path, "rb") as f:
            self.pdb_dict = pickle.load(f)
        logging.info(f"Loaded pdb dict from {self.pdb_dict_pkl_path}")
        with open(self.topology_dict_pkl_path, "rb") as f:
            self.topology_dict = pickle.load(f)
        logging.info(f"Loaded topology dict from {self.topology_dict_pkl_path}")
        with open(self.encodings_dict_pkl_path, "rb") as f:
            self.encodings_dict = pickle.load(f)
        logging.info(f"Loaded encodings dict from {self.encodings_dict_pkl_path}")
        with open(self.permutations_dict_pkl_path, "rb") as f:
            self.permutations_dict = pickle.load(f)
        logging.info(f"Loaded permutations dict from {self.permutations_dict_pkl_path}")

        # Build train transformations pipeline
        train_transform_list = [
            StandardizeTransform(self.std, self.hparams.num_dimensions),
            Random3DRotationTransform(self.hparams.num_dimensions),
        ]
        if self.hparams.com_augmentation:
            train_transform_list.append(
                CenterOfMassTransform(
                    self.hparams.num_dimensions,
                )
            )
        train_transform_list = train_transform_list + [
            AddencodingsTransform(self.encodings_dict),
            AddPermutationsTransform(self.permutations_dict, self.residue_tokenization_dict),
            PaddingTransform(self.hparams.num_atoms, self.hparams.num_dimensions),
        ]
        train_transforms = torchvision.transforms.Compose(train_transform_list)

        # Build training webdataset
        self.data_train = build_webdataset(
            path=self.hparams.webdataset_path, # TODO
            tar_pattern=self.hparams.webdataset_tar_pattern,
            max_seq_len=self.hparams.num_aa_max,
            transform=train_transforms,
        )

        self.val_sequences = None # TODO
        self.test_sequences = None # TODO

    def setup_potential(self, sequence: str):

        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        nonbondedMethod = openmm.app.CutoffNonPeriodic
        nonbondedCutoff = 2.0 * openmm.unit.nanometer
        temperature = 310

        # Initalize forcefield systemq
        system = forcefield.createSystem(
            self.pdb_dict[sequence].topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=None,
        )

        # Initialize integrator
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )

        # Initialize potential
        potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

        return potential

    def prepare_eval(self, eval_sequence: str):

        subsampled_trajectory_npz = np.load(
            os.path.join(self.val_data_path, f"{eval_sequence}_subsampled.npz")
            if eval_sequence in self.val_sequences
            else os.path.join(self.test_data_path, f"{eval_sequence}_subsampled.npz")
        )

        true_samples = torch.from_numpy(subsampled_trajectory_npz["positions"])
        tica_model = torch.from_numpy(subsampled_trajectory_npz["tica_model"])

        true_samples = self.normalize(true_samples)
        permutations = self.permutations_dict[eval_sequence]
        encodings = self.encodings_dict[eval_sequence]
        potential = self.setup_potential(eval_sequence)
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model
