import logging
import math
import os
import shutil
from typing import Any, Optional

import numpy as np
import torch
import torchvision

from src.data.components.data_prep import (
    build_lmdb,
    check_files,
    cross_reference_files,
    download_data,
    load_lmdb_metadata,
    load_pdbs_and_topologies,
)
from src.data.components.encoding import get_encoding_dict
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.transforms.add_encoding import AddEncodingTransform
from src.data.components.transforms.atom_noise import AtomNoiseTransform
from src.data.components.transforms.center_of_mass import CenterOfMassTransform
from src.data.components.transforms.padding import PaddingTransform
from src.data.components.transforms.rotation import Random3DRotationTransform
from src.data.components.transforms.standardize import StandardizeTransform
from src.data.transferable_peptide_datamodule import TransferablePeptideDataModule

MEAN_ATOMS_PER_AA = 17.67


class TetraPeptideDataModule(TransferablePeptideDataModule):
    def __init__(
        self,
        data_dir: str,
        huggingface_repo_id: str,
        huggingface_train_data_dir: str,
        huggingface_val_data_dir: str,
        num_aa: int,
        num_dimensions: int,
        num_particles: int,
        dim: int,  # dim of largest system
        com_augmentation: bool = False,
        atom_noise_augmentation_factor: float = 0.0,
        # TODO maybe make this all just *args?
        num_samples_per_seq: int = 10_000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_eval_samples: int = 10_000,
        num_val_sequences: int = 20,
        energy_hist_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            num_aa=num_aa,
            num_dimensions=num_dimensions,
            num_particles=num_particles,
            dim=dim,
            com_augmentation=com_augmentation,
            atom_noise_augmentation_factor=atom_noise_augmentation_factor,
            num_samples_per_seq=num_samples_per_seq,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_eval_samples=num_eval_samples,
            num_val_sequences=num_val_sequences,
            energy_hist_config=energy_hist_config,
        )

        assert dim == num_dimensions * num_particles, "dim must be equal to num_dimensions * num_particles"

        self.train_data_path = f"{self.hparams.data_dir}/train"
        self.val_data_path = f"{self.hparams.data_dir}/val"

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        download_data(
            huggingface_repo_id=self.hparams.huggingface_repo_id,
            huggingface_data_dir=self.hparams.huggingface_train_data_dir,
            local_dir=self.train_data_path,
        )
        download_data(
            huggingface_repo_id=self.hparams.huggingface_repo_id,
            huggingface_data_dir=self.hparams.huggingface_val_data_dir,
            local_dir=self.val_data_path,
        )

        train_npz_paths, train_pdb_paths = check_files(self.train_data_path + "/4AA-large/train")
        val_npz_paths, val_pdb_paths = check_files(self.val_data_path + "/4AA-large/val")

        cross_reference_files(train_npz_paths, val_npz_paths)

        if os.path.exists(self.train_data_path + "/seq.lmdb"):
            shutil.rmtree(self.train_data_path + "/seq.lmdb")
        if not os.path.exists(self.train_data_path + "/4AA-large/train/seq.lmdb"):
            build_lmdb(
                train_npz_paths,
                train_pdb_paths,
                lmdb_path=self.train_data_path + "/seq.lmdb",
            )
        else:
            logging.info(f"LMDB file {self.train_data_path}/seq.lmdb already exists, skipping LMDB creation")

        if os.path.exists(self.val_data_path + "/seq.lmdb"):
            shutil.rmtree(self.val_data_path + "/seq.lmdb")
        if not os.path.exists(self.val_data_path + "/4AA-large/val/seq.lmdb"):
            build_lmdb(
                val_npz_paths,
                val_pdb_paths,
                lmdb_path=self.val_data_path + "/seq.lmdb",
            )
        else:
            logging.info(f"LMDB file {self.val_data_path}/seq.lmdb already exists, skipping LMDB creation")

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

        train_metadata = load_lmdb_metadata(self.train_data_path + "/seq.lmdb")
        val_metadata = load_lmdb_metadata(self.val_data_path + "/seq.lmdb")

        train_max_num_particles = train_metadata["max_num_particles"]
        val_max_num_particles = val_metadata["max_num_particles"]

        self.std = torch.tensor(train_metadata["std"])  # train data std used for standardization

        assert train_max_num_particles >= val_max_num_particles, (
            "Train largest system must be greater than or equal to val largest system for pos_embed learning."
        )
        # Need to check here so TarFlow is correctly initalized for data
        assert self.hparams.num_particles > train_max_num_particles, (
            "TarFlow num_particles must be greater than the largest system size. "
            + f"Max num particles={train_max_num_particles}. Set num_particles in data config "
            + f"to {train_max_num_particles + 1}."
        )

        pdb_paths = [*train_metadata["pdb_paths"].values(), *val_metadata["pdb_paths"].values()]

        self.pdb_dict, self.topology_dict = load_pdbs_and_topologies(pdb_paths, self.hparams.num_aa)

        self.encoding_dict = get_encoding_dict(self.topology_dict)

        transform_list = [
            StandardizeTransform(self.std, self.hparams.num_dimensions),
            Random3DRotationTransform(self.hparams.num_dimensions),
        ]
        if self.hparams.com_augmentation:
            # Center of mass augmentation has std 1/sqrt(N) where N is the mean number of atoms
            # in the system. This is the same center of mass std deviation as the prior.
            transform_list.append(
                CenterOfMassTransform(
                    1 / math.sqrt(MEAN_ATOMS_PER_AA * self.hparams.num_aa),
                    self.hparams.num_dimensions,
                )
            )
        if self.hparams.atom_noise_augmentation_factor:
            # Mean min dist is the average min interatomic distance in the training data
            # corresponding to a C-H bond. As this is in unnormalized scale we must
            # divide by the std to get the correct scale for the noise.
            # The atom noise augmentation factor is a scaling factor for the noise
            transform_list.append(
                AtomNoiseTransform(
                    self.hparams.atom_noise_augmentation_factor * train_metadata["mean_min_dist"] / self.std,
                )
            )
        transform_list = transform_list + [
            AddEncodingTransform(self.topology_dict),
            PaddingTransform(self.hparams.num_particles, self.hparams.num_dimensions),
        ]

        transforms = torchvision.transforms.Compose(transform_list)

        self.data_train = PeptideDataset(
            self.train_data_path + "/seq.lmdb",
            num_dimensions=self.hparams.num_dimensions,
            transform=transforms,
        )

        self.data_val = PeptideDataset(
            self.val_data_path + "/seq.lmdb",
            num_dimensions=self.hparams.num_dimensions,
            transform=transforms,
        )

        # TODO prob need a more stable way of doing this - maybe just reading from a list?
        val_sequences = list(val_metadata["num_samples"].keys())
        np.random.seed(0)  # Set a deterministic seed
        np.random.shuffle(val_sequences)
        self.val_sequences = val_sequences[: self.hparams.num_val_sequences]

        self.val_npz_paths = {k: val_metadata["npz_paths"][k] for k in self.val_sequences}
