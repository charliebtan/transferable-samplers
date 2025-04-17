import copy
import logging
import math
import os
from typing import Any, Optional

import huggingface_hub
import torch
import torchvision

from src.data.components.atom_noise import AtomNoiseTransform
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.rotation import Random3DRotationTransform
from src.data.transferable_peptide_datamodule import TransferablePeptideDataModule
from src.evaluation.plots.plot_atom_distances import interatomic_dist

MEAN_MIN_DIST_DICT = {
    2: 0.4658  # can be saved in dict after computing in setup_data
}
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

        # Download the training data
        huggingface_hub.snapshot_download(
            repo_id=self.hparams.huggingface_repo_id,
            repo_type="dataset",
            allow_patterns=f"{self.hparams.huggingface_train_data_dir}/*",
            local_dir=self.train_data_path,
        )

        # Download the validation data
        huggingface_hub.snapshot_download(
            repo_id=self.hparams.huggingface_repo_id,
            repo_type="dataset",
            allow_patterns=f"{self.hparams.huggingface_val_data_dir}/*",
            local_dir=self.val_data_path,
        )

    def setup_data(self):
        train_data_dict, self.max_num_particles = self.load_data_as_tensor_dict(self.train_data_path)
        val_data_dict, _ = self.load_data_as_tensor_dict(self.val_data_path)

        new_val_data_dict = {}
        for key in val_data_dict.keys():
            if key in self.train_sequences:
                logging.info(f"Key {key} found in train_sequences, removing from val_sequences")
            else:
                assert key in self.val_sequences, f"Key {key} not found in val_sequences"
                new_val_data_dict[key] = val_data_dict[key]
        val_data_dict = new_val_data_dict

        for key in train_data_dict.keys():
            assert key in self.train_sequences, f"Key {key} not found in train_sequences"
            assert key not in self.val_sequences, f"Key {key} found in val_sequences"

        for key in self.val_sequences:
            assert key in val_data_dict, f"Key {key} not found in val_data_dict"
            assert key not in train_data_dict, f"Key {key} found in train_data_dict"

        for key in self.train_sequences:
            assert key in train_data_dict, f"Key {key} not found in train_data_dict"
            assert key not in val_data_dict, f"Key {key} found in val_data_dict"

        common_keys = set(train_data_dict.keys()).intersection(set(val_data_dict.keys()))
        logging.info(f"Common keys between train and val data dict: {common_keys}")

        # Need to check here so TarFlow is correctly initalized for data
        assert self.hparams.num_particles > self.max_num_particles, (
            "TarFlow num_particles must be greater than the largest system size. "
            + f"Max num particles={self.max_num_particles}. Set num_particles in data config "
            + f"to {self.max_num_particles + 1}."
        )

        # TarFlow dim is the dim of the largest system + a single padded token
        assert self.hparams.dim > self.max_num_particles * self.hparams.num_dimensions, (
            "TarFlow dim must be greater than the largest system size + a single padded token. "
            + f"Set dim in data config to {(self.max_num_particles + 1) * self.hparams.num_dimensions}."
        )

        weighted_vars = [x.var(unbiased=False) * x.shape[0] for x in train_data_dict.values()]
        total_samples = sum(x.shape[0] for x in train_data_dict.values())
        self.std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_samples)

        train_data_dict = self.normalize_tensor_dict(train_data_dict)
        val_data_dict = self.normalize_tensor_dict(val_data_dict)

        # Setup transforms
        transform_list = [Random3DRotationTransform(self.hparams.num_dimensions)]
        if self.hparams.com_augmentation:
            transform_list.append(
                CenterOfMassTransform(
                    self.hparams.num_dimensions,
                    1 / math.sqrt(MEAN_ATOMS_PER_AA * self.hparams.num_aa),
                )
            )
        if self.hparams.atom_noise_augmentation_factor:
            if self.hparams.num_aa not in MEAN_MIN_DIST_DICT:
                mean_min_dists = []
                for key, data in train_data_dict.items():
                    num_samples = data.shape[0]
                    data = data.reshape(num_samples, -1, self.hparams.num_dimensions)
                    dists = interatomic_dist(data, flatten=False)
                    mean_min_dist = dists.min(dim=1)[0].mean()
                    mean_min_dists.append(mean_min_dist)

                # this is effectively just the length of a carbon-hydrogen bond
                mean_min_dist = torch.mean(torch.tensor(mean_min_dists))
                logging.warning(
                    f"MEAN_MIN_DIST={mean_min_dist.item()} for peptide of length {self.hparams.num_aa}. "
                    "Save it in MEAN_MIN_DIST_DICT to save on computation"
                )
            else:
                mean_min_dist = MEAN_MIN_DIST_DICT[self.hparams.num_aa]

            transform_list.append(
                AtomNoiseTransform(
                    self.hparams.atom_noise_augmentation_factor * mean_min_dist,
                )
            )
        transforms = torchvision.transforms.Compose(transform_list)

        # slice out subset of val_data_dict
        val_data_dict = {
            key: val for i, (key, val) in enumerate(val_data_dict.items()) if i < self.hparams.num_val_sequences
        }
        self.val_sequences = list(val_data_dict.keys())

        train_data_dict = self.add_encodings_to_tensor_dict(train_data_dict)
        val_data_dict = self.add_encodings_to_tensor_dict(val_data_dict)

        self.val_data_dict = copy.deepcopy(val_data_dict)  # without padding for sampling eval

        train_data_dict = self.pad_and_mask_tensor_dict(train_data_dict)
        val_data_dict = self.pad_and_mask_tensor_dict(val_data_dict)  # with padding for loss eval

        train_data_list = self.tensor_dict_to_samples_list(train_data_dict)
        val_data_list = self.tensor_dict_to_samples_list(val_data_dict)

        self.data_train = PeptideDataset(train_data_list, transform=transforms)
        self.data_val = PeptideDataset(val_data_list, transform=transforms)
