import os

import numpy as np
import torch
from bgmol.datasets import AImplicitUnconstrained

from src.data.components.peptide_dataset import PeptideDataset
from src.data.peptide_datamodule import PeptideDataModule


class ALDPDataModule(PeptideDataModule):
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        super().prepare_data()

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        # yes a hack but only way without changing bgmol
        self.bgmol_dataset = AImplicitUnconstrained(
            read=True,
            download=True if "AImplicitUnconstrained" not in os.listdir() else False,
        )

    def setup_potential(self):
        self.topology = self.bgmol_dataset.system.mdtraj_topology
        self.potential = self.bgmol_dataset.get_energy_model()

    def setup_atom_encoding(self):
        self.encodings = {
            "atom_type": torch.zeros((0,)),
            "aa_pos": torch.zeros((0,)),
            "aa_type": torch.zeros((0,)),
        }

    def setup_data(self):
        # Load the data - data is 10 times smaller in bgflow dataset than in npy
        train_data = np.load(self.data_path, allow_pickle=True) / 10.0
        test_data = torch.tensor(self.bgmol_dataset.xyz).view(-1, self.hparams.dim)

        # Reshape and tensorize
        train_data = train_data.reshape(-1, self.hparams.dim)
        train_data = torch.tensor(train_data).float()
        test_data = test_data.reshape(-1, self.hparams.dim)
        test_data = torch.tensor(test_data).float()

        if self.hparams.make_iid:
            raise NotImplementedError("IID not implemented for this dataset")

        # Zero center of mass
        train_data = self.zero_center_of_mass(train_data)
        test_data = self.zero_center_of_mass(test_data)

        # Compute std on only train data
        self.std = train_data.std()

        # Standardize the data
        train_data = self.normalize(train_data)
        test_data = self.normalize(test_data)

        # Create training dataset with transforms applied
        self.data_train = PeptideDataset(train_data, transform=self.transforms)

        # Split val and test data
        self.data_val, self.data_test = (
            test_data[: self.hparams.num_val_samples],
            test_data[self.hparams.num_val_samples :],
        )

        # Randomized ordering of val samples
        val_rng = np.random.default_rng(0)
        self.data_val = torch.tensor(val_rng.permutation(self.data_val))

        # Randomized ordering / subset of test samples
        test_rng = np.random.default_rng(1)
        self.data_test = torch.tensor(test_rng.permutation(self.data_test))[: self.hparams.num_test_samples]

        # Create training dataset with transforms applied
        self.data_train = PeptideDataset(train_data, transform=self.transforms, encodings=self.encodings)

        # I actually thought better to apply transforms to val and test data too
        self.data_val = PeptideDataset(self.data_val, transform=self.transforms, encodings=self.encodings)
        self.data_test = PeptideDataset(self.data_test, transform=self.transforms, encodings=self.encodings)


if __name__ == "__main__":
    _ = ALDPDataModule()
