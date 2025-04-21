import pickle

import lmdb
import torch

from src.data.components.prepare_data import load_lmdb_metadata


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, num_dimensions: int, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.num_dimensions = num_dimensions

        self.index = load_lmdb_metadata(lmdb_path)["index"]

        # Open the LMDB environment
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Read the length
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))  # noqa: S301

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:08}".encode()
            sample = pickle.loads(txn.get(key))  # noqa: S301

        if self.transform is not None:
            x = torch.tensor(sample["x"]).float()
            x = x.view(-1, self.num_dimensions)
            sample = self.transform(
                {
                    **sample,
                    "x": x,
                }
            )
            sample["x"] = sample["x"].view(-1)
        return sample

    def get_seq_data(self, seq_name: str):
        with self.env.begin() as txn:
            indexes = self.index[seq_name]
            samples = []
            for idx in indexes:
                key = f"{idx:08}".encode()
                sample = torch.tensor(pickle.loads(txn.get(key))["x"])  # noqa: S301
                samples.append(sample)
        return torch.stack(samples)
