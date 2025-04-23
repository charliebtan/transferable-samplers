import pickle

import lmdb
import torch

from src.data.components.prepare_data import load_lmdb_metadata


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, num_dimensions: int, aa_range: list[int] = None, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.num_dimensions = num_dimensions

        self.metadata = load_lmdb_metadata(lmdb_path)
        self.seq_to_idx = self.metadata["seq_to_idx"]

        if aa_range is not None:
            self.seq_to_idx = {k: v for k, v in self.seq_to_idx.items() if len(v) in aa_range}
        self.length = sum(self.metadata["num_samples"][k] for k in self.seq_to_idx.keys())

        self.env = None

    def __len__(self):
        return self.length

    def _load_lmdb_entry(self, txn, idx):
        key = f"{idx:08}".encode()
        return pickle.loads(txn.get(key))  # noqa: S301

    def __getitem__(self, idx):
        if self.env is None:
            # Lazily open the LMDB environment when the first item is accessed
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)

        with self.env.begin() as txn:
            sample = self._load_lmdb_entry(txn, idx)

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
        if self.env is None:
            # Lazily open the LMDB environment when the first item is accessed
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)

        with self.env.begin() as txn:
            if seq_name not in self.seq_to_idx:
                raise KeyError(f"Sequence name {seq_name} not found in the dataset - has it been filtered out?")
            indexes = self.seq_to_idx[seq_name]
            xs = []
            for idx in indexes:
                x = self._load_lmdb_entry(txn, idx)["x"]
                xs.append(torch.tensor(x))
        return torch.stack(xs)
