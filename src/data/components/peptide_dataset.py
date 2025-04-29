import pickle

import lmdb
import torch

from src.data.components.prepare_data import load_lmdb_metadata


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, seq_names: list[str], num_dimensions: int, transform=None):
        self.lmdb_path = lmdb_path
        self.seq_names = seq_names
        self.transform = transform
        self.num_dimensions = num_dimensions

        self.metadata = load_lmdb_metadata(lmdb_path)

        # Filter the metadata based on the provided seq_names
        for key in self.metadata.keys():
            if isinstance(self.metadata[key], dict):
                self.metadata[key] = {k: v for k, v in self.metadata[key].items() if k in seq_names}

        # The following will now only include the sequences in seq_names
        self.seq_to_idx = self.metadata["seq_to_idx"]
        self.valid_indices = []
        for seq_name in self.seq_to_idx:
            self.valid_indices.extend(self.seq_to_idx[seq_name])

        self.length = len(self.valid_indices)

        self.env = None  # LMDB environment is lazily loaded due to multiple processes accessing itj

    def __len__(self):
        return self.length

    def _load_lmdb_entry(self, txn, idx):
        key = f"{idx:08}".encode()
        return pickle.loads(txn.get(key))  # noqa: S301

    def __getitem__(self, idx):
        lmdb_idx = self.valid_indices[idx]  # transform the index to the global index

        if self.env is None:
            # Lazily open the LMDB environment when the first item is accessed
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)

        with self.env.begin() as txn:
            sample = self._load_lmdb_entry(txn, lmdb_idx)

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


class PeptideDatasetFromBuffer(torch.utils.data.Dataset):
    def __init__(self, buffer, num_dimensions: int, transform=None):
        self.buffer = buffer
        self.transform = transform
        self.num_dimensions = num_dimensions

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        # Sample the length of the idx
        # since the input idx will not correspond to
        # the internal idx
        batch_size = 1
        if not isinstance(idx, int):
            batch_size = len(idx)

        x, _ = self.buffer.sample(batch_size)
        sample = {"x": x}
        if self.transform is not None:
            x = x.view(-1, self.num_dimensions)
            sample = self.transform(
                {
                    **sample,
                    "x": x,
                }
            )
            sample["x"] = sample["x"].view(-1)
        return sample

    def add(self, x):
        self.buffer.add(x)
