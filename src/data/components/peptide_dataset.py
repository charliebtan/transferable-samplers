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
        self.index = self.metadata["seq_idx"]

        if aa_range is not None:
            self.index = {k: v for k, v in self.index.items() if len(v) in aa_range}
        self.length = sum(self.metadata["num_samples"][k] for k in self.index.keys())

    def worker_init(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset.env = lmdb.open(dataset.lmdb_path, readonly=True, lock=False, readahead=False)

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
            if seq_name not in self.index:
                raise KeyError(f"Sequence name {seq_name} not found in the dataset - has it been filtered out?")
            indexes = self.index[seq_name]
            samples = []
            for idx in indexes:
                key = f"{idx:08}".encode()
                sample = torch.tensor(pickle.loads(txn.get(key))["x"])  # noqa: S301
                samples.append(sample)
        return torch.stack(samples)
