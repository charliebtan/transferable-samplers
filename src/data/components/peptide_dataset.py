import torch
import lmdb
import pickle

class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Open the LMDB environment
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        # Read the length
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f'{idx:08}'.encode()
            sample = pickle.loads(txn.get(key))

        # Convert numpy arrays to tensors if needed
        if isinstance(sample["positions"], np.ndarray):
            sample["positions"] = torch.from_numpy(sample["positions"])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample