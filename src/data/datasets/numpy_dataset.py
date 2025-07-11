import numpy as np
import torch


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, npy_array, num_dimensions: int, transform=None):
        self.npy_array = npy_array
        self.num_dimensions = num_dimensions
        self.transform = transform

    def __len__(self):
        return len(self.npy_array)

    def __getitem__(self, idx):
        x = self.npy_array[idx]
        if self.transform is not None:
            x = torch.tensor(x).float()
            x = x.view(-1, self.num_dimensions)
            sample = self.transform(
                {
                    "x": x,
                }
            )
            sample["x"] = sample["x"].reshape(-1)
        return sample
