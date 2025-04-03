import torch


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        # sample might be a tuple (if TensorDataset has multiple tensors)
        # or a single tensor if there's just one
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)
