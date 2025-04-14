import torch


class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, data: list[dict[str, torch.Tensor]], transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)
