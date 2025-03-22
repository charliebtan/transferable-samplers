from dataclasses import dataclass

import torch

@dataclass
class SamplesData:
    samples: torch.Tensor
    energy: torch.Tensor

    def __post_init__(self):
        assert len(self.samples) == len(self.energy)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return SamplesData(self.samples[index], self.energy[index])