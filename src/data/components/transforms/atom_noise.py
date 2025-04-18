import torch


class AtomNoiseTransform(torch.nn.Module):
    def __init__(self, std, num_dimensions=3):
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = x + self.std * torch.randn_like(x)

        return {
            **data,
            "x": x,
        }
