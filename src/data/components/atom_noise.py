import torch


class AtomNoiseTransform(torch.nn.Module):
    def __init__(self, std, num_dimensions=3):
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]
        assert len(x.shape) == 1, "only process single molecules"

        x = x.reshape(-1, self.num_dimensions)
        x = x + self.std * torch.randn_like(x)

        mask = data.get("mask", None)
        if mask is not None:
            x = x * mask[:, None]

        data.update({"x": x})

        return data
