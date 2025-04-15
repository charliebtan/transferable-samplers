import torch


class AtomNoiseTransform(torch.nn.Module):
    def __init__(self, std, num_dimensions=3):
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]
        x = x.reshape(-1, self.num_dimensions)
        x = x + self.std * torch.randn_like(x)

        mask = data["mask"]
        x = x * mask[:, None]

        data.update({"x": x})

        return data
