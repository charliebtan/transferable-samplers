import torch


class StandardizeTransform(torch.nn.Module):
    def __init__(self, std, num_dimensions):
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        # Calculate the current center of mass
        center_of_mass = x.mean(dim=0)
        x = (x - center_of_mass) / self.std

        return {
            **data,
            "x": x,
        }
