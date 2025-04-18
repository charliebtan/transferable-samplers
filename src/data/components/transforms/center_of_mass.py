import torch


class CenterOfMassTransform(torch.nn.Module):
    def __init__(self, std, num_dimensions):
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data):
        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        # Generate noise and adjust the center of mass
        noise = torch.randn_like(x[0]) * self.std

        # Shift all particles so that the center of mass is moved
        x = x + noise

        return {
            **data,
            "x": x,
        }


if __name__ == "__main__":
    pass
