import torch


class CenterOfMassTransform(torch.nn.Module):
    def __init__(self, num_dimensions, std):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.std = std

    def forward(self, data):
        assert len(data.shape) == 1, "only process single molecules"
        data = data.reshape(-1, self.num_dimensions)
        # Calculate the current center of mass
        center_of_mass = data.mean(dim=0)

        # Generate noise and adjust the center of mass
        noise = torch.randn_like(center_of_mass) * self.std
        new_center_of_mass = center_of_mass + noise

        # Shift all particles so that the center of mass is moved
        data = data + (new_center_of_mass - center_of_mass)

        # Reshape back to original shape
        data = data.reshape(-1)
        return data


if __name__ == "__main__":
    pass
