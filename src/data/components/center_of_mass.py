import torch


class CenterOfMassTransform(torch.nn.Module):
    def __init__(self, num_dimensions, std):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.std = std

    def forward(self, data):
        x = data["x"]
        mask = data["mask"]

        assert len(x.shape) == 1, "only process single molecules"

        num_particles = mask.sum()

        x = x.reshape(-1, self.num_dimensions)

        x, padding = x[:num_particles], x[num_particles:]  # slice out the data and padding

        assert torch.sum(padding) == 0, "padding should be zero"

        # Calculate the current center of mass
        center_of_mass = x.mean(dim=0)

        # Generate noise and adjust the center of mass
        noise = torch.randn_like(center_of_mass) * self.std
        new_center_of_mass = center_of_mass + noise

        # Shift all particles so that the center of mass is moved
        x = x + (new_center_of_mass - center_of_mass)

        x = torch.cat([x, padding])  # re-add the padding back

        # Reshape back to original shape
        x = x.reshape(-1)

        data.update({"x": x})

        return data


if __name__ == "__main__":
    pass
