from typing import Any

import torch


class CenterOfMassTransform(torch.nn.Module):
    """Applies Gaussian noise to the center of mass of the molecule."""

    def __init__(self, std: float, num_dimensions: int) -> None:
        """
        Args:
            std (float): Standard deviation of the Gaussian noise to be added.
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "x".
        Returns:
            Dict[str, Any]: The updated data dictionary with added noise to the center of mass.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

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
