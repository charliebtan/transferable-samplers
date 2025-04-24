from typing import Any

import torch


class StandardizeTransform(torch.nn.Module):
    """Zero center of mass and normalize the coordinates of the molecule."""

    def __init__(self, std: float, num_dimensions: int) -> None:
        """
        Args:
            std (float): Standard deviation for normalization.
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()
        self.std = std
        self.num_dimensions = num_dimensions

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data: The input data dictionary containing (at least) the key "x".
        Returns:
            data: The updated data dictionary with standardized coordinates."
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

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
