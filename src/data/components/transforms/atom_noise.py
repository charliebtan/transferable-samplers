from typing import Any

import torch


class AtomNoiseTransform(torch.nn.Module):
    """Adds Gaussian noise to the atom coordinates in the data dictionary."""

    def __init__(self, std: float, num_dimensions: int = 3) -> None:
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
            Dict[str, Any]: The updated data dictionary with added noise to the atom coordinates.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = x + self.std * torch.randn_like(x)

        return {
            **data,
            "x": x,
        }
