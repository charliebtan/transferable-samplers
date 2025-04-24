from typing import Any

import torch
from scipy.spatial.transform import Rotation as R


class Random3DRotationTransform(torch.nn.Module):
    """Applies a random 3D rotation to the input data coordinates."""

    def __init__(self, num_dimensions: int) -> None:
        """
        Args:
            num_dimensions (int): Number of dimensions for the atom coordinates. Must be 3.
        """
        super().__init__()

        if num_dimensions != 3:
            raise ValueError("Random3DRotationTransform only supports 3D rotations.")

        self.num_dimensions = num_dimensions

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data: The input data dictionary containing (at least) the key "x".
        Returns:
            data: The updated data dictionary with the rotated coordinates.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        x: torch.Tensor = data["x"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = x.unsqueeze(0)
        rot = torch.tensor(R.random(len(x)).as_matrix()).to(x)
        x = torch.einsum("bij,bki->bkj", rot, x)
        x = x.squeeze(0)

        return {
            **data,
            "x": x,
        }
