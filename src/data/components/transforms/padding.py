from typing import Any

import torch


class PaddingTransform(torch.nn.Module):
    """Pads the input data to a fixed size and creates a mask for the padded elements."""

    def __init__(self, max_num_particles: int, num_dimensions: int) -> None:
        """
        Args:
            max_num_particles (int): Max number of particles to pad to.
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()
        self.max_num_particles = max_num_particles
        self.num_dimensions = num_dimensions

    def pad_data(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        num_particles = x.shape[0]
        pad_tensor = torch.zeros(self.max_num_particles - num_particles, self.num_dimensions)
        return torch.cat([x, pad_tensor])

    def pad_encoding(self, encoding: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in encoding.items():
            if not key == "seq_len":  # don't pad seq_len - is single value per sample
                encoding[key] = torch.cat(
                    [value, torch.zeros(self.max_num_particles - value.shape[0], dtype=torch.int64)]
                )
        return encoding

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        num_particles = x.shape[0]
        true_mask = torch.ones(num_particles)
        false_mask = torch.zeros(self.max_num_particles - num_particles)
        return torch.cat([true_mask, false_mask]).bool()

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing the keys "x" and "encoding".
        Returns:
            Dict[str, Any]: The updated data dictionary with padded data and mask.
        """
        assert "mask" not in data, "data already has a mask, cannot pad again"

        x = data["x"]
        encoding = data["encoding"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        mask = self.create_mask(x)  # must make mask before padding!

        x = self.pad_data(x)
        encoding = self.pad_encoding(encoding)

        return {
            **data,
            "x": x,
            "encoding": encoding,
            "mask": mask,
        }
