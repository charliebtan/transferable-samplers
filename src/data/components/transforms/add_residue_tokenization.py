from typing import Any

import torch


class AddResidueTokenizationTransform(torch.nn.Module):
    """Adds residue_tokenization to the data dictionary based on the sequence name."""

    def __init__(self, residue_tokenization_dict: dict[str, Any]) -> None:
        """
        Args:
            residue_tokenization_dict (Dict[str, Any]): A dictionary mapping sequence names to their respective residue_tokenization.
        """
        super().__init__()
        self.residue_tokenization_dict = residue_tokenization_dict

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "seq_name".
        Returns:
            Dict[str, Any]: The updated data dictionary with the residue_tokenization added.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        seq_name = data["seq_name"]
        return {
            **data,
            "residue_tokenization": self.residue_tokenization_dict[seq_name],
        }
