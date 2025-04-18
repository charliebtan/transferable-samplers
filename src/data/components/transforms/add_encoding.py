from typing import Any

import torch


class AddEncodingTransform(torch.nn.Module):
    """Adds encoding to the data dictionary based on the sequence name."""

    def __init__(self, encoding_dict: dict[str, Any]) -> None:
        """
        Args:
            encoding_dict (Dict[str, Any]): A dictionary mapping sequence names to their respective encodings.
        """
        super().__init__()
        self.encoding_dict = encoding_dict

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing (at least) the key "seq_name".
        Returns:
            Dict[str, Any]: The updated data dictionary with the encoding added.
        """
        assert "mask" not in data, "data should be unpadded (so without a mask)"

        seq_name = data["seq_name"]
        return {
            **data,
            "encoding": self.encoding_dict[seq_name],
        }
