from itertools import permutations
from typing import Any

from src.data.components import residue_tokenization
import torch

DONE_SEQUENCES = []
UNPADDED_DICT = {}
PADDED_DICT = {}
COLLECT_SEQUENCES = False

class PaddingTransform(torch.nn.Module):
    """Pads the input data to a fixed size and creates a mask for the padded elements."""

    def __init__(self, max_num_particles: int, num_dimensions: int, max_num_residues: int) -> None:
        """
        Args:
            max_num_particles (int): Max number of particles to pad to.
            num_dimensions (int): Number of dimensions for the atom coordinates. Default is 3.
        """
        super().__init__()
        self.max_num_particles = max_num_particles
        self.num_dimensions = num_dimensions
        self.max_num_residues = max_num_residues

    def pad_data(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        num_particles = x.shape[0]
        pad_tensor = torch.zeros(self.max_num_particles - num_particles, self.num_dimensions)
        return torch.cat([x, pad_tensor])

    def pad_encoding(self, encoding: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        padded_encoding = {}
        for key, value in encoding.items():
            if not key == "seq_len":  # don't pad seq_len - is single value per sample
                padded_encoding[key] = torch.cat(
                    [value, torch.zeros(self.max_num_particles - value.shape[0], dtype=torch.int64)]
                )
            else:
                padded_encoding[key] = value
        return padded_encoding

    def create_permutation_mask(self, permutations: dict[str, torch.Tensor], padded_seq_len: int) -> dict[str, torch.Tensor]:
        num_tokens = next(iter(permutations.values())).shape[0]
        assert all(len(v) == num_tokens for v in permutations.values()), "All permutations must have same length"
        true_mask = torch.ones(num_tokens, dtype=torch.bool)
        false_mask = torch.zeros(padded_seq_len - num_tokens, dtype=torch.bool)
        return torch.cat([true_mask, false_mask])

    def pad_permutations(self, permutations: dict[str, torch.Tensor], padded_seq_len: int) -> dict[str, torch.Tensor]:
        num_tokens = next(iter(permutations.values())).shape[0]
        assert all(len(v) == num_tokens for v in permutations.values()), "All permutations must have same length"
        pad_len = padded_seq_len - num_tokens
        if not pad_len:
            return permutations.copy()
        else:
            padded_permutations = {}
            for key, value in permutations.items():
                pad_start = torch.max(value).item() + 1
                pad_values = torch.arange(pad_start, pad_start + pad_len, dtype=torch.int64)
                padded_permutations[key] = torch.cat([value, pad_values])
            return padded_permutations

    def pad_tokenization_map(self, tokenization_map: torch.Tensor, padded_seq_len: int) -> dict[str, torch.Tensor]:
        num_tokens = tokenization_map.shape[0]
        assert not num_tokens > padded_seq_len, "tokenization map has more tokens than padded sequence length"
        pad_len = padded_seq_len - num_tokens
        if not pad_len:
            return tokenization_map.clone()
        else:
            single_pad_tensor = torch.ones_like(tokenization_map[0:1]) * -1  # padding with -1
            pad_tensor = single_pad_tensor.repeat(pad_len, 1)
            return torch.cat([tokenization_map, pad_tensor])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Args:
            data (Dict[str, Any]): The input data dictionary containing the keys "x", "encoding" and possibly "permutations".
        Returns:
            Dict[str, Any]: The updated data dictionary with padded data, encoding, a mask, and possibly padded permutations.
        """
        assert "mask" not in data, "data already has a mask, cannot pad again"

        if not data["seq_name"] in DONE_SEQUENCES and COLLECT_SEQUENCES:
            print(len(DONE_SEQUENCES), "sequences done, processing", data["seq_name"])
            TODO = True
            DONE_SEQUENCES.append(data["seq_name"])
            UNPADDED_DICT[data["seq_name"]] = data
        else:
            TODO = False
            return None

        x = data["x"]
        encoding = data["encoding"]
        permutations = data["permutations"]

        assert len(x.shape) == 2, f"only process single molecules, got shape of {x.shape}"
        assert x.shape[1] == self.num_dimensions, f"expected {self.num_dimensions} dimensions, got {x.shape[1]}"

        x = self.pad_data(x)

        atom_mask = self.create_permutation_mask(permutations["atom"]["permutations"], padded_seq_len=self.max_num_particles)
        residue_mask = self.create_permutation_mask(permutations["residue"]["permutations"], padded_seq_len=self.max_num_residues)

        encoding = self.pad_encoding(encoding)

        atom_permutations = self.pad_permutations(permutations["atom"]["permutations"], padded_seq_len=self.max_num_particles)
        residue_permutations = self.pad_permutations(permutations["residue"]["permutations"], padded_seq_len=self.max_num_residues)
        residue_tokenization_map = self.pad_tokenization_map(permutations["residue"]["tokenization_map"], padded_seq_len=self.max_num_residues)

        padded_batch =  {
            **data,
            "x": x,
            "encoding": encoding,
            "permutations": {
                "atom": {"permutations": atom_permutations, 
                         "mask": atom_mask},
                "residue": {"permutations": residue_permutations,
                            "mask": residue_mask,
                            "tokenization_map": residue_tokenization_map
                            },
            },
        }

        if TODO:
            PADDED_DICT[data["seq_name"]] = padded_batch

        if len(DONE_SEQUENCES) == 16800 and COLLECT_SEQUENCES
            import pickle
            with open("src/data/components/transforms/padded_dict.pkl", "wb") as f:
                pickle.dump(PADDED_DICT, f)
            with open("src/data/components/transforms/unpadded_dict.pkl", "wb") as f:
                pickle.dump(UNPADDED_DICT, f)

        return padded_batch
