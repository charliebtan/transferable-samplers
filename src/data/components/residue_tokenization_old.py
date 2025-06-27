import yaml
from pathlib import Path
from tqdm import tqdm
from itertools import product
import torch
from collections import defaultdict
import logging

# Load a YAML file and return its contents as a dictionary
def load_yaml_as_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

def standardize_atom_name(atom_name: str, aa_name: str) -> str:
    if atom_name[0] == "H" and atom_name[-1] in ("1", "2", "3"):
        # For these AA the H-X-N atoms are not interchangable
        if aa_name in ("HIS", "HIE", "PHE", "TRP", "TYR") and atom_name[:2] in (
            "HE",
            "HD",
            "HZ",
            "HH",
        ):
            pass
        else:
            atom_name = atom_name[:-1]

    # Standarize side-chain O atom encoding
    if atom_name[:2] == "OE" or atom_name[:2] == "OD":
        atom_name = atom_name[:-1]

    return atom_name

def get_residue_tokenization(
    permutations_definition_dict, topology, residue_cache=None, max_atoms_per_residue=32
):

    backbone_order = permutations_definition_dict["backbone"]["n2c"]

    residue_list = list(topology.residues)
    residue_tokenization = []

    for i, residue in enumerate(residue_list):
        residue_name = residue.name if residue.name != "HIS" else "HIE"

        if i == 0:
            residue_name_with_terminals = "N-" + residue_name
        elif i == len(residue_list) - 1:
            residue_name_with_terminals = "C-" + residue_name
        else:
            residue_name_with_terminals = residue_name

        assert residue_name in permutations_definition_dict["sidechain"], \
            f"Residue {residue_name} not in sidechain definitions"

        atoms = list(residue.atoms)
        input_atom_ordering = [standardize_atom_name(atom.name, residue_name) for atom in atoms]
        first_atom_index = list(residue.atoms)[0].index
        if residue_cache is not None:
            cached = residue_cache.get(residue_name_with_terminals)
            if cached is not None:
                if input_atom_ordering != cached["input_atom_ordering"]:
                    raise AssertionError(
                        f"Inconsistent atom name order for {residue_name_with_terminals}.\n"
                        f"Expected: {cached['input_atom_ordering']}\nFound:    {input_atom_ordering}"
                    )

                cached_indices = cached["residue_atom_indices"]
                adjusted_indices = torch.where(
                    cached_indices != -1,
                    cached_indices + first_atom_index,
                    cached_indices  # leave -1s unchanged
                )
                residue_tokenization.append(adjusted_indices)
                continue

        # Build name → atom index map once
        atom_index_by_name = {atom.name: atom.index for atom in residue.atoms}

        # === Canonical ordering ===

        # 1. Backbone atoms always in fixed order (including optional terminal atoms)
        backbone_indices = []
        for name in backbone_order:
            if residue_name == "GLY" and name in ["HA", "HA2", "HA3"]:
                if name == "HA2":
                    # For glycine residue token, we only consider HA2 as part of the backbone
                    backbone_indices.append(atom_index_by_name["HA2"])
                else:
                    pass
            elif name in ["HA2", "HA3"]:
                pass
            else:
                backbone_indices.append(atom_index_by_name.get(name, -1))

        assert len(backbone_indices) == 9 # some jazzy logic so just making sure the correct number of backbone atoms are present
            
        if residue_name == "GLY":
            # Add the second HA for glycine residues
            sidechain_indices = [atom_index_by_name["HA3"]]
        else:
            # 2. Sidechain atoms (from permutation definition)
            sidechain_atoms = permutations_definition_dict["sidechain"][residue_name]["standard"]
            sidechain_indices = [atom_index_by_name[name] for name in sidechain_atoms] # do not allow sidechain atoms to be missing from atom_index_by_name

        # 3. Combine and pad
        index_list = backbone_indices + sidechain_indices
        assert len(index_list) <= max_atoms_per_residue
        index_list += [-1] * (max_atoms_per_residue - len(index_list))

        index_list = torch.tensor(index_list, dtype=torch.long)

        # Store
        residue_tokenization.append(index_list)

        if residue_cache is not None:
            adjusted_indices = torch.where(
                    index_list != -1,
                    index_list - first_atom_index,
                    index_list  # leave -1s unchanged
                )
            residue_cache[residue_name_with_terminals] = {
                "input_atom_ordering": input_atom_ordering,
                "residue_atom_indices": adjusted_indices,
            }
    used_atom_indices = torch.cat([r[r != -1] for r in residue_tokenization])

    # Final integrity checks
    assert len(used_atom_indices) == topology.n_atoms, f"used_atom_indices length (must be {topology.n_atoms}, got {len(used_atom_indices)})"
    assert max(used_atom_indices) == topology.n_atoms - 1, f"used_atom_indices max index (must be {topology.n_atoms - 1}, got {max(used_atom_indices)})"
    assert min(used_atom_indices) == 0, f"used_atom_indices min index (must be 0, got {min(used_atom_indices)})"
    unique, counts = used_atom_indices.unique(return_counts=True)
    duplicates = unique[counts > 1]
    assert len(duplicates) == 0, f"used_atom_indices contains duplicate atom indices: {duplicates.tolist()}"
        
    return torch.stack(residue_tokenization, dim=0)


def get_residue_tokenization_dict(topology_dict):

    # Load the permutations definition from YAML file
    permutations_definition_dict = load_yaml_as_dict("src/data/components/permutations.yaml")

    residue_tokenization_dict = defaultdict(dict)
    residue_cache = {}  # Cache for residue permutations

    for seq_name, topology in tqdm(topology_dict.items(), desc="Generating residue tokenizations"):
        # Initialize the dictionary for this sequence
        residue_tokenization_dict[seq_name] = get_residue_tokenization(
            permutations_definition_dict,
            topology,
            residue_cache,
        )

    return residue_tokenization_dict
