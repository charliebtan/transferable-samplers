import yaml
from pathlib import Path
from tqdm import tqdm
from itertools import product
import torch
from collections import defaultdict
from src.data.components.encoding import ATOM_TYPE_ENCODING_DICT
import logging

ATOM_TYPES_LIST = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'H', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 'HB3', 'HD1', 'HD11', 'HD12', 'HD13', 'HD2', 'HD21', 'HD22', 'HD23', 'HD3', 'HE', 'HE1', 'HE2', 'HE21', 'HE22', 'HE3', 'HG', 'HG1', 'HG11', 'HG12', 'HG13', 'HG2', 'HG21', 'HG22', 'HG23', 'HG3', 'HH', 'HH11', 'HH12', 'HH2', 'HH21', 'HH22', 'HZ', 'HZ1', 'HZ2', 'HZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG']

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
    topology, residue_cache=None, max_atoms_per_residue=82 # TODO load from ATOM_TYPES_LIST
):

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

        # Create a list of indices for the residue's atoms based on the standard atom names
        index_list = [atom_index_by_name.get(name, -1) for name in ATOM_TYPES_LIST]

        # assert len(index_list) <= max_atoms_per_residue
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

    tokenization = torch.stack(residue_tokenization, dim=0)
    n2c_permutation = torch.arange(tokenization.shape[0], dtype=torch.long)
        
    return {
        "tokenization_map": tokenization,
        "permutations": {
            "n2c": n2c_permutation,
            "c2n": n2c_permutation.flip(0),
        }
    }

def get_residue_tokenization_dict(topology_dict):

    residue_tokenization_dict = defaultdict(dict)
    residue_cache = {}  # Cache for residue permutations

    for seq_name, topology in tqdm(topology_dict.items(), desc="Generating residue tokenizations"):
        residue_tokenization_dict[seq_name] = get_residue_tokenization(
            topology,
            residue_cache)

    return residue_tokenization_dict
