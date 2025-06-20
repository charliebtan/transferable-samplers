import yaml
from pathlib import Path
from tqdm import tqdm
from itertools import product
import torch
from collections import defaultdict


# Load a YAML file and return its contents as a dictionary
def load_yaml_as_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

# Generate a specific atom index permutation based on multiple ordering strategies
def get_permutation(permutations_definition_dict, topology, sequence_ordering, global_type, sidechain_variant, heavy_type, residue_cache=None):

    # Validate input strategy options
    if sequence_ordering not in ["n2c", "c2n"]:
        raise ValueError(f"Unknown sequence ordering: {sequence_ordering}")
    if global_type not in ["residue-by-residue", "backbone-first"]:
        raise ValueError(f"Unknown global type: {global_type}")
    if heavy_type not in ["group-by-group", "heavy-first"]:
        raise ValueError(f"Unknown heavy type: {heavy_type}")
    if sidechain_variant not in ["standard", "variant"]:
        raise ValueError(f"Unknown sidechain variant: {sidechain_variant}")

    # Lists to hold atom index permutations for backbone and sidechain atoms
    backbone_permutation = []
    sidechain_permutation = []

    # Get residues in forward or reverse order
    residue_list = list(topology.residues)
    if sequence_ordering != "n2c":
        residue_list = list(reversed(residue_list))

    # Backbone atom names for this ordering (e.g., N2C or C2N)
    backbone_permutation_definition = permutations_definition_dict["backbone"][sequence_ordering]

    # Iterate through each residue to build permutations
    for i, residue in enumerate(residue_list):

        residue_name = residue.name if residue.name != "HIS" else "HIE"  # timewarp data has HIS labelled as HIE
        residue_atoms = list(residue.atoms)

        # Determine the residue name with terminal modifications based on its position in the sequence
        if i == 0 and sequence_ordering == "n2c" or i == len(residue_list) - 1 and sequence_ordering == "c2n":
            residue_name_with_terminals = "N-" + residue_name
        elif i == 0 and sequence_ordering == "c2n" or  i == len(residue_list) - 1 and sequence_ordering == "n2c":
            residue_name_with_terminals = "C-" + residue_name
        else:
            residue_name_with_terminals = residue_name

        if residue_cache is not None and residue_cache.get(residue_name_with_terminals) is not None:
            # If residue is cached, use the cached backbone and sidechain permutations
            residue_backbone_permutation = residue_cache[residue_name_with_terminals]["backbone"] + residue_atoms[0].index
            residue_sidechain_permutation = residue_cache[residue_name_with_terminals]["sidechain"] + residue_atoms[0].index

            assert len(residue_backbone_permutation) + len(residue_sidechain_permutation) == len(residue_atoms), \
                f"Cached permutation for {residue_name_with_terminals} has wrong length: {len(residue_backbone_permutation) + len(residue_sidechain_permutation)} != {len(residue_atoms)}" \

        else:

            residue_backbone_permutation = []
            residue_sidechain_permutation = []

            # Get permutation rules for the residue's sidechain
            sidechain_permutations_definition_dict = permutations_definition_dict["sidechain"][residue_name]

            # Choose the appropriate variant strategy for sidechain atom ordering
            if sidechain_variant == "variant":
                if "ring_reverse" in sidechain_permutations_definition_dict:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["ring_reverse"]
                elif "branch_order_reverse" in sidechain_permutations_definition_dict:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["branch_order_reverse"]
                else:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["standard"]
            else:
                sidechain_permutation_definition = sidechain_permutations_definition_dict["standard"]

            # Check for overlap between backbone and sidechain definitions
            overlap = set(backbone_permutation_definition) & set(sidechain_permutation_definition)
            if overlap:
                raise ValueError(f"Atom(s) {overlap} defined in both backbone and sidechain for residue {residue_name}")

            # Assign each atom to backbone or sidechain permutation based on its name
            for atom in residue.atoms:
                if atom.name in backbone_permutation_definition:
                    residue_backbone_permutation.append(atom.index)
                elif atom.name in sidechain_permutation_definition:
                    residue_sidechain_permutation.append(atom.index)
                else:
                    raise ValueError(f"Atom {atom.name} not found in any permutation definition for residue {residue_name}")

            # If required, sort sidechain atoms with heavy atoms first, then hydrogens
            if heavy_type == "heavy-first":
                residue_sidechain_permutation = [
                    idx for idx in residue_sidechain_permutation if topology.atom(idx).element.symbol != "H"
                ] + [
                    idx for idx in residue_sidechain_permutation if topology.atom(idx).element.symbol == "H"
                ]

            residue_backbone_permutation = torch.tensor(residue_backbone_permutation, dtype=int)
            residue_sidechain_permutation = torch.tensor(residue_sidechain_permutation, dtype=int)

            if residue_cache is not None:
                # Cache the backbone and sidechain permutations for this residue
                residue_cache[residue_name_with_terminals] = {
                    "backbone": residue_backbone_permutation - residue_atoms[0].index, # Normalize indices to start from 0
                    "sidechain": residue_sidechain_permutation - residue_atoms[0].index, # Normalize indices to start from 0
                }

        # Save the permutations per residue
        backbone_permutation.append(residue_backbone_permutation)
        sidechain_permutation.append(residue_sidechain_permutation)

    # Flatten the residue-wise permutations into a full permutation
    permutation = []
    if global_type == "residue-by-residue":
        for i in range(topology.n_residues): # Loop through residues once
            permutation.append(backbone_permutation[i])
            permutation.append(sidechain_permutation[i])
    elif global_type == "backbone-first":
        for i in range(topology.n_residues): # Loop through residues once - taking backbone atoms
            permutation.append(backbone_permutation[i])
        for i in range(topology.n_residues): # Loop through residues again - taking sidechain atoms
            permutation.append(sidechain_permutation[i])
    permutation = torch.cat(permutation)

    # Final integrity checks
    assert len(permutation) == topology.n_atoms, f"Permutation length (must be {topology.n_atoms}, got {len(permutation)})"
    assert max(permutation) == topology.n_atoms - 1, f"Permutation max index (must be {topology.n_atoms - 1}, got {max(permutation)})"
    assert min(permutation) == 0, f"Permutation min index (must be 0, got {min(permutation)})"
    unique, counts = permutation.unique(return_counts=True)
    duplicates = unique[counts > 1]
    assert len(duplicates) == 0, f"Permutation contains duplicate atom indices: {duplicates.tolist()}"

    return permutation

def get_permutations_dict(topology_dict):

    # Load the permutations definition from YAML file
    permutations_definition_dict = load_yaml_as_dict("src/data/components/permutations.yaml")

    permutations_dict = defaultdict(dict)

    sequence_orderings = ["n2c"] #, "c2n"]
    global_types = ["residue-by-residue"] # , "backbone-first"]
    sidechain_variants = ["standard"] # , "variant"]
    heavy_types = ["group-by-group"] # , "heavy-first"]

    # Generate all combinations of configuration settings
    configs = list(product(sequence_orderings, global_types, sidechain_variants, heavy_types))
    total = len(configs) * len(topology_dict) # total number of permutations to generate

    with tqdm(total=total, desc="Generating permutations") as pbar: # progress bar for tracking progress
        for sequence_ordering, global_type, sidechain_variant, heavy_type in configs:
            residue_cache = {} # New cache for each configuration
            for seq_name, topology in topology_dict.items():
                key = f"{sequence_ordering}_{global_type}_{sidechain_variant}_{heavy_type}"
                permutation = get_permutation(
                    permutations_definition_dict,
                    topology,
                    sequence_ordering,
                    global_type,
                    sidechain_variant,
                    heavy_type,
                    residue_cache
                )
                permutations_dict[seq_name][key] = permutation
                permutations_dict[seq_name][key + "_flip"] = torch.flip(permutation, dims=[0]) # Also add flipped version of the permutation

                pbar.update(1)

    return permutations_dict
