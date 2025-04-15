import numpy as np
import torch


def get_atom_types(topology):
    atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))

    return atom_types


def get_adj_list(topology):
    adj_list = torch.from_numpy(
        np.array(
            [(b.atom1.index, b.atom2.index) for b in topology.bonds],
            dtype=np.int32,
        )
    )
    return adj_list
