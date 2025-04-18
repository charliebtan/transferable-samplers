import torch

"""Start encodings from 1 to leave 0 for zero padding"""

ATOM_TYPE_ENCODING_DICT = {
    "C": 1,
    "CA": 2,
    "CB": 3,
    "CD": 4,
    "CD1": 5,
    "CD2": 6,
    "CE": 7,
    "CE1": 8,
    "CE2": 9,
    "CE3": 10,
    "CG": 11,
    "CG1": 12,
    "CG2": 13,
    "CH2": 14,
    "CZ": 15,
    "CZ2": 16,
    "CZ3": 17,
    "H": 18,
    "HA": 19,
    "HB": 20,
    "HD": 21,
    "HD1": 22,
    "HD2": 23,
    "HE": 24,
    "HE1": 25,
    "HE2": 26,
    "HE3": 27,
    "HG": 28,
    "HG1": 29,
    "HG2": 30,
    "HH": 31,
    "HH1": 32,
    "HH2": 33,
    "HZ": 34,
    "HZ2": 35,
    "HZ3": 36,
    "N": 37,
    "ND1": 38,
    "ND2": 39,
    "NE": 40,
    "NE1": 41,
    "NE2": 42,
    "NH1": 43,
    "NH2": 44,
    "NZ": 45,
    "O": 46,
    "OD": 47,
    "OE": 48,
    "OG": 49,
    "OG1": 50,
    "OH": 51,
    "OXT": 52,
    "SD": 53,
    "SG": 54,
}

AA_TYPE_ENCODING_DICT = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
}

AA_CODE_CONVERSION = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def get_encoding(topology):
    aa_pos_encoding = []
    aa_type_encoding = []
    atom_type_encoding = []

    for i, aa in enumerate(topology.residues):
        for atom in aa.atoms:
            aa_pos_encoding.append(i + 1)  # shifted to account for pad tokens
            aa_type_encoding.append(AA_TYPE_ENCODING_DICT[aa.name])

            # TODO double check this with Leon
            # Standarize side-chain H atom encoding
            if atom.name[0] == "H" and atom.name[-1] in ("1", "2", "3"):
                # For these AA the H-X-N atoms are not interchangable
                if aa.name in ("HIS", "PHE", "TRP", "TYR") and atom.name[:2] in (
                    "HE",
                    "HD",
                    "HZ",
                    "HH",
                ):
                    pass
                else:
                    atom.name = atom.name[:-1]

            # Standarize side-chain O atom encoding
            if atom.name[:2] == "OE" or atom.name[:2] == "OD":
                atom.name = atom.name[:-1]

            atom_type_encoding.append(ATOM_TYPE_ENCODING_DICT[atom.name])

    atom_type_encoding = torch.tensor(atom_type_encoding, dtype=torch.int64)
    aa_pos_encoding = torch.tensor(aa_pos_encoding, dtype=torch.int64)
    aa_type_encoding = torch.tensor(aa_type_encoding, dtype=torch.int64)

    encoding = {
        "atom_type": atom_type_encoding,
        "aa_pos": aa_pos_encoding,
        "aa_type": aa_type_encoding,
    }

    return encoding


def get_encoding_dict(topology_dict):
    for seq_name, topology in topology_dict.items():
        encoding = get_encoding(topology)
        topology_dict[seq_name] = encoding

    return topology_dict
