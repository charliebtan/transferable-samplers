import torch

ATOM_TYPE_ENCODING_DICT = {
    "C": 0,
    "CA": 1,
    "CB": 2,
    "CD": 3,
    "CD1": 4,
    "CD2": 5,
    "CE": 6,
    "CE1": 7,
    "CE2": 8,
    "CE3": 9,
    "CG": 10,
    "CG1": 11,
    "CG2": 12,
    "CH2": 13,
    "CZ": 14,
    "CZ2": 15,
    "CZ3": 16,
    "H": 17,
    "HA": 18,
    "HB": 19,
    "HD": 20,
    "HD1": 21,
    "HD2": 22,
    "HE": 23,
    "HE1": 24,
    "HE2": 25,
    "HE3": 26,
    "HG": 27,
    "HG1": 28,
    "HG2": 29,
    "HH": 30,
    "HH1": 31,
    "HH2": 32,
    "HZ": 33,
    "HZ2": 34,
    "HZ3": 35,
    "N": 36,
    "ND1": 37,
    "ND2": 38,
    "NE": 39,
    "NE1": 40,
    "NE2": 41,
    "NH1": 42,
    "NH2": 43,
    "NZ": 44,
    "O": 45,
    "OD": 46,
    "OE": 47,
    "OG": 48,
    "OG1": 49,
    "OH": 50,
    "OXT": 51,
    "SD": 52,
    "SG": 53,
}

AA_TYPE_ENCODING_DICT = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}


def get_atom_encoding(topology):
    aa_pos_encoding = []
    aa_type_encoding = []
    atom_type_encoding = []

    for i, aa in enumerate(topology.residues):
        for atom in aa.atoms:
            aa_pos_encoding.append(i)
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

    atom_type_encoding = torch.tensor(atom_type_encoding)
    aa_pos_encoding = torch.tensor(aa_pos_encoding)
    aa_type_encoding = torch.tensor(aa_type_encoding)

    encodings = {
        "atom_type": atom_type_encoding,
        "aa_pos": aa_pos_encoding,
        "aa_type": aa_type_encoding,
    }

    return encodings
