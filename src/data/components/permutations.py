BACKBONE_PERMUTATIONS = {
    "n2c": ["N", "H1", "H2", "H3", "CA", "HA", "HA3", "C", "O", "OXT"],
    "c2n": ["OXT", "C", "O", "CA", "HA", "HA3", "N", "H1", "H2", "H3"]
}

SIDECHAIN_PERMUTATIONS = {
    "ALA": {
        "groupwise": ["CB", "HB1", "HB2", "HB3"],
        "heavy_light": ["CB", "HB1", "HB2", "HB3"],
    },
    "ARG": {
        "groupwise": ["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "HD2", "HD3", "NE", "HE", "CZ", "NH1", "HH11", "HH12", "NH2", "HH21", "HH22"],
        "heavy_light": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH11", "HH12", "HH21", "HH22"],
    },
    "ASN": {
    "groupwise": ["CB", "HB2", "HB3", "CG", "OD1", "ND2", "HD21", "HD22"],
    "heavy_light": ["CB", "CG", "OD1", "ND2", "HB2", "HB3", "HD21", "HD22"],
    },
    "ASP": {
    "groupwise": ["CB", "HB2", "HB3", "CG", "OD1", "OD2"],
    "heavy_light": ["CB", "CG", "OD1", "OD2", "HB2", "HB3"],
    },
    "CYS": {
    "groupwise": ["CB", "HB2", "HB3", "SG", "HG"],
    "heavy_light": ["CB", "SG", "HB2", "HB3", "HG"],
    },
    "GLN": {
    "groupwise": ["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "NE2", "HE21", "HE22"],
    "heavy_light": ["CB", "CG", "CD", "OE1", "NE2", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"],
    },
    "GLU": {
    "groupwise": ["CB", "HB2", "HB3", "CG", "HG2", "HG3", "CD", "OE1", "OE2"],
    "heavy_light": ["CB", "CG", "CD", "OE1", "OE2", "HB2", "HB3", "HG2", "HG3"],
    },
    "GLY": {
        "groupwise": [],
        "heavy_light": [],
    },
    "HIE": {
    "groupwise": ["CB", "HB2", "HB3", "CG", "ND1", "CE1", "HE1", "NE2", "HE2", "CD2", "HD2"],
    "groupwise_ring_reverse": ["CB", "HB2", "HB3", "CG", "CD2", "HD2", "NE2", "HE2", "CE1", "HE1", "ND1"],
    "heavy_light": ["CB", "CG", "ND1", "CE1", "NE2", "CD2", "HB2", "HB3", "HE1", "HE2", "HD2"],
    "heavy_light_ring_reverse": ["CB", "CG", "CD2", "NE2", "CE1", "ND1", "HB2", "HB3", "HD2", "HE2", "HE1"],
    },
    "ILE": {
    "groupwise": [
        "CB", "HB",
        "CG2", "HG21", "HG22", "HG23",
        "CG1", "HG12", "HG13",
        "CD1", "HD11", "HD12", "HD13"
    ],
    "groupwise_branch_flip": [
        "CB", "HB",
        "CG1", "HG12", "HG13",
        "CD1", "HD11", "HD12", "HD13",
        "CG2", "HG21", "HG22", "HG23",
    ],
    "heavy_light": [
        "CB", "CG2", "CG1", "CD1",
        "HB", "HG21", "HG22", "HG23", "HG12", "HG13", "HD11", "HD12", "HD13"
    ],
    "heavy_light_branch_flip": [
        "CB", "CG1", "CD1", "CG2", 
        "HB", "HG12", "HG13", "HD11", "HD12", "HD13", "HG21", "HG22", "HG23", 
    ],
    },
    "LYS": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG", "HG2", "HG3",
        "CD", "HD2", "HD3",
        "CE", "HE2", "HE3",
        "NZ", "HZ1", "HZ2", "HZ3"
    ],
    "heavy_light": [
        "CB", "CG", "CD", "CE", "NZ",
        "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"
    ],
},
"MET": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG", "HG2", "HG3",
        "SD",
        "CE", "HE1", "HE2", "HE3"
    ],
    "heavy_light": [
        "CB", "CG", "SD", "CE",
        "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"
    ],
},
"PHE": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG",
        "CD1", "HD1",
        "CE1", "HE1",
        "CZ", "HZ",
        "CE2", "HE2",
        "CD2", "HD2"
    ],
    "groupwise_ring_reverse": [
        "CB", "HB2", "HB3",
        "CG",
        "CD2", "HD2",
        "CE2", "HE2",
        "CZ", "HZ",
        "CE1", "HE1",
        "CD1", "HD1"
    ],
    "heavy_light": [
        "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2",
        "HB2", "HB3", "HD1", "HE1", "HZ", "HE2", "HD2"
    ],
    "heavy_light_ring_reverse": [
        "CB", "CG", "CD2", "CE2", "CZ", "CE1", "CD1",
        "HB2", "HB3", "HD2", "HE2", "HZ", "HE1", "HD1"
    ],
},
"PRO": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG", "HG2", "HG3",
        "CD", "HD2", "HD3"
    ],
    "groupwise_ring_reverse": [
        "CD", "HD2", "HD3",
        "CG", "HG2", "HG3",
        "CB", "HB2", "HB3"
    ],
    "heavy_light": [
        "CB", "CG", "CD",
        "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"
    ],
    "heavy_light_ring_reverse": [
        "CD", "CG", "CB",
        "HD2", "HD3", "HG2", "HG3", "HB2", "HB3"
    ],
},
"SER": {
    "groupwise": ["CB", "HB2", "HB3", "OG", "HG"],
    "heavy_light": ["CB", "OG", "HB2", "HB3", "HG"],
},
"THR": {
    "groupwise": [
        "CB", "HB",
        "OG1", "HG1",
        "CG2", "HG21", "HG22", "HG23"
    ],
    "groupwise_branch_flip": [
        "CB", "HB",
        "CG2", "HG21", "HG22", "HG23",
        "OG1", "HG1"
    ],
    "heavy_light": [
        "CB", "OG1", "CG2",
        "HB", "HG1", "HG21", "HG22", "HG23"
    ],
    "heavy_light_branch_flip": [
        "CB", "CG2", "OG1",
        "HB", "HG21", "HG22", "HG23", "HG1"
    ],
},
"TRP": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG",
        "CD1", "HD1",
        "NE1", "HE1",
        "CE2",
        "CZ2", "HZ2",
        "CH2", "HH2",
        "CZ3", "HZ3",
        "CE3", "HE3",
        "CD2"
    ],
    "groupwise_ring_reverse": [
        "CB", "HB2", "HB3",
        "CG",
        "CD2",
        "CE3", "HE3",
        "CZ3", "HZ3",
        "CH2", "HH2",
        "CZ2", "HZ2",
        "CE2",
        "NE1", "HE1",
        "CD1", "HD1",
    ],
    "heavy_light": [
        "CB", "CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2",
        "HB2", "HB3", "HD1", "HE1", "HE3", "HZ3", "HH2", "HZ2"
    ],
    "heavy_light_ring_reverse": [
        "CB", "CG", "CD2", "CE3", "CZ3", "CH2", "CZ2", "CE2", "NE1", "CD1",
        "HB2", "HB3", "HE3", "HZ3", "HH2", "HZ2", "HE1", "HD1"
    ],
},
"TYR": {
    "groupwise": [
        "CB", "HB2", "HB3",
        "CG",
        "CD1", "HD1",
        "CE1", "HE1",
        "CZ",
        "OH", "HH",
        "CE2", "HE2",
        "CD2", "HD2"
    ],
    "groupwise_ring_reverse": [
        "CB", "HB2", "HB3",
        "CG",
        "CD2", "HD2",
        "CE2", "HE2",
        "CZ",
        "OH", "HH",
        "CE1", "HE1",
        "CD1", "HD1"
    ],
    "heavy_light": [
        "CB", "CG", "CD1", "CE1", "CZ", "OH", "CE2", "CD2",
        "HB2", "HB3", "HD1", "HE1", "HH", "HE2", "HD2"
    ],
    "heavy_light_ring_reverse": [
        "CB", "CG", "CD2", "CE2", "CZ", "OH", "CE1", "CD1",
        "HB2", "HB3", "HD2", "HE2", "HH", "HE1", "HD1"
    ],
},
"VAL": {
    "groupwise": [
        "CB", "HB",
        "CG1", "HG11", "HG12", "HG13",
        "CG2", "HG21", "HG22", "HG23"
    ],
    "groupwise_branch_flip": [
        "CB", "HB",
        "CG2", "HG21", "HG22", "HG23",
        "CG1", "HG11", "HG12", "HG13"
    ],
    "heavy_light": [
        "CB", "CG1", "CG2",
        "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"
    ],
    "heavy_light_branch_flip": [
        "CB", "CG2", "CG1",
        "HB", "HG21", "HG22", "HG23", "HG11", "HG12", "HG13"
    ],
}
}

def get_permutation(topology, backbone=False, heavy_light=False, structural_variant=False):
    pass

def get_permutations(topology):
    pass

def get_permutations_dict(topology_dict):
    permutations_dict = {}
    for seq_name, topology in topology_dict.items():
        permutations_dict[seq_name] = get_permutations(topology)
    return permutations_dict
