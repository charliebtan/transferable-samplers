# TODO this could be done using yam

# Taken from TBG OSF
TEST_SUBSET_2 = {
    "AC": 0,
    "AT": 1,
    "ET": 2,
    "GN": 3,
    "GP": 4,
    "HT": 5,
    "IM": 6,
    "KG": 7,
    "KQ": 8,
    "KS": 9,
    "LW": 10,
    "NF": 11,
    "NY": 12,
    "RL": 13,
    "RV": 14,
    "TD": 15,
}

# Subset of huggingface - given by Leon
# THERE IS NO H IN THE TEST SUBSET 4
TEST_SUBSET_4 = {
    "SAEL": 16,
    "RYDT": 17,
    "CSFQ": 18,
    "FALS": 19,
    "CSGS": 20,
    "LPEM": 21,
    "LYVI": 22,
    "AYTG": 23,
    "VCVS": 24,
    "AAEW": 25,
    "FKVP": 26,
    "NQFM": 27,
    "DTDL": 28,
    "CTSA": 29,
    "ANYT": 30,
    "VTST": 31,
    "AWKC": 32,
    "RGSP": 33,
    "AVEK": 34,
    "FIYG": 35,
    "VLSM": 36,
    "QADY": 37,
    "DQAL": 38,
    "TFFL": 39,
    "FIGE": 40,
    "KKQF": 41,
    "SLTC": 42,
    "ITQD": 43,
    "DFKS": 44,
    "QDED": 45,
}

ALL_TEST_SUBSET = {**TEST_SUBSET_2, **TEST_SUBSET_4}

TEST_SUBSET_DICT = {
    "2": TEST_SUBSET_2,
    "4": TEST_SUBSET_4,
}

if __name__ == "__main__":
    all_codes = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    # Check all AA present at least once in the subsets
    for subset_name, subset in TEST_SUBSET_DICT.items():
        missing_codes = [code for code in all_codes if not any(code in key for key in subset.keys())]
        if missing_codes:
            print(f"Subset {subset_name} is missing the following codes: {missing_codes}")
        else:
            print(f"Subset {subset_name} contains all codes.")
