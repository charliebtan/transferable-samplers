import os

# Configuration
BASE_DIR_1 = "/home/mila/t/tanc/scratch/self-consume-bg/logs/eval/multiruns/2025-05-11_01-44-04"
BASE_DIR_2 = "/home/mila/t/tanc/scratch/self-consume-bg/logs/eval/multiruns/2025-05-10_02-21-17"
SEQ_NAMES = [
    "AC",
    "AT",
    "ET",
    "GN",
    "GP",
    "HT",
    "IM",
    "KG",
    "KQ",
    "KS",
    "LW",
    "NF",
    "NY",
    "RL",
    "RV",
    "TD",
    "SAEL",
    "RYDT",
    "CSFQ",
    "FALS",
    "CSGS",
    "LPEM",
    "LYVI",
    "AYTG",
    "VCVS",
    "AAEW",
    "FKVP",
    "NQFM",
    "DTDL",
    "CTSA",
    "ANYT",
    "VTST",
    "AWKC",
    "RGSP",
    "AVEK",
    "FIYG",
    "VLSM",
    "QADY",
    "DQAL",
    "TFFL",
    "FIGE",
    "KKQF",
    "SLTC",
    "ITQD",
    "DFKS",
    "QDED",
]


def count_successful_sequences(base_dir, seq_names):
    found_runs = {seq: [] for seq in seq_names}

    for i in range(0, 500):
        path = f"{base_dir}/{i}"

        if os.path.exists(path):
            path = os.path.join(path, "test")
            if os.path.exists(path):
                seq = os.listdir(path)[0]
                path = os.path.join(path, seq)

                if len(os.listdir(path)) == 0:
                    continue
                else:
                    idx = os.listdir(path)[0][-4]
                    found_runs[seq].append(idx)

    return found_runs


if __name__ == "__main__":
    results_1 = count_successful_sequences(BASE_DIR_1, SEQ_NAMES)
    results_2 = count_successful_sequences(BASE_DIR_2, SEQ_NAMES)
    for seq in SEQ_NAMES:
        found_runs = results_1[seq] + results_2[seq]
        not_found_runs = []
        for i in range(10):
            if str(i) not in found_runs:
                not_found_runs.append(i)
        print(f"{seq}: {not_found_runs}")
