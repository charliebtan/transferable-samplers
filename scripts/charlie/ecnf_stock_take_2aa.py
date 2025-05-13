import os

ALL_DATES = ['2025-05-11_22-22-44']

BASE_ROOT = "/home/mila/t/tanc/scratch/self-consume-bg/logs/eval/multiruns"
SEQ_NAMES = [
    "AC", "AT", "ET", "GN", "GP", "HT", "IM", "KG", "KQ", "KS", "LW", "NF", "NY",
    "RL", "RV", "TD",]

def count_successful_sequences(base_dir, seq_names):
    found_runs = {seq: [] for seq in seq_names}

    for i in range(0, 500):
        path = os.path.join(base_dir, str(i))
        if os.path.exists(path):
            test_path = os.path.join(path, "test")
            if os.path.exists(test_path):
                seq_list = os.listdir(test_path)
                if not seq_list:
                    continue
                seq = seq_list[0]
                result_path = os.path.join(test_path, seq)
                if os.path.exists(result_path) and os.listdir(result_path):
                    idx = os.listdir(result_path)[0][-4]
                    found_runs[seq].append(idx)

    return found_runs

if __name__ == "__main__":
    # Aggregate results from all dates
    total_results = {seq: [] for seq in SEQ_NAMES}

    for date in ALL_DATES:
        base_dir = os.path.join(BASE_ROOT, date)
        results = count_successful_sequences(base_dir, SEQ_NAMES)
        for seq in SEQ_NAMES:
            total_results[seq].extend(results[seq])

    # Report missing runs
    for seq in SEQ_NAMES:
        found_runs = total_results[seq]
        not_found_runs = [i for i in range(10) if str(i) not in found_runs]
        print(f"{seq}: {not_found_runs}")