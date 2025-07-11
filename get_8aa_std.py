import webdataset as wds
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==== Config ====
NUM_DIMENSIONS = 3
MIN_STD, MAX_STD = 0.1, 1.5
MIN_MAX_ABS, MAX_MAX_ABS = 0.5, 4.0
NUM_WORKERS = 8

TAR_DIRS = ["/scratch/t/tanc/webdataset_4", "/project/aip-necludov/tanc/webdataset_4"]

# === Gather .tar files ===
tar_paths = []
for dir_path in TAR_DIRS:
    tar_paths.extend(glob.glob(os.path.join(dir_path, "*.tar")))
assert tar_paths, "No .tar files found!"

print(f"{len(tar_paths)} tar files found.")

# === Verification per tar ===
def get_8aa_data(tar_path):
    dataset = (
        wds.WebDataset([tar_path], shardshuffle=False).decode()
        .to_tuple("__key__", "bin")  # no decode() if raw binary
    )
    data = []
    for key, sample_bytes in dataset:
        if len(key.split("_")[0]) == 4:
            arr = np.frombuffer(sample_bytes, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)
            com = np.mean(arr, axis=0)
            arr = arr - com
            data.append(arr)
    return data

# === Limit number of tar files for debugging ===
tar_paths = tar_paths[:10]

# === Run multiprocessing ===
all_data = []

for tar_path in tqdm(tar_paths, desc="Processing tar files"):
    all_data.extend(get_8aa_data(tar_path))

for i in range(len(all_data)):
    print(np.concatenate(all_data[: 10 ** i]).std())