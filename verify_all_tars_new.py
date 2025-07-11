import webdataset as wds
import torch
import numpy as np

from tqdm import tqdm
import os

NUM_DIMENSIONS = 3
MIN_STD = 0.1
MAX_STD = 1.5
MIN_MAX_ABS = 0.5
MAX_MAX_ABS = 4.0

import glob

TAR_DIRS = ["/scratch/t/tanc/webdataset_4", "/project/aip-necludov/tanc/webdataset_4"]

tar_paths = []
for dir_path in TAR_DIRS:
    tar_paths.extend(glob.glob(os.path.join(dir_path, "*.tar")))

print(len(tar_paths), "tar files found across directories:", TAR_DIRS)

assert len(tar_paths) > 0, "No .tar files found!"

tar_paths = tar_paths[:1]

dataset = (
    wds.WebDataset(tar_paths, shardshuffle=False, resampled=False)
    .decode()
    .to_tuple("__key__", "bin")  # or "npy"
)

# ==== Verification Loop ====
bad_samples = []
total = 0

num_atoms_max = 0

my_set = set()

for key, sample_bytes in tqdm(dataset, desc="Verifying samples"):
    try:
        # Convert raw bytes to numpy array
        arr = np.frombuffer(sample_bytes, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)

        num_atoms = arr.shape[0]

        sequence = key.split("_")[0]
        seq_len = len(sequence)

        my_set.add(sequence)

        if num_atoms > num_atoms_max and seq_len <= 8:
            num_atoms_max = num_atoms
            print("\n", num_atoms_max)
        continue

        # Check dtype
        assert arr.dtype == np.float32, f"{key}: Unexpected dtype {arr.dtype}"

        # Check shape
        assert arr.ndim == 2 and arr.shape[1] == NUM_DIMENSIONS, f"{key}: Bad shape {arr.shape}"

        # Check standard deviation
        std = np.std(arr)
        max_abs = np.max(np.abs(arr))
        if not (MIN_STD <= std <= MAX_STD):
            raise ValueError(f"{key}: Unexpected std {std:.4f}")
        if not (MIN_MAX_ABS <= max_abs <= MAX_MAX_ABS):
            raise ValueError(f"{key}: Unexpected max abs value {max_abs:.4f}")

        total += 1
        print(f"Total: {total} [OK] {key}: {arr.shape} (std: {std:.4f})")

    except Exception as e:
        print(f"[ERROR] {key}: {e}")
        bad_samples.append(key)
        breakpoint()

print(len(my_set), "unique sequences found")

# ==== Summary ====
print(f"\nTotal verified samples: {total}")
print(f"Failed samples: {len(bad_samples)}")
if bad_samples:
    print("Bad sample keys:")
    for k in bad_samples:
        print(" -", k)