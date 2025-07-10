import webdataset as wds
import torch
import numpy as np
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

NUM_DIMENSIONS = 3
MIN_STD = 0.1
MAX_STD = 1.5
MIN_MAX_ABS = 0.5
MAX_MAX_ABS = 4.0
NUM_WORKERS = 64  # Adjust based on CPU and disk speed

TAR_DIRS = ["/scratch/t/tanc/webdataset_4", "/project/aip-necludov/tanc/webdataset_4"]

tar_paths = []
for dir_path in TAR_DIRS:
    tar_paths.extend(glob.glob(os.path.join(dir_path, "*.tar")))

assert len(tar_paths) > 0, "No .tar files found!"
print(f"{len(tar_paths)} tar files found across {TAR_DIRS}")

dataset = (
    wds.WebDataset(tar_paths, shardshuffle=False, resampled=False)
    .decode()
    .to_tuple("__key__", "bin")
)

bad_samples = []
verified_count = 0

def verify_sample(sample):
    key, sample_bytes = sample
    try:
        arr = np.frombuffer(sample_bytes, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)

        assert arr.dtype == np.float32, f"{key}: Unexpected dtype {arr.dtype}"
        assert arr.ndim == 2 and arr.shape[1] == NUM_DIMENSIONS, f"{key}: Bad shape {arr.shape}"

        std = np.std(arr)
        if not (MIN_STD <= std <= MAX_STD):
            raise ValueError(f"{key}: Unexpected std {std:.4f}")

        max_abs = np.max(np.abs(arr))
        if not (MIN_MAX_ABS <= max_abs <= MAX_MAX_ABS):
            raise ValueError(f"{key}: Unexpected max abs value {max_abs:.4f}")

        return (key, None)  # No error
    except Exception as e:
        return (key, str(e))

# Run in parallel
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(verify_sample, sample) for sample in dataset]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying samples"):
        key, error = future.result()
        if error:
            bad_samples.append((key, error))
        else:
            verified_count += 1

# ==== Summary ====
print(f"\n✅ Total verified samples: {verified_count}")
print(f"❌ Failed samples: {len(bad_samples)}")
if bad_samples:
    print("Bad sample keys:")
    for key, err in bad_samples:
        print(f" - {key}: {err}")
