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
def verify_tar(tar_path):
    dataset = (
        wds.WebDataset([tar_path], shardshuffle=False)
        .decode()
        .to_tuple("__key__", "bin")
    )
    bad = []
    count = 0
    for key, sample_bytes in dataset:
        try:
            arr = np.frombuffer(sample_bytes, dtype=np.float32).reshape(-1, NUM_DIMENSIONS)
            std = np.std(arr)
            max_abs = np.max(np.abs(arr))
            if not (MIN_STD <= std <= MAX_STD):
                raise ValueError(f"std {std:.4f} out of range")
            if not (MIN_MAX_ABS <= max_abs <= MAX_MAX_ABS):
                raise ValueError(f"max_abs {max_abs:.4f} out of range")
            count += 1
        except Exception as e:
            bad.append((key, tar_path, str(e)))
    return (tar_path, count, bad)

# === Run in parallel ===
total = 0
all_bad = []

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(verify_tar, tar): tar for tar in tar_paths}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying tarfiles"):
        tar_path, count, bad = future.result()
        total += count
        all_bad.extend(bad)

# === Report summary ===
print(f"\n✅ Total verified samples: {total}")
print(f"❌ Total bad samples: {len(all_bad)}")
if all_bad:
    print("\nBad sample details:")
    for key, tar_path, err in all_bad:
        print(f" - {key} (in {os.path.basename(tar_path)}): {err}")
