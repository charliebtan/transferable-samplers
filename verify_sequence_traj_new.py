import os
import tarfile
from tqdm import tqdm
from collections import defaultdict

# === CONFIG ===
TAR_DIRS = [
    "/scratch/t/tanc/webdataset_4",
    "/project/aip-necludov/tanc/webdataset_4",
]
EXPECTED_TIME_INDEXES = set(range(0, 200000, 2))

# === Gather all tar paths ===
tar_paths = []
for dir_path in TAR_DIRS:
    tar_paths.extend([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".tar")
    ])

print(f"Found {len(tar_paths)} tar files.")

# === Track time indexes per sequence ===
time_index_map = defaultdict(set)

# === Process each tar file ===
for tar_path in tqdm(sorted(tar_paths), desc="Processing tar files"):
    try:
        with tarfile.open(tar_path, "r") as tar:
            for name in tar.getnames():
                if not name.endswith(".bin"):
                    continue
                base = os.path.basename(name)
                try:
                    seq, index_str = base[:-4].rsplit("_", 1)
                    time_index = int(index_str)
                    time_index_map[seq].add(time_index)
                except ValueError:
                    print(f"Skipping malformed filename: {base}")
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")

# === Validate ===
missing = {}
for seq, found_indexes in tqdm(time_index_map.items(), desc="Validating sequences"):
    missing_indexes = EXPECTED_TIME_INDEXES - found_indexes
    if missing_indexes:
        missing[seq] = sorted(missing_indexes)

# === Report ===
print(f"\nChecked {len(time_index_map)} sequences.")

if missing:
    print(f"\n❌ {len(missing)} sequences are missing time indexes!")
    for seq, miss in list(missing.items())[:10]:  # show first 10
        print(f"- {seq}: {len(miss)} missing (e.g. {miss[:5]}...)")
else:
    print("✅ All sequences have the complete set of time indexes.")
