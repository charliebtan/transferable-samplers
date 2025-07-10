import os
import tarfile
import io
import numpy as np

TAR_DIRS = ["/scratch/t/tanc/webdataset_4", "/project/aip-necludov/tanc/webdataset_4"]
EXPECTED_SECOND_DIM = 3
EXPECTED_DTYPE = np.float32

MIN_STD = 1e-9
MAX_STD = 100

def verify_sample(file_bytes, filename):
    try:
        # Load from raw bytes
        buffer = io.BytesIO(file_bytes)
        array = np.frombuffer(buffer.read(), dtype=EXPECTED_DTYPE)

        if array.size % EXPECTED_SECOND_DIM != 0:
            return False, f"Shape mismatch: size {array.size} not divisible by {EXPECTED_SECOND_DIM}"

        reshaped = array.reshape(-1, EXPECTED_SECOND_DIM)

        # Check values
        if np.isnan(reshaped).any():
            return False, "Contains NaNs"
        if np.isinf(reshaped).any():
            return False, "Contains Infs"

        if reshaped.shape[0] > 1000:
            return False, f"Too long: {reshaped.shape[0]} timesteps"

        # Std sanity check
        std = reshaped.std()
        if not (MIN_STD <= std <= MAX_STD):
            return False, f"Unusual std: {std:.5f}"

        return True, reshaped.shape

    except Exception as e:
        return False, f"Exception: {e}"

def verify_all_tars(tar_dir):
    tar_files = sorted(f for f in os.listdir(tar_dir) if f.endswith(".tar"))
    total_files = 0
    errors = []
    for tar_name in tar_files:
        tar_path = os.path.join(tar_dir, tar_name)
        print(f"Processing {tar_name}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue

                    f = tar.extractfile(member)
                    if f is None:
                        errors.append((tar_name, member.name, "Could not extract"))
                        continue

                    breakpoint()
                    data = f.read()
                    ok, info = verify_sample(data, member.name)

                    if not ok:
                        errors.append((tar_name, member.name, info))

                    total_files += 1

        except Exception as e:
            errors.append((tar_name, None, f"Failed to open tar: {e}"))

    print(f"\nChecked {total_files} files across {len(tar_files)} tarfiles.")
    if errors:
        print(f"\n⚠️ {len(errors)} errors found:")
        for tar, fname, err in errors[:20]:  # show first few
            print(f"  {tar} / {fname}: {err}")
    else:
        print("✅ All samples verified successfully.")

if __name__ == "__main__":
    for TAR_DIR in TAR_DIRS:
        if not os.path.exists(TAR_DIR):
            print(f"Directory {TAR_DIR} does not exist. Skipping.")
            continue
        print(f"Verifying tar files in {TAR_DIR}...")
        verify_all_tars(TAR_DIR)
