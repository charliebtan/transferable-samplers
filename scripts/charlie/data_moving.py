import glob
import os

import numpy as np
from tqdm import tqdm

# 2AA
# train_data = np.load("/home/mila/t/tanc/scratch/self-consume-bg/data/2aa/all_train.npy", allow_pickle=True).item()
# val_data = np.load("/home/mila/t/tanc/scratch/self-consume-bg/data/2aa/all_val.npy", allow_pickle=True).item()
#
# i = 0
# prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
# for key, value in train_data.items():
#     i += 1
#     num_samples = value.shape[0]
#     np.savez(f"{prefix}/{key}-traj-arrays.npz", positions=value.reshape(num_samples, -1, 3))
#     print(i)
#
# prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/val"
# for key, value in val_data.items():
#     i += 1
#     num_samples = value.shape[0]
#     np.savez(f"{prefix}/{key}-traj-arrays.npz", positions=value.reshape(num_samples, -1, 3))
#     print(i)

# train_pdb_file_source = "/home/mila/t/tanc/scratch/self-consume-bg/data/2aa/pdb_train"
# train_pdb_file_dest = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
#
# val_pdb_file_source = "/home/mila/t/tanc/scratch/self-consume-bg/data/2aa/pdb_val"
# val_pdb_file_dest = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/val"

# # Copy train pdb files
# for pdb_file in tqdm(glob.glob(f"{train_pdb_file_source}/*.pdb")):
#     shutil.copy(pdb_file, train_pdb_file_dest)

# Copy validation pdb files
# for pdb_file in tqdm(glob.glob(f"{val_pdb_file_source}/*.pdb")):
#     shutil.copy(pdb_file, val_pdb_file_dest)

# 4AA

# def download_data() -> None:
#     """
#     Downloads a dat repo from a Hugging Face repository.
#
#     Args:
#         huggingface_repo_id (str): The ID of the Hugging Face repository.
#         huggingface_data_dir (str): The directory in the repository containing the data.
#         local_dir (str): The local directory to save the downloaded data.
#     """
#     huggingface_hub.snapshot_download(
#         repo_id="microsoft/timewarp",
#         repo_type="dataset",
#         allow_patterns="4AA-huge/test/**",
#         local_dir="/home/mila/t/tanc/scratch/self-consume-bg/data/new/test",
#         max_workers=4,
#     )
#
# download_data()

# source = "/home/mila/t/tanc/scratch/self-consume-bg/data/4aa/train/4AA-large/train"
# dest = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
#
# # Copy all files from source to destination
# if not os.path.exists(dest):
#     os.makedirs(dest)
#
# for file in tqdm(glob.glob(f"{source}/*")):
#     shutil.copy(file, dest)

# source = "/home/mila/t/tanc/scratch/self-consume-bg/data/4aa/val/4AA-large/val"
# dest = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/val"
#
# # Copy all files from source to destination
# if not os.path.exists(dest):
#     os.makedirs(dest)
#
# for file in tqdm(glob.glob(f"{source}/*")):
#     shutil.copy(file, dest)

# 3/5/6/7AA

# root_dir = "/home/mila/t/tanc/scratch/md-runner/data/md/"
# dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# dirs_8 = [d for d in dirs if len(d) == 8]

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

# Draw a random sample of 30 from dirs
# np.random.seed(0)
# sampled_dirs = np.random.choice(dirs_8, size=min(30, len(dirs_8)), replace=False).tolist()
# all_aa = "".join(sampled_dirs)
# assert all(code in all_aa for code in all_codes), "Not all codes are in all_aa"
#
# dirs = [d for d in dirs if d not in sampled_dirs]

# def process_md_dir(dirs, prefix):
#
#     for d in tqdm(dirs):
#
#         path = f"{root_dir}/{d}/{d}_310_50000"
#
#         npz_files = [file for file in glob.glob(f"{path}/*.npz") if not file.endswith("_vel.npz")]
#         npz_files = sorted(npz_files, key=lambda x: int(os.path.basename(x).split("-")[0][:-4]))
#
#         npz_list = []
#         for npz_file in npz_files:
#             npz_data = np.load(npz_file, allow_pickle=True)
#             npz_list.append(npz_data["all_positions"])
#
#         if len(npz_list):
#             npz_cat = np.concatenate(npz_list, axis=0)
#             np.savez(f"{prefix}/{d}-traj-arrays.npz", positions=npz_cat)
#         else:
#             print(f"No data found in {path}")
#
# dest_dirs = os.listdir("/home/mila/t/tanc/scratch/self-consume-bg/data/new/train")
#
# val_dirs = [dir for dir in dirs if f"{dir}-traj-arrays.npz" not in dest_dirs]
# val_dirs = [dir for dir in val_dirs if len(os.listdir(f"/home/mila/t/tanc/scratch/md-runner/data/md/{dir}"))]
#
# assert len(val_dirs) == 30, "Validation directories are not empty"

# prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
# process_md_dir(dirs, prefix)

# prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/val"
# process_md_dir(val_dirs, prefix)
#
# pdb_dir = "/home/mila/t/tanc/scratch/md-runner/data/pdbs/"
# pdb_files = [f for f in os.listdir(pdb_dir) if os.path.isfile(os.path.join(pdb_dir, f))]
#
# prefix_train = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
# prefix_val = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/val"
#
# for pdb_file in tqdm(pdb_files):
#
#     seq_name = pdb_file.split(".")[0]
#
#     if seq_name in val_dirs:
#         shutil.copy(os.path.join(pdb_dir, pdb_file), f"{prefix_val}/{seq_name}-traj-state0.pdb")
#     elif seq_name in dirs:
#         shutil.copy(os.path.join(pdb_dir, pdb_file), f"{prefix_train}/{seq_name}-traj-state0.pdb")


# Generating subset
# files = os.listdir("/home/mila/t/tanc/scratch/self-consume-bg/data/new/val")
# npz_files = [file for file in files if file.endswith("-traj-arrays.npz")]
#
# seq_names = [file.split("-")[0] for file in npz_files]
#
# seq_names_2 = [name for name in seq_names if len(name) == 2]
# seq_names_4 = [name for name in seq_names if len(name) == 4]
# seq_names_8 = [name for name in seq_names if len(name) == 8]
#
# np.random.seed(0)
#
#
# def draw_random_subset(seq_names, size):
#     sampled = []
#     while len(sampled) < size:
#         subset = np.random.choice(seq_names, size=size, replace=False).tolist()
#         all_aa = "".join(subset)
#         if all(code in all_aa for code in all_codes):
#             sampled = subset
#         else:
#             print("Not all codes are in the sampled subset. Retrying...")
#     return sampled
#
#
# sampled_seq_names_2 = draw_random_subset(seq_names_2, 30)
# sampled_seq_names_4 = draw_random_subset(seq_names_4, 30)
# sampled_seq_names_8 = draw_random_subset(seq_names_8, 30)
#
# all_sampled_seq_names = sampled_seq_names_2 + sampled_seq_names_4 + sampled_seq_names_8
#
# for i, name in enumerate(all_sampled_seq_names):
#     print(f"{name}: {i}")
#
#
# VALIDATION_SUBSET_MIXED = {
#     "HD": 0,
#     "DQ": 1,
#     "SM": 2,
#     "AL": 3,
#     "NG": 4,
#     "TG": 5,
#     "HQ": 6,
#     "DD": 7,
#     "MF": 8,
#     "WY": 9,
#     "VPAA": 30,
#     "VGPY": 31,
#     "YFGV": 32,
#     "CKQV": 33,
#     "LNTG": 34,
#     "RAHW": 35,
#     "TVGR": 36,
#     "VSNK": 37,
#     "NRTL": 38,
#     "NALE": 39,
#     "EFWNDGED": 60,
#     "RPVHFCMY": 61,
#     "IDFAELFV": 62,
#     "IWGYQNFM": 63,
#     "FDVSNTVE": 64,
#     "VKHELQPE": 65,
#     "YYMYVAAG": 66,
#     "GYAADIYH": 67,
#     "FDFTFRCL": 68,
#     "TQRFRNCL": 69,
# }
#
# data_subset = VALIDATION_SUBSET_MIXED.keys()
#
# all_aa = "".join(data_subset)
# if all(code in all_aa for code in all_codes):
#     print("All codes are present in the sampled subset.")

# 3/5/6/7AA - majdi

# root_dir = "/network/scratch/m/majdi.hassan/md-runner/data/md"
# dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
#
# def process_md_dir(dirs, prefix):
#     for d in tqdm(dirs):
#         path = f"{root_dir}/{d}/{d}_310_50000"
#
#         npz_files = [file for file in glob.glob(f"{path}/*.npz") if not file.endswith("_vel.npz")]
#         npz_files = sorted(npz_files, key=lambda x: int(os.path.basename(x).split("-")[0][:-4]))
#
#         npz_list = []
#         for npz_file in npz_files:
#             npz_data = np.load(npz_file, allow_pickle=True)
#             npz_list.append(npz_data["all_positions"])
#
#         if len(npz_list):
#             npz_cat = np.concatenate(npz_list, axis=0)
#             np.savez(f"{prefix}/{d}-traj-arrays.npz", positions=npz_cat)
#         else:
#             print(f"No data found in {path}")
#
#
# prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
# process_md_dir(dirs, prefix)
#
# pdb_dir = "/network/scratch/m/majdi.hassan/md-runner/data/pdbs"
# pdb_files = [f for f in os.listdir(pdb_dir) if os.path.isfile(os.path.join(pdb_dir, f))]
#
# prefix_train = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"
#
# for pdb_file in tqdm(pdb_files):
#     seq_name = pdb_file.split(".")[0]
#     shutil.copy(os.path.join(pdb_dir, pdb_file), f"{prefix_train}/{seq_name}-traj-state0.pdb")


from glob import glob

prefix = "/home/mila/t/tanc/scratch/self-consume-bg/data/new/train"

# Get all .npz files in the target directory
npz_files = glob(os.path.join(prefix, "*-traj-arrays.npz"))

corrupted = []

print("🔍 Checking for corrupted .npz files...")

for f in tqdm(npz_files):
    try:
        _ = np.load(f)  # attempt to load
    except Exception as e:
        print(f"❌ Corrupted: {f} ({e})")
        corrupted.append(f)

if not corrupted:
    print("✅ All .npz files loaded successfully!")
else:
    print(f"\n⚠️ Found {len(corrupted)} corrupted file(s).")
