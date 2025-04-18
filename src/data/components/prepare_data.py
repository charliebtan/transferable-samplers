import logging
import os
import pickle

import huggingface_hub
import lmdb
import mdtraj as md
import numpy as np
import openmm.app
import torch
from tqdm import tqdm

from src.evaluation.plots.plot_atom_distances import interatomic_dist  # TODO move this


def download_data(huggingface_repo_id: str, huggingface_data_dir: str, local_dir: str) -> None:
    """
    Downloads a dat repo from a Hugging Face repository.

    Args:
        huggingface_repo_id (str): The ID of the Hugging Face repository.
        huggingface_data_dir (str): The directory in the repository containing the data.
        local_dir (str): The local directory to save the downloaded data.
    """
    huggingface_hub.snapshot_download(
        repo_id=huggingface_repo_id,
        repo_type="dataset",
        allow_patterns=f"{huggingface_data_dir}/*",
        local_dir=local_dir,
        max_workers=4,
    )


def check_files(data_dir: str) -> tuple[list[str], list[str]]:
    """
    Checks for matching .npz and .pdb files in the given directory.

    Args:
        data_dir (str): The directory containing the data files.

    Returns:
        Tuple[List[str], List[str]]: Lists of valid .npz and .pdb file paths.
    """
    all_files = os.listdir(data_dir)

    npz_files = [file for file in all_files if file.endswith(".npz")]
    pdb_files = [file for file in all_files if file.endswith(".pdb")]

    npz_paths = [os.path.join(data_dir, file) for file in npz_files]
    pdb_paths = [os.path.join(data_dir, file) for file in pdb_files]

    for path in npz_paths[:]:
        if path.replace("-traj-arrays.npz", "-traj-state0.pdb") not in pdb_paths:
            logging.warning(f"File {path} does not have a matching pdb file")
            npz_paths.remove(path)

    for path in pdb_paths[:]:
        if path.replace("-traj-state0.pdb", "-traj-arrays.npz") not in npz_paths:
            logging.warning(f"File {path} does not have a matching npz file")
            pdb_paths.remove(path)

    return npz_paths, pdb_paths


def cross_reference_files(train_npz_paths: list[str], val_npz_paths: list[str]) -> None:
    """
    Ensures there are no common sequences between training and validation datasets.

    Args:
        train_npz_paths (List[str]): List of training .npz file paths.
        val_npz_paths (List[str]): List of validation .npz file paths.

    Raises:
        AssertionError: If common sequences are found between training and validation datasets.
    """
    train_sequences = [os.path.basename(path).split("-")[0] for path in train_npz_paths]
    val_sequences = [os.path.basename(path).split("-")[0] for path in val_npz_paths]

    common_keys = set(train_sequences).intersection(set(val_sequences))
    assert len(common_keys) == 0, f"Common keys found between train and val data dict: {common_keys}"


def build_lmdb(
    npz_paths: list[str],
    pdb_paths: list[str],
    lmdb_path: str,
    map_size: int = 1 << 40,
    batch_size: int = 10 * 50_000,
) -> None:
    """
    Builds an LMDB file from the given .npz and .pdb data.

    Args:
        npz_paths (List[str]): List of .npz file paths.
        pdb_paths (List[str]): List of .pdb file paths.
        lmdb_path (str): Path to save the LMDB file.
        map_size (int, optional): Maximum size of the LMDB file. Defaults to 1 << 40.
        batch_size (int, optional): Number of samples per batch. Defaults to 10 * 50_000.
    """
    if os.path.exists(lmdb_path):
        logging.warning(f"LMDB file {lmdb_path} already exists, skipping LMDB creation")
        return

    # Ensure GPU is available
    assert torch.cuda.is_available(), "GPU is required for data preprocessing - slow otherwise!"

    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    txn.put(b"__len__", pickle.dumps(len(npz_paths)))

    # Global metadata
    min_num_particles = float("inf")
    max_num_particles = 0
    weighted_vars = []
    mean_min_dists = []

    # Sequence metadata
    num_samples_dict = {}
    num_particles_dict = {}
    npz_paths_dict = {}
    pdb_paths_dict = {}

    global_idx = 0
    done = False
    for npz_path in tqdm(npz_paths, desc="Building LMBD"):
        seq_name = os.path.basename(npz_path).split("-")[0]

        with np.load(npz_path, allow_pickle=False) as data:
            x = data["positions"]  # shape (N, num_particles, num_dimensions)

        assert len(x.shape) == 3, f"Expected 3D array, got {x.shape}"

        num_samples, num_particles, _ = x.shape

        # Update min and max num particles
        min_num_particles = min(min_num_particles, num_particles)
        max_num_particles = max(max_num_particles, num_particles)

        x_tensor = torch.from_numpy(x).to("cuda:0")  # move to GPU

        # Compute mean min dist
        dists = interatomic_dist(x_tensor, flatten=False)
        mean_min_dist = dists.min(dim=1)[0].mean()
        mean_min_dists.append(mean_min_dist)

        # Compute std
        x_tensor = x_tensor - x_tensor.mean(dim=1, keepdim=True)  # Center for std computation
        positions_var = x_tensor.var(unbiased=False)
        weighted_vars.append(positions_var * num_samples)

        assert seq_name not in num_samples_dict.keys(), f"Duplicate sequence name {seq_name} found in {npz_path}"
        assert npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb") in pdb_paths

        num_samples_dict[seq_name] = num_samples
        num_particles_dict[seq_name] = num_particles
        npz_paths_dict[seq_name] = npz_path
        pdb_paths_dict[seq_name] = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")

        for pos_idx in range(x.shape[0]):
            # Add a sample to the LMDB batch
            sample = {"seq_name": seq_name, "x": x[pos_idx]}

            key = f"{global_idx:08}".encode()
            value = pickle.dumps(sample)
            txn.put(key, value)

            global_idx += 1

            if global_idx % batch_size == 0:
                # commit the current batch
                txn.commit()
                txn = env.begin(write=True)
                # TODO remove!!
                done = True
                break

        if done:
            break

    total_num_samples = global_idx
    std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_num_samples)
    mean_min_dist = torch.mean(torch.tensor(mean_min_dists))

    metadata = {
        "total_num_samples": total_num_samples,
        "min_num_particles": min_num_particles,
        "max_num_particles": max_num_particles,
        "std": std,
        "mean_min_dist": mean_min_dist,
        "num_samples": num_samples_dict,
        "num_particles": num_particles_dict,
        "pdb_paths": pdb_paths_dict,
        "npz_paths": npz_paths_dict,
    }

    txn.put(b"__meta__", pickle.dumps(metadata))  # store metadata
    txn.put(b"__len__", pickle.dumps(global_idx))  # store number of samples
    txn.commit()
    env.sync()
    env.close()


def load_lmdb_metadata(lmdb_path: str, key: bytes = b"__meta__") -> dict:
    """
    Loads metadata from an LMDB file.

    Args:
        lmdb_path (str): Path to the LMDB file.
        key (bytes, optional): Key to retrieve metadata. Defaults to b"__meta__".

    Returns:
        Dict: Metadata dictionary.

    Raises:
        KeyError: If the metadata key is not found in the LMDB file.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        value = txn.get(key)
        if value is None:
            raise KeyError(f"Metadata key {key} not found in LMDB at {lmdb_path}")
        metadata = pickle.loads(value)  # noqa: S301
    env.close()
    return metadata


def load_pdbs_and_topologies(
    pdb_paths: list[str], num_aa: int
) -> tuple[dict[str, openmm.app.PDBFile], dict[str, md.Topology]]:
    """
    Loads PDB files and their topologies.

    Args:
        pdb_paths (List[str]): List of .pdb file paths.
        num_aa (int): Number of amino acids expected in each PDB file.

    Returns:
        Tuple[Dict[str, openmm.app.PDBFile], Dict[str, md.Topology]]: Dictionaries of PDB files and topologies.

    Raises:
        AssertionError: If a PDB file does not meet the expected criteria.
    """
    pdb_dict = {}
    topology_dict = {}
    logging.info("Loading .pdb files")
    for path in tqdm(pdb_paths):
        seq = os.path.basename(path).split("-")[0]

        pdb = openmm.app.PDBFile(path)
        topology = md.load_topology(path)

        assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"
        assert len(list(pdb.topology.residues())) == num_aa, "PDB does not match the number of amino acids"

        pdb_dict[seq] = pdb
        topology_dict[seq] = topology

    return pdb_dict, topology_dict
