import logging
import math
import os
import pickle

import lmdb
import lz4
import mdtraj as md
import numpy as np
import openmm.app
import torch
from tqdm import tqdm

from src.evaluation.plots.plot_atom_distances import interatomic_dist  # TODO move this


def check_and_get_files(data_dir: str) -> tuple[list[str], list[str]]:
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


@torch.no_grad()
def build_lmdb(
    npz_paths: list[str],
    pdb_paths: list[str],
    lmdb_path: str,
    subset: dict[str, list[int]] = None,  # {seq_name: random_seed}
    map_size: int = 3 * (1 << 40),
    resume: bool = False,
) -> None:
    """
    Builds an LMDB file from the given .npz and .pdb data.

    Args:
        npz_paths (List[str]): List of .npz file paths.
        pdb_paths (List[str]): List of .pdb file paths.
        lmdb_path (str): Path to save the LMDB file.
        subset (Dict[str, list[int]], optional): Dict of sequence names and random seeds for sampling. Defaults to None.
        map_size (int, optional): Maximum size of the LMDB file. Defaults to 3 * (1 << 40).
        resume (bool, optional): Whether to resume from an existing LMDB file. Defaults to False.
    """
    if os.path.exists(lmdb_path) and not resume:
        logging.warning(f"LMDB file {lmdb_path} already exists, skipping LMDB creation")
        return
    assert not (not os.path.exists(lmdb_path) and resume), "LMDB file does not exist, cannot resume"

    # Ensure GPU is available - TODO is it actually faster?
    assert torch.cuda.is_available(), "GPU is required for data preprocessing - slow otherwise!"

    # Was getting an error trying to create the LMDB file with a map size larger than 1TB
    env = lmdb.open(lmdb_path, map_size=min(map_size, 1 << 40), writemap=True)
    env.set_mapsize(map_size)  # Set the map size to the full size after creation
    txn = env.begin(write=True)

    if not resume:
        metadata = {
            "total_num_samples": 0,
            "min_num_particles": float("inf"),
            "max_num_particles": 0,
            "weighted_vars": {},
            "num_samples": {},
            "num_particles": {},
            "pdb_paths": {},
            "npz_paths": {},
            "seq_idx": {},
        }
    else:
        metadata = load_lmdb_metadata(lmdb_path)

    np.random.seed(0)
    np.random.shuffle(npz_paths)  # shuffle the npz files for more even timing estimate

    global_idx = 0
    for seq_idx, npz_path in enumerate(tqdm(npz_paths, desc="Building LMDB")):
        seq_name = os.path.basename(npz_path).split("-")[0]

        if subset is not None:
            if seq_name not in subset.keys():
                # avoids adding all the extra ones we're not going to use
                logging.info(f"Sequence {seq_name} not found in subset, skipping")
                continue

        if seq_name in metadata["num_samples"].keys():
            # already added this sequence
            logging.info(f"Sequence {seq_name} already added, skipping")
            continue

        assert npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb") in pdb_paths

        with np.load(npz_path, allow_pickle=False) as data:
            x = data["positions"]  # shape (N, num_particles, num_dimensions)

        if len(seq_name) == 2:
            x = x / 30.0  # 2AA is currently scaled wrong in the files

        assert len(x.shape) == 3, f"Expected 3D array, got {x.shape}"

        if subset is not None:
            # draw a deterministic random subset of 10k samples
            np.random.seed(subset[seq_name])
            x = x[np.random.choice(x.shape[0], size=10_000, replace=False)]

        num_samples, num_particles, _ = x.shape

        x_tensor = torch.from_numpy(x).to("cuda:0")  # move to GPU

        # Check data is on the correct scale - checks the minimum distances between particles
        # (corresponding to an N-H bond) is close to the expected value of 0.1 nm
        dists = interatomic_dist(
            x_tensor[:: max(1, x_tensor.shape[0] // 100)], flatten=False
        )  # only compute dists on a subset
        mean_min_dist = dists.min(dim=1)[0].mean()
        assert math.isclose(mean_min_dist.item(), 0.1, rel_tol=0.1), (
            f"Mean min dist {mean_min_dist.item()} is not close to expected value of 0.1, "
            "data is probably not scaled correctly (to nanometers)"
        )

        # Compute var
        x_tensor = x_tensor - x_tensor.mean(dim=1, keepdim=True)  # Center for var computation
        positions_var = x_tensor.var(unbiased=False)

        # Update global metadata
        metadata["min_num_particles"] = min(metadata["min_num_particles"], num_particles)
        metadata["max_num_particles"] = max(metadata["max_num_particles"], num_particles)
        metadata["total_num_samples"] += num_samples

        # Add sequence metadata
        metadata["weighted_vars"][seq_name] = float(positions_var * num_samples)
        metadata["num_samples"][seq_name] = int(num_samples)
        metadata["num_particles"][seq_name] = int(num_particles)
        metadata["npz_paths"][seq_name] = npz_path
        metadata["pdb_paths"][seq_name] = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")
        metadata["seq_idx"][seq_name] = []

        for i in range(num_samples):
            # Create sequence data item
            sample = {"seq_name": seq_name, "x": x[i]}
            metadata["seq_idx"][seq_name].append(global_idx)

            # Prepare the data for LMDB
            key = f"{global_idx:08}".encode()
            value = lz4.frame.compress(pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL))  # Compress the data
            txn.put(key, value)
            global_idx += 1

        metadata["seq_idx"][seq_name] = np.array(
            metadata["seq_idx"][seq_name], dtype=np.int32
        )  # convert to numpy array

        if seq_idx % 500 == 0:
            # Store the data in the LMDB
            txn.put(b"__meta__", pickle.dumps(metadata))  # store metadata
            txn.put(b"__len__", pickle.dumps(global_idx))  # store number of samples
            txn.commit()
            txn = env.begin(write=True)

    # Store the final data
    txn.put(b"__meta__", pickle.dumps(metadata))  # store metadata
    txn.put(b"__len__", pickle.dumps(global_idx))  # store number of samples

    # Store the data in the LMDB
    txn.commit()


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
    pdb_paths: list[str], num_aa_range: int
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
        assert len(list(pdb.topology.residues())) in num_aa_range, "PDB does not match the number of amino acids"

        pdb_dict[seq] = pdb
        topology_dict[seq] = topology

    return pdb_dict, topology_dict
