import mdtraj as md
import numpy as np
import openmm
import torch


def get_atom_types(topology):
    atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))

    return atom_types


def get_adj_list(topology):
    adj_list = torch.from_numpy(
        np.array(
            [(b.atom1.index, b.atom2.index) for b in topology.bonds],
            dtype=np.int32,
        )
    )
    return adj_list


import logging
import os
import pickle

import huggingface_hub
import lmdb
from tqdm import tqdm

from src.evaluation.plots.plot_atom_distances import interatomic_dist  # TODO move this


def download_data(huggingface_repo_id: str, huggingface_data_dir: str, local_dir: str) -> None:
    # Download the training data
    huggingface_hub.snapshot_download(
        repo_id=huggingface_repo_id,
        repo_type="dataset",
        allow_patterns=f"{huggingface_data_dir}/*",
        local_dir=local_dir,
        max_workers=4,
    )


def check_files(data_dir):
    all_files = os.listdir(data_dir)

    npz_files = [file for file in all_files if file.endswith(".npz")]
    pdb_files = [file for file in all_files if file.endswith(".pdb")]

    npz_paths = [os.path.join(data_dir, file) for file in npz_files]
    pdb_paths = [os.path.join(data_dir, file) for file in pdb_files]

    for path in npz_paths:
        if path.replace("-traj-arrays.npz", "-traj-state0.pdb") not in pdb_paths:
            breakpoint()
            logging.warning(f"File {path} does not have a matching pdb file")
            npz_paths.remove(path)

    for path in pdb_paths:
        if path.replace("-traj-state0.pdb", "-traj-arrays.npz") not in npz_paths:
            logging.warning(f"File {path} does not have a matching npz file")
            pdb_paths.remove(path)

    return npz_paths, pdb_paths


def cross_reference_files(train_npz_paths, val_npz_paths):
    train_sequences = [os.path.basename(path).split("-")[0] for path in train_npz_paths]
    val_sequences = [os.path.basename(path).split("-")[0] for path in val_npz_paths]

    common_keys = set(train_sequences).intersection(set(val_sequences))
    assert len(common_keys) == 0, f"Common keys found between train and val data dict: {common_keys}"


def build_lmdb(npz_paths, pdb_paths, lmdb_path, zero_center_of_mass, map_size=1 << 40, batch_size=10 * 50_000):
    """
    Builds a LMDB file from the data in the given path.
    """

    if os.path.exists(lmdb_path):
        logging.warning(f"LMDB file {lmdb_path} already exists, skipping LMDB creation")
        return
    env = lmdb.open(lmdb_path, map_size=map_size)

    txn = env.begin(write=True)
    txn.put(b"__len__", pickle.dumps(len(npz_paths)))

    global_idx = 0

    min_num_particles = 1e1
    max_num_particles = 0
    weighted_vars = []
    seq_names = []
    mean_min_dists = []

    done = False  # TODO remove
    for path in tqdm(npz_paths, desc="Building LMBD"):
        seq_name = os.path.basename(path).split("-")[0]

        with np.load(path, allow_pickle=False) as data:
            x = data["positions"]  # shape (N, num_particles, num_dimensions)

        assert len(x.shape) == 3, f"Expected 3D array, got {x.shape}"

        num_samples, num_particles, _ = x.shape

        # upddate min and max num particles
        min_num_particles = min(min_num_particles, num_particles)
        max_num_particles = max(max_num_particles, num_particles)

        # the below line will throw an error if no gpu available - the data processing
        # will be prohibitively slow on cpu so we require this
        assert torch.cuda.is_available(), "GPU is required for data preprocessing"
        x_tensor = torch.from_numpy(x).to("cuda:0")  # move to gpu

        # compute mean min dist
        dists = interatomic_dist(x_tensor, flatten=False)
        mean_min_dist = dists.min(dim=1)[0].mean()
        mean_min_dists.append(mean_min_dist)

        # compute std
        x_tensor = x_tensor.view(num_samples, -1)
        x_tensor = zero_center_of_mass(x_tensor)  # have to center for std computation
        positions_var = x_tensor.var(unbiased=False)  # TODO why ubiased=False?
        weighted_vars.append(positions_var * num_samples)

        assert seq_name not in seq_names, f"Duplicate sequence name {seq_name} found in {path}"
        seq_names.append(seq_name)

        for pos_idx in range(x.shape[0]):
            sample = {"seq_name": seq_name, "x": x[pos_idx]}

            key = f"{global_idx:08}".encode()
            value = pickle.dumps(sample)
            txn.put(key, value)

            global_idx += 1

            if global_idx % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
                done = True
                break

        if done:
            break

    total_num_samples = global_idx + 1
    std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_num_samples)
    mean_min_dist = torch.mean(torch.tensor(mean_min_dists))

    metadata = {
        "num_samples": total_num_samples,
        "min_num_particles": min_num_particles,
        "max_num_particles": max_num_particles,
        "std": std,
        "seq_names": seq_names,
        "pdb_paths": pdb_paths,
    }

    txn.put(b"__meta__", pickle.dumps(metadata))  # store metadata
    txn.put(b"__len__", pickle.dumps(global_idx))  # store number of samples
    txn.commit()
    env.sync()
    env.close()


def load_lmdb_metadata(lmdb_path, key=b"__meta__"):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        value = txn.get(key)
        if value is None:
            raise KeyError(f"Metadata key {key} not found in LMDB at {lmdb_path}")
        metadata = pickle.loads(value)
    env.close()
    return metadata


def load_pdbs_and_topologies(pdb_paths, num_aa):
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
