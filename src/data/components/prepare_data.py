import logging
import math
import os
import pickle

import lmdb
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


def cross_reference_files(npz_paths_a: list[str], npz_paths_b: list[str]) -> None:
    """
    Ensures there are no common sequences between two datasets.

    Args:
        npz_paths_a (List[str]): List of .npz file paths for dataset A.
        npz_paths_b (List[str]): List of .npz file paths for dataset B.

    Raises:
        AssertionError: If common sequences are found between the two datasets.
    """
    sequences_a = [os.path.basename(path).split("-")[0] for path in npz_paths_a]
    sequences_b = [os.path.basename(path).split("-")[0] for path in npz_paths_b]
    common_keys = set(sequences_a).intersection(set(sequences_b))
    assert len(common_keys) == 0, f"Common keys found between: {common_keys}"


@torch.no_grad()
def build_lmdb(
    npz_paths: list[str],
    pdb_paths: list[str],
    lmdb_prefix_path: str,
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

    lmdb_paths = [f"{lmdb_prefix_path}_{i}.lmdb" for i in range(4)]

    if any(os.path.exists(path) and not resume for path in lmdb_paths):
        if not all(os.path.exists(path) for path in lmdb_paths):
            raise ValueError("Some LMDB files exist but not all - cannot proceed")
        else:
            logging.info("LMDB files already exist, skipping LMDB creation")
            return

    if resume:
        assert all(os.path.exists(path) for path in lmdb_paths), "Cannot resume if some LMDB files don't exist"

    # Ensure GPU is available - TODO is it actually faster?
    assert torch.cuda.is_available(), "GPU is required for data preprocessing - slow otherwise!"

    envs = []
    txns = []

    for path in lmdb_paths:
        env = lmdb.open(path, map_size=min(map_size, 1 << 40), writemap=True)
        env.set_mapsize(map_size)
        envs.append(env)
        txns.append(env.begin(write=True))

    if not resume:
        metadata = {
            "vars": {},
            "num_samples": {},
            "num_particles": {},
            "pdb_paths": {},
            "npz_paths": {},
            "seq_to_idx": {},
        }
        global_idx = 0
    else:
        metadata = load_lmdb_metadata(lmdb_paths[0])
        global_idx = pickle.loads(txns[0].get(b"__len__"))  # noqa: S301
        global_idx += 1  # start from the next index

    np.random.seed(0)
    np.random.shuffle(npz_paths)

    for seq_idx, npz_path in enumerate(tqdm(npz_paths, desc="Building LMDBs")):
        seq_name = os.path.basename(npz_path).split("-")[0]

        if subset and seq_name not in subset:
            # avoids adding all the extra ones we're not going to use
            logging.info(f"Sequence {seq_name} not found in subset, skipping")
            continue

        if seq_name in metadata["num_samples"]:
            # already added this sequence
            logging.info(f"Sequence {seq_name} already added, skipping")
            continue

        pdb_path = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")
        assert pdb_path in pdb_paths

        metadata["npz_paths"][seq_name] = npz_path
        metadata["pdb_paths"][seq_name] = pdb_path

        with np.load(npz_path, allow_pickle=False) as data:
            x = data["positions"]  # shape (N, num_particles, num_dimensions)

        if len(seq_name) == 2 and "test" not in lmdb_prefix_path:
            x = x / 30.0  # train / val 2AA is currently scaled wrong in the files # TODO fix files

        assert len(x.shape) == 3, f"Expected 3D array, got {x.shape}"

        if subset:
            # draw a deterministic random subset of 10k samples
            np.random.seed(subset[seq_name])
            x = x[np.random.choice(x.shape[0], size=10_000, replace=False)]

        num_particles = x.shape[1]

        metadata["num_particles"][seq_name] = int(num_particles)

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
        x_var = x_tensor.var(unbiased=False)
        metadata["vars"][seq_name] = float(x_var)

        if "small" in lmdb_prefix_path:
            step = 10
        elif "medium" in lmdb_prefix_path:
            step = 5
        else:
            step = 1

        seq_to_idx = []
        for t_idx in range(0, x.shape[0], step):
            # Create sequence data item
            sample = {"seq_name": seq_name, "x": x[t_idx]}
            seq_to_idx.append(global_idx)

            # Prepare the data for LMDB
            key = f"{global_idx:08}".encode()
            value = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            for txn in txns:
                txn.put(key, value)
            global_idx += 1

        num_samples = len(seq_to_idx)

        metadata["num_samples"][seq_name] = num_samples
        metadata["seq_to_idx"][seq_name] = np.array(seq_to_idx, dtype=np.int32)

        if seq_idx % 500 == 0:
            for i in range(4):
                txns[i].put(b"__meta__", pickle.dumps(metadata))  # store metadata
                txns[i].put(b"__len__", pickle.dumps(global_idx))  # store number of samples
                txns[i].commit()
                txns[i] = envs[i].begin(write=True)

    # Store the final data
    for i in range(4):
        txns[i].put(b"__meta__", pickle.dumps(metadata))  # store metadata
        txns[i].put(b"__len__", pickle.dumps(global_idx))  # store number of samples
        txns[i].commit()


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
        for pdb, pdb_path in metadata["pdb_paths"].items():
            # TODO: temp fix for wrong dir path. FIX
            _pdb_path = pdb_path.replace("/home/mila/t/tanc/scratch", "/network/scratch/t/tanc")
            metadata["pdb_paths"][pdb] = _pdb_path

    env.close()
    return metadata


def load_pdbs_and_topologies(
    pdb_paths: list[str],
    num_aa_range: list[int],
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

    # CHECK ATOM ORDERING

    # def standardize_atom_name(atom_name: str, aa_name: str) -> str:
    #     """
    #     Standardizes the atom name to a consistent format.

    #     Args:
    #         atom_name (str): The original atom name.

    #     Returns:
    #         str: The standardized atom name.
    #     """
    #     # TODO double check this with Leon
    #     # Standarize side-chain H atom encoding
    #     if atom_name[0] == "H" and atom_name[-1] in ("1", "2", "3"):
    #         # For these AA the H-X-N atoms are not interchangable
    #         if aa_name in ("HIS", "PHE", "TRP", "TYR") and atom_name[:2] in (
    #             "HE",
    #             "HD",
    #             "HZ",
    #             "HH",
    #         ):
    #             pass
    #         else:
    #             atom_name = atom_name[:-1]

    #     return atom_name

    # logging.info("Checking atom order consistency for each amino acid")
    # atom_order_dict = {}

    # for seq, pdb in pdb_dict.items():
    #     residues = list(pdb.topology.residues())
    #     for i, residue in enumerate(residues):

    #         residue_name = residue.name

    #         atom_names = [atom.name for atom in residue.atoms()]
    #         atom_names = [standardize_atom_name(atom_name, residue.name) for atom_name in atom_names]

    #         # Handle N-terminal and C-terminal residues separately
    #         if i == 0:
    #             residue_key = f"N{residue_name}"
    #         elif i == len(residues) - 1:
    #             residue_key = f"C{residue_name}"
    #         else:
    #             residue_key = residue_name

    #         if residue_key not in atom_order_dict:
    #             # Record the atom ordering for this residue
    #             atom_order_dict[residue_key] = atom_names
    #         else:
    #             # Check consistency with the recorded atom ordering
    #             assert atom_names == atom_order_dict[residue_key], (
    #             f"Inconsistent atom order in residue {residue} of sequence {seq}"
    #             f"Expected: {atom_order_dict[residue_key]}, Found: {atom_names}"
    #             )

    # CHECK ATOM BONDING

    # atom_order_dict = {}

    # for seq, pdb in pdb_dict.items():
    #     residues = list(pdb.topology.residues())
    #     for i, residue in enumerate(residues):

    #         residue_name = residue.name

    #         # Handle N-terminal and C-terminal residues separately
    #         if i == 0:
    #             residue_key = f"N{residue_name}"
    #         elif i == len(residues) - 1:
    #             residue_key = f"C{residue_name}"
    #         else:
    #             residue_key = residue_name

    #         if residue_key not in atom_order_dict:
    #             atom_order_dict[residue_key] = None

    #             print(residue_key)
    #             for bond in residue.bonds():
    #                 atom1 = bond.atom1
    #                 atom2 = bond.atom2

    #                 res1 = atom1.residue
    #                 res2 = atom2.residue

    #                 print(f"Bond: ({atom1.name} - {res1.index}, {atom2.name} - {res2.index})")

    #             breakpoint()

    return pdb_dict, topology_dict
