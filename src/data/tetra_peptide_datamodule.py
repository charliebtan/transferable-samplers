import copy
import logging
import math
import os
from typing import Any, Optional
import numpy as np
from tqdm import tqdm

import pickle
import lmdb

import huggingface_hub
import torch
import torchvision

from src.data.components.atom_noise import AtomNoiseTransform
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.rotation import Random3DRotationTransform
from src.data.transferable_peptide_datamodule import TransferablePeptideDataModule
from src.evaluation.plots.plot_atom_distances import interatomic_dist

MEAN_MIN_DIST_DICT = {
    2: 0.4658  # can be saved in dict after computing in setup_data
}
MEAN_ATOMS_PER_AA = 17.67


class TetraPeptideDataModule(TransferablePeptideDataModule):
    def __init__(
        self,
        data_dir: str,
        huggingface_repo_id: str,
        huggingface_train_data_dir: str,
        huggingface_val_data_dir: str,
        num_aa: int,
        num_dimensions: int,
        num_particles: int,
        dim: int,  # dim of largest system
        com_augmentation: bool = False,
        atom_noise_augmentation_factor: float = 0.0,
        # TODO maybe make this all just *args?
        num_samples_per_seq: int = 10_000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_eval_samples: int = 10_000,
        num_val_sequences: int = 20,
        energy_hist_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            num_aa=num_aa,
            num_dimensions=num_dimensions,
            num_particles=num_particles,
            dim=dim,
            com_augmentation=com_augmentation,
            atom_noise_augmentation_factor=atom_noise_augmentation_factor,
            num_samples_per_seq=num_samples_per_seq,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            num_eval_samples=num_eval_samples,
            num_val_sequences=num_val_sequences,
            energy_hist_config=energy_hist_config,
        )

        assert dim == num_dimensions * num_particles, "dim must be equal to num_dimensions * num_particles"

        self.train_data_path = f"{self.hparams.data_dir}/train"
        self.val_data_path = f"{self.hparams.data_dir}/val"


    def download_data(self, huggingface_data_dir: str, local_dir: str) -> None:


        # Download the training data
        huggingface_hub.snapshot_download(
            repo_id=self.hparams.huggingface_repo_id,
            repo_type="dataset",
            allow_patterns=f"{huggingface_data_dir}/*",
            local_dir=local_dir,
            max_workers=4,
        )

    def check_data(self, data_dir):

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

    def build_lmdb(self, npz_paths, lmdb_path, map_size = 1 << 40, batch_size = 100):
        """
        Builds a LMDB file from the data in the given path.
        """

        if os.path.exists(lmdb_path):
            logging.warning(f"LMDB file {lmdb_path} already exists, skipping LMDB creation")
            return

        env = lmdb.open(lmdb_path, map_size=map_size)

        txn = env.begin(write=True)
        txn.put(b'__len__', pickle.dumps(len(npz_paths)))

        for idx, path in enumerate(tqdm(npz_paths)):
            seq_id = os.path.basename(path).split("-")[0]

            with np.load(path, allow_pickle=False) as data:
                positions_array = data["positions"].copy()  # force load, then release file handle

            sample = {
                "seq_id": seq_id, # TODO change to seq_name - requires rerunning
                "positions": positions_array  # still a NumPy array, LMDB-friendly
            }

            key = f'{idx:08}'.encode()
            value = pickle.dumps(sample)
            txn.put(key, value)

            # 🔁 Commit and open new transaction every `batch_size` samples
            if (idx + 1) % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)

        # Final commit
        txn.commit()
        env.sync()
        env.close() 
        
    def load_lmdb_as_dict(lmdb_path):
        """
        Load an LMDB where each entry is a dict with keys 'seq_id' and 'positions'.
        Returns:
            data_dict: {seq_id: positions_array}
        """
        data_dict = {}

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            length_bytes = txn.get(b'__len__')
            if length_bytes is None:
                raise ValueError(f"Missing '__len__' key in LMDB at {lmdb_path}")
            length = pickle.loads(length_bytes)

            for idx in tqdm(range(length), desc="Loading LMDB"):
                key = f'{idx:08}'.encode()
                sample = pickle.loads(txn.get(key))

                seq_id = sample["seq_id"]
                positions = sample["positions"]
                data_dict[seq_id] = positions

        env.close()
        return data_dict

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        self.download_data(
            huggingface_data_dir=self.hparams.huggingface_train_data_dir,
            local_dir=self.train_data_path,
        )
        self.download_data(
            huggingface_data_dir=self.hparams.huggingface_val_data_dir,
            local_dir=self.val_data_path,
        )

        train_npz_paths, train_pdb_paths = self.check_data(self.train_data_path + "/4AA-large/train")
        val_npz_paths, val_pdb_paths = self.check_data(self.val_data_path + "/4AA-large/val")

        if not os.path.exists(self.train_data_path + "/4AA-large/train/seq.lmdb"):
            self.build_seq_lmdb(train_npz_paths, lmdb_path = self.train_data_path + "/seq.lmdb")
        else:
            logging.info(f"LMDB file {self.train_data_path}/seq.lmdb already exists, skipping LMDB creation")

        if not os.path.exists(self.val_data_path + "/4AA-large/val/seq.lmdb"):
            self.build_seq_lmdb(val_npz_paths, lmdb_path = self.val_data_path + "/seq.lmdb")
        else:
            logging.info(f"LMDB file {self.val_data_path}/seq.lmdb already exists, skipping LMDB creation")

        # pdb_dict = {}
        # topology_dict = {}
        # logging.info("Loading .pdb files")
        # for path in tqdm(pdb_files):
        #     seq = path.split("-")[0]

        #     pdb = openmm.app.PDBFile(path + "/" + path)
        #     topology = md.load_topology(path + "/" + path)

        #     assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"
        #     assert len(list(pdb.topology.residues())) == self.hparams.num_aa, (
        #         "PDB does not match the number of amino acids"
        #     )

        #     pdb_dict[seq] = pdb
        #     topology_dict[seq] = topology

        # return data_dict, pdb_dict, topology_dict

    def scan_lmdb(self, lmdb_path):

        max_num_particles = 0

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            length_bytes = txn.get(b'__len__')
            if length_bytes is None:
                raise ValueError(f"Missing '__len__' key in LMDB at {lmdb_path}")
            length = pickle.loads(length_bytes)

            for idx in tqdm(range(length), desc="Scanning LMDB"):
                key = f'{idx:08}'.encode()
                sample_bytes = txn.get(key)
                if sample_bytes is None:
                    raise KeyError(f"Missing entry at index {idx}")
                sample = pickle.loads(sample_bytes)

                seq_id = sample["seq_id"]
                positions = sample["positions"]

                num_samples, num_particles = positions.shape[0], positions.shape[1]

                logging.info(f"Found {num_samples} samples for sequence {seq_id}")

                max_num_particles = max(max_num_particles, num_particles)

        env.close()

    def compute_std(self, data):

        total_samples = 0
        weighted_vars = []
        for key, data in data.items():
            total_samples += data.shape[0]
            data = self.zero_center_of_mass(data)
            weighted_var = data.var(unbiased=False) * data.shape[0] 
            weighted_vars.append(weighted_var)
        std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_samples)

        return std

    def setup_data(self):

        self.scan_lmdb(self.train_data_path + "/seq.lmdb")
        self.scan_lmdb(self.val_data_path + "/seq.lmdb")
        breakpoint()


        logging.info("Loading train data")
        train_data_dict, self.max_num_particles = self.load_data(self.train_data_path + "/4AA-large/train")
        logging.info("Loading val data")
        val_data_dict, _ = self.load_data(self.val_data_path + "/4AA-large/val")

        new_val_data_dict = {}
        for key in val_data_dict.keys():
            if key in self.train_sequences:
                logging.info(f"Key {key} found in train_sequences, removing from val_sequences")
            else:
                assert key in self.val_sequences, f"Key {key} not found in val_sequences"
                new_val_data_dict[key] = val_data_dict[key]
        val_data_dict = new_val_data_dict

        for key in train_data_dict.keys():
            assert key in self.train_sequences, f"Key {key} not found in train_sequences"
            assert key not in self.val_sequences, f"Key {key} found in val_sequences"

        for key in self.val_sequences:
            assert key in val_data_dict, f"Key {key} not found in val_data_dict"
            assert key not in train_data_dict, f"Key {key} found in train_data_dict"

        for key in self.train_sequences:
            assert key in train_data_dict, f"Key {key} not found in train_data_dict"
            assert key not in val_data_dict, f"Key {key} found in val_data_dict"

        common_keys = set(train_data_dict.keys()).intersection(set(val_data_dict.keys()))
        logging.info(f"Common keys between train and val data dict: {common_keys}")

        # Need to check here so TarFlow is correctly initalized for data
        assert self.hparams.num_particles > self.max_num_particles, (
            "TarFlow num_particles must be greater than the largest system size. "
            + f"Max num particles={self.max_num_particles}. Set num_particles in data config "
            + f"to {self.max_num_particles + 1}."
        )

        # TarFlow dim is the dim of the largest system + a single padded token
        assert self.hparams.dim > self.max_num_particles * self.hparams.num_dimensions, (
            "TarFlow dim must be greater than the largest system size + a single padded token. "
            + f"Set dim in data config to {(self.max_num_particles + 1) * self.hparams.num_dimensions}."
        )

        self.std = self.compute_std(train_data_dict)

        train_data_dict = self.standardize_tensor_dict(train_data_dict)
        val_data_dict = self.standardize_tensor_dict(val_data_dict)

        # Setup transforms
        transform_list = [Random3DRotationTransform(self.hparams.num_dimensions)]
        if self.hparams.com_augmentation:
            transform_list.append(
                CenterOfMassTransform(
                    self.hparams.num_dimensions,
                    1 / math.sqrt(MEAN_ATOMS_PER_AA * self.hparams.num_aa),
                )
            )
        if self.hparams.atom_noise_augmentation_factor:
            if self.hparams.num_aa not in MEAN_MIN_DIST_DICT:
                mean_min_dists = []
                for key, data in train_data_dict.items():
                    num_samples = data.shape[0]
                    data = data.reshape(num_samples, -1, self.hparams.num_dimensions)
                    dists = interatomic_dist(data, flatten=False)
                    mean_min_dist = dists.min(dim=1)[0].mean()
                    mean_min_dists.append(mean_min_dist)

                # this is effectively just the length of a carbon-hydrogen bond
                mean_min_dist = torch.mean(torch.tensor(mean_min_dists))
                logging.warning(
                    f"MEAN_MIN_DIST={mean_min_dist.item()} for peptide of length {self.hparams.num_aa}. "
                    "Save it in MEAN_MIN_DIST_DICT to save on computation"
                )
            else:
                mean_min_dist = MEAN_MIN_DIST_DICT[self.hparams.num_aa]

            transform_list.append(
                AtomNoiseTransform(
                    self.hparams.atom_noise_augmentation_factor * mean_min_dist,
                )
            )
        transforms = torchvision.transforms.Compose(transform_list)

        # slice out subset of val_data_dict
        val_data_dict = {
            key: val for i, (key, val) in enumerate(val_data_dict.items()) if i < self.hparams.num_val_sequences
        }
        self.val_sequences = list(val_data_dict.keys())

        train_data_dict = self.add_encodings_to_tensor_dict(train_data_dict)
        val_data_dict = self.add_encodings_to_tensor_dict(val_data_dict)

        self.val_data_dict = copy.deepcopy(val_data_dict)  # without padding for sampling eval

        train_data_dict = self.pad_and_mask_tensor_dict(train_data_dict)
        val_data_dict = self.pad_and_mask_tensor_dict(val_data_dict)  # with padding for loss eval

        train_data_list = self.tensor_dict_to_samples_list(train_data_dict)
        val_data_list = self.tensor_dict_to_samples_list(val_data_dict)

        self.data_train = PeptideDataset(train_data_list, transform=transforms)
        self.data_val = PeptideDataset(val_data_list, transform=transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    "the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # self.setup_topolgy()
        # self.setup_atom_encoding()
        self.setup_data()
        # self.setup_atom_types()
        # self.setup_adj_list()
