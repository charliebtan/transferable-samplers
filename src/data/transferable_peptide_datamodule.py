import logging
import math
import os
import zipfile
from typing import Any, Callable, Optional

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
import torchvision
import wget
from bgflow import OpenMMBridge, OpenMMEnergy

from src.data.base_datamodule import BaseDataModule
from src.data.components.atom_noise import AtomNoiseTransform
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.data_types import SamplesData
from src.data.components.encodings import AA_CODE_CONVERSION, get_atom_encoding
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.symmetry import resolve_chirality
from src.data.components.utils import get_adj_list, get_atom_types
from src.evaluation.metrics.evaluate_peptide_data import evaluate_peptide_data

MEAN_MIN_DIST_DICT = {
    2: 0.4658  # can be saved in dict after computing in setup_data
}


class TransferablePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        train_data_url: str,
        val_data_url: str,
        train_data_filename: str,
        val_data_filename: str,
        train_pdb_zip_url: str,  # expects a dir of pdbs
        val_pdb_zip_url: str,  # expects a dir of pdbs
        num_aa: int,
        num_dimensions: int,
        num_particles: int,
        dim: int,  # dim of largest system
        make_iid: bool = False,
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
        super().__init__()

        assert dim == num_dimensions * num_particles, "dim must be equal to num_dimensions * num_particles"

        self.train_data_path = f"{self.hparams.data_dir}/{self.hparams.train_data_filename}"
        self.val_data_path = f"{self.hparams.data_dir}/{self.hparams.val_data_filename}"

        self.train_pdb_zip_path = f"{self.hparams.data_dir}/pdb_train.zip"
        self.val_pdb_zip_path = f"{self.hparams.data_dir}/pdb_val.zip"

        self.train_pdb_path = f"{self.hparams.data_dir}/pdb_train"
        self.val_pdb_path = f"{self.hparams.data_dir}/pdb_val"

        # Setup transforms
        transform_list = [Random3DRotationTransform(self.hparams.num_dimensions)]
        if self.hparams.com_augmentation:
            transform_list.append(
                CenterOfMassTransform(
                    self.hparams.num_dimensions,
                    1 / math.sqrt(17.67 * self.hparams.num_aa),
                )
            )
        if self.hparams.atom_noise_augmentation_factor:
            transform_list.append(
                AtomNoiseTransform(
                    self.hparams.atom_noise_augmentation_factor * MEAN_MIN_DIST_DICT[self.hparams.num_aa],
                )
            )
        self.transforms = torchvision.transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        # Download data files
        if not os.path.exists(self.train_data_path):
            wget.download(self.hparams.train_data_url, self.train_data_path)

        if not os.path.exists(self.val_data_path):
            wget.download(self.hparams.val_data_url, self.val_data_path)

        # Download + extract pdb files
        if not os.path.exists(self.train_pdb_zip_path):
            wget.download(self.hparams.train_pdb_zip_url, self.train_pdb_zip_path)
            os.makedirs(self.train_pdb_path, exist_ok=False)
            with zipfile.ZipFile(self.train_pdb_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.train_pdb_path)

        if not os.path.exists(self.val_pdb_zip_path):
            wget.download(self.hparams.val_pdb_zip_url, self.val_pdb_zip_path)
            os.makedirs(self.val_pdb_path, exist_ok=False)
            with zipfile.ZipFile(self.val_pdb_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.val_pdb_path)

    def setup_topolgy(self):
        train_pdb_files = os.listdir(self.train_pdb_path)
        val_pdb_files = os.listdir(self.val_pdb_path)

        self.pdb_dict = {}
        self.topology_dict = {}

        self.train_sequences = []
        self.val_sequences = []

        for filelist, pdb_path in zip([train_pdb_files, val_pdb_files], [self.train_pdb_path, self.val_pdb_path]):
            for filename in filelist:
                if not filename.endswith(".pdb"):
                    logging.info(f"Skipping non-PDB file: {filename}")
                    continue

                filepath = os.path.join(pdb_path, filename)
                pdb = openmm.app.PDBFile(filepath)

                assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"
                assert len(list(pdb.topology.residues())) == self.hparams.num_aa, (
                    "PDB does not match the number of amino acids"
                )

                name = "".join([AA_CODE_CONVERSION[aa.name] for aa in pdb.topology.residues()])

                if pdb_path == self.train_pdb_path:
                    self.train_sequences.append(name)
                else:
                    self.val_sequences.append(name)

                self.pdb_dict[name] = pdb
                self.topology_dict[name] = md.load_topology(filepath)

    def setup_atom_encoding(self):
        self.encoding_dict = {}
        for name, topology in self.topology_dict.items():
            self.encoding_dict[name] = get_atom_encoding(topology)

    def pad_data(self, x):
        assert len(x.shape) == 2
        pad_tensor = torch.zeros((x.shape[0], self.hparams.num_particles * self.hparams.num_dimensions - x.shape[1]))
        return torch.cat([x, pad_tensor], dim=1)

    def pad_encoding(self, encoding):
        for key, value in encoding.items():
            encoding[key] = torch.cat(
                [value, torch.zeros(self.hparams.num_particles - value.shape[0], dtype=torch.int64)]
            )
        return encoding

    def create_mask(self, x):
        assert len(x.shape) == 1
        num_particles = x.shape[0] // self.hparams.num_dimensions
        true_mask = torch.ones(num_particles)
        false_mask = torch.zeros(self.hparams.num_particles - num_particles)
        return torch.cat([true_mask, false_mask]).bool()

    def load_data_as_tensor_dict(self, path):
        data = np.load(path, allow_pickle=True).item()

        tensor_dict = {}

        max_num_particles = 0

        # Load + center + tensorize data
        i = 0
        for key, data in data.items():
            num_samples = data.shape[0]
            num_particles = data.shape[1] // self.hparams.num_dimensions
            max_num_particles = max(max_num_particles, num_particles)
            assert not data.shape[1] // num_samples
            data = torch.tensor(data).float()
            data = self.zero_center_of_mass(data)

            rng = np.random.default_rng(seed=i)
            data = torch.tensor(rng.permutation(data))[: self.hparams.num_samples_per_seq]  # TODO - need to copy Leon

            tensor_dict[key] = data

        return tensor_dict, max_num_particles

    def normalize_tensor_dict(self, tensor_dict):
        # TODO check the normalization
        for key, data in tensor_dict.items():
            tensor_dict[key] = self.normalize(data)
        return tensor_dict

    def add_encodings_to_tensor_dict(self, tensor_dict):
        for key, data in tensor_dict.items():
            encoding = self.encoding_dict[key]
            tensor_dict[key] = {
                "x": data,
                "encoding": encoding,
            }
        return tensor_dict

    def pad_and_mask_tensor_dict(self, tensor_dict):
        for key, data in tensor_dict.items():
            x = self.pad_data(data["x"])
            encoding = self.pad_encoding(data["encoding"])

            mask = self.create_mask(data["x"][0])

            tensor_dict[key] = {
                "x": x,
                "encoding": encoding,
                "mask": mask,
            }

        return tensor_dict

    def tensor_dict_to_samples_list(self, tensor_dict):
        data_list = []
        for data in tensor_dict.values():
            for i in range(data["x"].shape[0]):  # Iterate over each batch item
                data_list.append(
                    {
                        **data,
                        "x": data["x"][i],
                    }
                )
        return data_list

    def setup_data(self):
        train_data_dict, self.max_num_particles = self.load_data_as_tensor_dict(self.train_data_path)
        val_data_dict, _ = self.load_data_as_tensor_dict(self.val_data_path)

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

        weighted_vars = [x.var(unbiased=False) * x.shape[0] for x in train_data_dict.values()]
        total_samples = sum(x.shape[0] for x in train_data_dict.values())
        self.std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_samples)

        train_data_dict = self.normalize_tensor_dict(train_data_dict)
        val_data_dict = self.normalize_tensor_dict(val_data_dict)

        # Compute mean min dist across all pairs of atoms if not
        # found in the MEAN_MIN_DIST_DICT
        if (
            self.hparams.atom_noise_augmentation_factor and 
            MEAN_MIN_DIST_DICT.get(self.hparams.num_aa) is not None
        ):
            from src.evaluation.plots.plot_atom_distances import interatomic_dist
            mean_min_dists = []
            for key, data in train_data_dict.items():
                num_samples = data.shape[0]
                data = data.reshape(num_samples, -1, self.hparams.num_dimensions)
                dists = interatomic_dist(data, flatten=False)
                mean_min_dist = dists.min(dim=1)[0].mean()
                mean_min_dists.append(mean_min_dist)

            # this is effectively just the length of a carbon-hydrogen bond
            mean_min_dist = torch.mean(torch.tensor(mean_min_dists))
            logging.info(
                f"MEAN_MIN_DIST={mean_min_dist.item()} for peptide of length {self.hparams.num_aa}. "
                "Save it in MEAN_MIN_DIST_DICT to save on computation"
            )

        # slice out subset of val_data_dict
        val_data_dict = {
            key: val for i, (key, val) in enumerate(val_data_dict.items()) if i < self.hparams.num_val_sequences
        }
        self.val_sequences = list(val_data_dict.keys())

        train_data_dict = self.add_encodings_to_tensor_dict(train_data_dict)
        val_data_dict = self.add_encodings_to_tensor_dict(val_data_dict)

        train_data_dict = self.pad_and_mask_tensor_dict(train_data_dict)

        train_data_list = self.tensor_dict_to_samples_list(train_data_dict)
        val_data_list = self.tensor_dict_to_samples_list(val_data_dict)

        self.data_train = PeptideDataset(train_data_list, transform=self.transforms)
        self.data_val = PeptideDataset(val_data_list, transform=self.transforms)

        self.val_data_dict = val_data_dict

    def setup_atom_types(self):
        self.atom_types_dict = {}
        for name, topology in self.topology_dict.items():
            atom_types = get_atom_types(topology)
            self.atom_types_dict[name] = atom_types

    def setup_adj_list(self):
        self.adj_list_dict = {}
        for name, topology in self.topology_dict.items():
            adj_list = get_adj_list(topology)
            self.adj_list_dict[name] = adj_list

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

        self.setup_topolgy()
        self.setup_atom_encoding()
        self.setup_data()
        self.setup_atom_types()
        self.setup_adj_list()

    def setup_potential(self, val_sequence: str):
        # TODO!! CHECK THIS
        # MAJDI DO NOT LET ME MERGE THIS WITHOUT CHECKING LOL

        forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")
        nonbondedMethod = openmm.app.NoCutoff
        nonbondedCutoff = 0.9 * openmm.unit.nanometer
        temperature = 300

        # Initalize forcefield systemq
        system = forcefield.createSystem(
            self.pdb_dict[val_sequence].topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=None,
        )

        # Initialize integrator
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )

        # Initialize potential
        potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

        return potential

    def prepare_eval(self, val_sequence: str):
        true_data = self.val_data_dict[val_sequence]
        potential = self.setup_potential(val_sequence)
        energy_fn = lambda x: potential.energy(x).flatten()
        return true_data, energy_fn

    def metrics_and_plots(
        self,
        log_image_fn: Callable,
        sequence: str,
        true_data: SamplesData,
        proposal_data: SamplesData,
        resampled_data: SamplesData,
        smc_data: Optional[SamplesData] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics and plots at the end of an epoch."""

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        metrics = {}

        # plot_ramachandran(
        #     log_image_fn,
        #     true_data.samples[: self.hparams.num_eval_samples],
        #     self.topology_dict[sequence],
        #     prefix=prefix + "true",
        # )

        for data, name in [
            [proposal_data, "proposal"],
            [resampled_data, "resampled"],
            [smc_data, "smc"],
        ]:
            if data is None and name == "smc":
                continue

            if len(data) == 0:
                logging.warning(f"No {name} samples present.")

            data = data[: self.hparams.num_eval_samples * 2]  # slice out extra samples for those lost to symmetry

            symmetry_metrics, symmetry_change = resolve_chirality(
                true_data.samples,
                data.samples,
                self.adj_list_dict[sequence],
                self.atom_types_dict[sequence],
                prefix + name,
            )
            data = data[~symmetry_change]
            metrics.update(symmetry_metrics)

            if len(data) == 0:
                logging.warning(f"No {name} samples left after symmetry correction.")
            else:
                metrics.update(
                    evaluate_peptide_data(
                        true_data,
                        data,
                        topology=self.topology_dict[sequence],
                        num_eval_samples=self.hparams.num_eval_samples,
                        prefix=prefix + name,
                        compute_distribution_distances=False,
                    )
                )
                # plot_ramachandran(log_image_fn, data.samples, self.topology_dict[sequence], prefix=prefix + name)

        # logging.info("Plotting energies")
        # plot_energies(
        #     log_image_fn,
        #     true_data.energy[: self.hparams.num_eval_samples],
        #     proposal_data.energy if len(proposal_data) > 0 else None,
        #     resampled_data.energy if len(resampled_data) > 0 else None,
        #     smc_data.energy if (smc_data is not None and len(smc_data) > 0) else None,
        #     **self.hparams.energy_hist_config,
        #     prefix=prefix,
        # )

        # logging.info("Plotting interatomic distances")
        # plot_atom_distances(
        #     log_image_fn,
        #     true_data.samples[: self.hparams.num_eval_samples],
        #     proposal_data.samples if len(proposal_data) > 0 else None,
        #     resampled_data.samples if len(resampled_data) > 0 else None,
        #     smc_data.samples if (smc_data is not None and len(smc_data) > 0) else None,
        #     prefix=prefix,
        # )

        return metrics
