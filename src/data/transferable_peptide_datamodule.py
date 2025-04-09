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
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.data_types import SamplesData
from src.data.components.encodings import AA_CODE_CONVERSION, get_atom_encoding
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.rotation import Random3DRotationTransform
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran


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
        dim: int,  # dim of largest system
        make_iid: bool = False,
        com_augmentation: bool = False,
        # TODO maybe make this all just *args?
        num_train_samples: int = 100_000,  # First N trajectory samples
        num_val_samples: int = 20_000,  # Following M trajectory samples
        num_test_samples: int = 100_000,  # Random K trajectory samples from the rest
        num_test_samples_small: int = 10_000,  # Smaller subset for plotting etc
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_eval_samples: int = 10_000,
        energy_hist_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

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
                    1 / math.sqrt(10 * self.hparams.num_aa),  # TODO check this value
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
        self.val_sequences = []

        for filelist, pdb_path in zip([train_pdb_files, val_pdb_files], [self.train_pdb_path, self.val_pdb_path]):
            for filename in filelist:
                if not filename.endswith(".pdb"):
                    logging.info(f"Skipping non-PDB file: {filename}")
                    continue

                filepath = os.path.join(pdb_path, filename)
                pdb = openmm.app.PDBFile(filepath)

                assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"

                name = "".join([AA_CODE_CONVERSION[aa.name] for aa in pdb.topology.residues()])

                if pdb_path == self.val_pdb_path:
                    self.val_sequences.append(name)

                self.pdb_dict[name] = pdb
                self.topology_dict[name] = md.load_topology(filepath)

    def setup_atom_encoding(self):
        self.encoding_dict = {}
        for name, topology in self.topology_dict.items():
            self.encoding_dict[name] = get_atom_encoding(topology)

    def create_mask(self, x):
        assert len(x.shape) == 2
        true_mask = torch.ones((x.shape[0], x.shape[1]))
        false_mask = torch.zeros((x.shape[0], self.max_num_particles * self.hparams.num_dimensions - x.shape[1]))
        return torch.cat([true_mask, false_mask], dim=1)

    def pad_data(self, x):
        assert len(x.shape) == 2
        pad_tensor = torch.zeros((x.shape[0], self.max_num_particles * self.hparams.num_dimensions - x.shape[1]))
        return torch.cat([x, pad_tensor], dim=1)

    def pad_encoding(self, encoding):
        for key, value in encoding.items():
            encoding[key] = torch.cat([value, torch.zeros(self.max_num_particles - value.shape[0])])
        return encoding

    def load_data_as_tensor_dict(self, path):
        data = np.load(path, allow_pickle=True).item()

        tensor_dict = {}

        max_num_particles = 0

        # Load + center + tensorize data
        for key, data in data.items():
            num_samples = data.shape[0]
            num_particles = data.shape[1] // self.hparams.num_dimensions
            max_num_particles = max(max_num_particles, num_particles)
            assert not data.shape[1] // num_samples
            data = torch.tensor(data).float()

            tensor_dict[key] = data

        return tensor_dict, max_num_particles

    def preprocess_data(self, data_dict):
        # Process data + add encoding and mask
        for key, data in data_dict.items():
            data = self.zero_center_of_mass(data)
            data = self.normalize(data)
            mask = self.create_mask(data)
            data = self.pad_data(data)
            data_dict[key] = data
            encoding = self.pad_encoding(self.encoding_dict[key])
            data_dict[key] = {
                "x": data,
                "encoding": encoding,
                "mask": mask,
            }

        # Convert into individual data samples
        data_list = []
        for key, data in data_dict.items():
            for i in range(data["x"].shape[0]):  # Iterate over each batch item
                data_list.append(
                    {
                        "x": data["x"][i],
                        "encoding": data["encoding"],
                        "mask": data["mask"][i],
                    }
                )

        return data_list

    def setup_data(self):
        train_data_dict, self.max_num_particles = self.load_data_as_tensor_dict(self.train_data_path)
        val_data_dict, _ = self.load_data_as_tensor_dict(self.val_data_path)

        weighted_vars = [x.var() * x.shape[0] for x in train_data_dict.values()]
        self.std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / len(weighted_vars))

        train_data_list = self.preprocess_data(train_data_dict)
        val_data_list = self.preprocess_data(val_data_dict)

        self.data_train = PeptideDataset(train_data_list, transform=self.transforms)
        self.data_val = PeptideDataset(val_data_list, transform=None)

        self.val_data_dict = val_data_dict  # stored for eval

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
        # self.setup_atom_types()
        # self.setup_adj_list()

    def setup_potential(self, val_sequence: str):
        # TODO!! CHECK THIS
        # MAJDI DO NOT LET ME MERGE THIS WITHOUT CHECKING LOL

        forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")
        nonbondedMethod = openmm.app.NoCutoff
        nonbondedCutoff = 0.9 * openmm.unit.nanometer
        temperature = 300

        # Initalize forcefield system
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

        return true_data, potential

    def metrics_and_plots(
        self,
        log_dict_fn: Callable,
        log_image_fn: Callable,
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

        plot_ramachandran(
            log_image_fn, true_data.samples[: self.hparams.num_eval_samples], self.topology, prefix=prefix + "true"
        )

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

            symmetry_metrics, symmetry_change = self.resolve_chirality(true_data.samples, data.samples, prefix + name)
            data = data[~symmetry_change]
            metrics.update(symmetry_metrics)

            if len(data) == 0:
                logging.warning(f"No {name} samples left after symmetry correction.")
            else:
                metrics.update(self.compute_all_metrics(true_data, data, prefix + name))
                plot_ramachandran(log_image_fn, data.samples, self.topology, prefix=prefix + name)

        logging.info("Plotting energies")
        plot_energies(
            log_image_fn,
            true_data.energy[: self.hparams.num_eval_samples],
            proposal_data.energy if len(proposal_data) > 0 else None,
            resampled_data.energy if len(resampled_data) > 0 else None,
            smc_data.energy if (smc_data is not None and len(smc_data) > 0) else None,
            **self.hparams.energy_hist_config,
            prefix=prefix,
        )

        logging.info("Plotting interatomic distances")
        plot_atom_distances(
            log_image_fn,
            true_data.samples[: self.hparams.num_eval_samples],
            proposal_data.samples if len(proposal_data) > 0 else None,
            resampled_data.samples if len(resampled_data) > 0 else None,
            smc_data.samples if (smc_data is not None and len(smc_data) > 0) else None,
            prefix=prefix,
        )

        return metrics
