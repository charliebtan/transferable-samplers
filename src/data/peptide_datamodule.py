import logging
import math
import os
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
from src.data.components.encodings import get_atom_encoding
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.symmetry import resolve_chirality
from src.data.components.utils import get_adj_list, get_atom_types
from src.evaluation.metrics.evaluate_peptide_data import evaluate_peptide_data
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran
from src.evaluation.plots.plot_tica import plot_tica


class PeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        data_url: str,
        pdb_url: str,
        data_filename: str,
        pdb_filename: str,
        num_particles: int,
        num_dimensions: int,
        dim: int,
        tica_lagtime: int,
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
        assert dim == num_particles * num_dimensions

        self.data_path = f"{self.hparams.data_dir}/{self.hparams.data_filename}"
        self.pdb_path = f"{self.hparams.data_dir}/{self.hparams.pdb_filename}" if self.hparams.pdb_filename else None

        # Setup transforms
        transform_list = [Random3DRotationTransform(self.hparams.num_dimensions)]
        if self.hparams.com_augmentation:
            transform_list.append(
                CenterOfMassTransform(
                    self.hparams.num_dimensions,
                    1 / math.sqrt(self.hparams.num_particles),
                )
            )  # TODO optimal value?
        self.transforms = torchvision.transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        # Download data file
        if not os.path.exists(self.data_path):
            wget.download(self.hparams.data_url, self.data_path)

        # Download pdb file
        if self.pdb_path and not os.path.exists(self.pdb_path):
            wget.download(self.hparams.pdb_url, self.pdb_path)

    def setup_potential(self):
        if self.pdb_path is not None:
            self.topology = md.load_topology(self.pdb_path)
            self.pdb = openmm.app.PDBFile(self.pdb_path)
        else:
            self.topology = md.load(self.data_path).topology.to_openmm()

        # Different system configs for different datasets
        if self.hparams.num_particles == 42:
            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")
            nonbondedMethod = openmm.app.NoCutoff
            nonbondedCutoff = 0.9 * openmm.unit.nanometer
            temperature = 300
        else:
            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
            nonbondedMethod = openmm.app.CutoffNonPeriodic
            nonbondedCutoff = 2.0 * openmm.unit.nanometer
            temperature = 310

        # Initalize forcefield system
        system = forcefield.createSystem(
            self.pdb.topology,
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
        self.potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

    def setup_data(self):
        # Load the data
        if self.data_path[-3:] == "npy":
            data = np.load(self.data_path, allow_pickle=True)
        elif self.data_path[-2:] == "h5":
            data = md.load(self.data_path).xyz
        else:
            raise ValueError(f"Unknown file format for {self.data_path}")

        # Reshape and tensorize
        data = data.reshape(-1, self.hparams.dim)
        data = torch.tensor(data).float()

        if self.hparams.make_iid:
            rand_idx = torch.randperm(data.shape[0])
            data = data[rand_idx]

        # Zero center of mass
        data = self.zero_center_of_mass(data)

        # Split data
        train_data = data[: self.hparams.num_train_samples]
        test_data = data[self.hparams.num_train_samples :]

        # Compute std on only train data
        self.std = train_data.std()

        # Standardize the data
        train_data = self.normalize(train_data)
        test_data = self.normalize(test_data)

        # Split val and test data
        self.data_val_ordered, self.data_test_ordered = (
            test_data[: self.hparams.num_val_samples],
            test_data[self.hparams.num_val_samples :],
        )

        # Randomized ordering of val samples
        val_rng = np.random.default_rng(0)
        data_val = torch.tensor(val_rng.permutation(self.data_val_ordered))

        # Randomized ordering / subset of test samples
        test_rng = np.random.default_rng(1)
        data_test = torch.tensor(test_rng.permutation(self.data_test_ordered))[: self.hparams.num_test_samples]

        # Create training dataset with transforms applied
        self.data_train = PeptideDataset(train_data, transform=self.transforms, encodings=self.encoding)

        # I actually thought better to apply transforms to val and test data too
        self.data_val = PeptideDataset(data_val, transform=self.transforms, encodings=self.encoding)
        self.data_test = PeptideDataset(data_test, transform=self.transforms, encodings=self.encoding)

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
        else:
            self.batch_size_per_device = self.hparams.batch_size

        self.setup_potential()
        self.encoding = get_atom_encoding(self.topology)
        self.setup_data()
        self.atom_types = get_atom_types(self.topology)
        self.adj_list = get_adj_list(self.topology)

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

        if prefix.startswith("val"):
            true_traj_samples = self.as_pointcloud(self.unnormalize(self.data_val_ordered))
        elif prefix.startswith("test"):
            true_traj_samples = self.as_pointcloud(self.unnormalize(self.data_test_ordered))

        plot_ramachandran(
            log_image_fn, true_data.samples[: self.hparams.num_eval_samples], self.topology, prefix=prefix + "true"
        )

        plot_tica(
            log_image_fn,
            true_traj_samples,
            true_data.samples[: self.hparams.num_eval_samples],
            self.topology,
            self.hparams.tica_lagtime,
            prefix=prefix + "true",
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

            symmetry_metrics, symmetry_change = resolve_chirality(
                true_data.samples, data.samples, self.adj_list, self.atom_types, prefix + name
            )
            data = data[~symmetry_change]
            metrics.update(symmetry_metrics)

            if len(data) == 0:
                logging.warning(f"No {name} samples left after symmetry correction.")
            else:
                metrics.update(
                    evaluate_peptide_data(true_data, data, self.topology, self.hparams.num_eval_samples, prefix + name)
                )
                plot_ramachandran(log_image_fn, data.samples, self.topology, prefix=prefix + name)
                plot_tica(
                    log_image_fn,
                    true_traj_samples,
                    data.samples,
                    self.topology,
                    self.hparams.tica_lagtime,
                    prefix=prefix + name,
                )

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
