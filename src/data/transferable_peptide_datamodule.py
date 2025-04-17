import logging
import os
from typing import Any, Callable, Optional

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
from bgflow import OpenMMBridge, OpenMMEnergy

from src.data.base_datamodule import BaseDataModule
from src.data.components.data_types import SamplesData
from src.data.components.encodings import AA_CODE_CONVERSION, get_atom_encoding
from src.data.components.symmetry import resolve_chirality
from src.data.components.utils import get_adj_list, get_atom_types
from src.evaluation.metrics.evaluate_peptide_data import evaluate_peptide_data

MEAN_MIN_DIST_DICT = {
    2: 0.4658  # can be saved in dict after computing in setup_data
}
MEAN_ATOMS_PER_AA = 17.67


class TransferablePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
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
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        raise NotImplementedError("prepare_data is not implemented. Please implement this method in the subclass.")

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

    def standardize_tensor_dict(self, tensor_dict):
        for key, data in tensor_dict.items():
            tensor_dict[key] = self.normalize(self.zero_center_of_mass(data))
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
        raise NotImplementedError("setup_data is not implemented. Please implement this method in the subclass.")

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
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        nonbondedMethod = openmm.app.CutoffNonPeriodic
        nonbondedCutoff = 2.0 * openmm.unit.nanometer
        temperature = 310

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
