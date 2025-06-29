import logging
import math
import os
from typing import Callable, Optional
import pickle

import openmm
import openmm.app
import torch
import torchvision

from src.data.base_datamodule import BaseDataModule
from src.data.components.data_types import SamplesData
from src.data.components.encoding import get_encoding_dict
from src.data.components.permutations import get_permutations_dict
from src.data.components.residue_tokenization import get_residue_tokenization_dict
from src.data.components.openmm import OpenMMBridge, OpenMMEnergy
from src.data.components.peptide_dataset import PeptideDataset
from src.data.components.prepare_data import (
    build_lmdb,
    check_and_get_files,
    cross_reference_files,
    load_lmdb_metadata,
    load_pdbs_and_topologies,
    prepare_tica_models,
)
from src.data.components.symmetry import resolve_chirality
from src.data.components.test_subset import ALL_TEST_SUBSET, SCALING_SUBSET, TEST_SUBSET_DICT
from src.data.components.transforms.add_encoding import AddEncodingTransform
from src.data.components.transforms.atom_noise import AtomNoiseTransform
from src.data.components.transforms.add_residue_tokenization import AddResidueTokenizationTransform
from src.data.components.transforms.center_of_mass import CenterOfMassTransform
from src.data.components.transforms.padding import PaddingTransform
from src.data.components.transforms.rotation import Random3DRotationTransform
from src.data.components.transforms.standardize import StandardizeTransform
from src.data.components.transforms.add_permutations import AddPermutationsTransform
from src.data.components.validation_subset import ALL_VALIDATION_SUBSET, VALIDATION_SUBSET_DICT
from src.evaluation.metrics.evaluate_peptide_data import evaluate_peptide_data
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_com_norms import plot_com_norms
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran
from src.evaluation.plots.plot_tica import plot_tica


class TransferablePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        train_lmdb_prefix: str,
        val_lmdb_prefix: str,
        test_lmdb_prefix: str,
        num_aa_max: int,
        num_aa_min: int,
        num_dimensions: int,
        num_particles: int,
        dim: int,  # dim of largest system
        com_augmentation: bool = False,
        atom_noise_augmentation_factor: float = 0.0,
        # TODO maybe make this all just *args?
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_eval_samples: int = 10_000,
        num_val_sequences: int = 20,
        resume_build_lmdb: bool = False,
        do_plots: bool = True,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        assert dim == num_dimensions * num_particles, "dim must be equal to num_dimensions * num_particles"
        assert num_workers < 2, "need a copy of lmdb for each worker, use at most 1 worker"

        self.train_data_path = f"{data_dir}/train"
        self.val_data_path = f"{data_dir}/val"
        self.test_data_path = f"{data_dir}/test"

        self.train_lmdb_prefix_path = f"{data_dir}/{train_lmdb_prefix}"
        self.val_lmdb_prefix_path = f"{data_dir}/{val_lmdb_prefix}"
        self.test_lmdb_prefix_path = f"{data_dir}/{test_lmdb_prefix}"

        self.tica_models_path = f"{data_dir}/tica_models"

        self.num_aa_range = list(range(num_aa_min, num_aa_max + 1))

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        if not (
            all([os.path.exists(f"{self.train_lmdb_prefix_path}_{i}.lmdb") for i in range(self.trainer.world_size)])
            and all(os.path.exists(f"{self.val_lmdb_prefix_path}_{i}.lmdb") for i in range(self.trainer.world_size))
            and all(os.path.exists(f"{self.test_lmdb_prefix_path}_{i}.lmdb") for i in range(self.trainer.world_size))
        ):
            logging.info("LMDB files already exist, skipping build.")

            train_npz_paths, train_pdb_paths = check_and_get_files(self.train_data_path)
            val_npz_paths, val_pdb_paths = check_and_get_files(self.val_data_path)
            test_npz_paths, test_pdb_paths = check_and_get_files(self.test_data_path)

            logging.info("Cross referencing train and val")
            cross_reference_files(train_npz_paths, val_npz_paths)
            logging.info("Cross referencing train and test")
            cross_reference_files(train_npz_paths, test_npz_paths)
            logging.info("Cross referencing val and test")
            cross_reference_files(val_npz_paths, test_npz_paths)

            build_lmdb(
                train_npz_paths,
                train_pdb_paths,
                lmdb_prefix_path=self.train_lmdb_prefix_path,
                resume=self.hparams.resume_build_lmdb,
            )

            build_lmdb(
                val_npz_paths,
                val_pdb_paths,
                subset=ALL_VALIDATION_SUBSET,  # prevents adding sequences we aren't going to use
                lmdb_prefix_path=self.val_lmdb_prefix_path,
                resume=self.hparams.resume_build_lmdb,
            )

            build_lmdb(
                test_npz_paths,
                test_pdb_paths,
                subset=ALL_TEST_SUBSET,  # prevents adding sequences we aren't going to use
                lmdb_prefix_path=self.test_lmdb_prefix_path,
                resume=self.hparams.resume_build_lmdb,
            )

        if not os.path.exists(self.tica_models_path):
            logging.info("TICA models directory does not exist, creating it.")

            val_metadata = load_lmdb_metadata(f"{self.val_lmdb_prefix_path}_{self.trainer.local_rank}.lmdb")
            test_metadata = load_lmdb_metadata(f"{self.test_lmdb_prefix_path}_{self.trainer.local_rank}.lmdb")

            val_npz_paths = val_metadata["npz_paths"]
            test_npz_paths = test_metadata["npz_paths"]
            all_npz_paths = {**val_npz_paths, **test_npz_paths}

            val_pdb_paths = val_metadata["pdb_paths"]
            test_pdb_paths = test_metadata["pdb_paths"]
            all_pdb_paths = {**val_pdb_paths, **test_pdb_paths}

            prepare_tica_models(all_npz_paths, all_pdb_paths, dir=self.tica_models_path)

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

        # get the correct rank for the lmdb
        self.train_lmdb_path = f"{self.train_lmdb_prefix_path}_{self.trainer.local_rank}.lmdb"
        self.val_lmdb_path = f"{self.val_lmdb_prefix_path}_{self.trainer.local_rank}.lmdb"
        self.test_lmdb_path = f"{self.test_lmdb_prefix_path}_{self.trainer.local_rank}.lmdb"

        train_metadata = load_lmdb_metadata(self.train_lmdb_path)
        val_metadata = load_lmdb_metadata(self.val_lmdb_path)
        test_metadata = load_lmdb_metadata(self.test_lmdb_path)

        train_seq_names = list(train_metadata["num_samples"].keys())

        # Fitler out train sequences that are not in the num_aa_range
        train_seq_names = [seq_name for seq_name in train_seq_names if len(seq_name) in self.num_aa_range]

        all_val_seq_names = list(val_metadata["num_samples"].keys())
        # Find the correct validation subset
        if self.num_aa_range == [2]:
            val_seq_names = list(VALIDATION_SUBSET_DICT["2"].keys())
        elif self.num_aa_range == [4]:
            val_seq_names = list(VALIDATION_SUBSET_DICT["4"].keys())
        elif self.num_aa_range == [8]:
            val_seq_names = list(VALIDATION_SUBSET_DICT["8"].keys())
        elif self.num_aa_range == [2, 3, 4]:
            val_seq_names = list(VALIDATION_SUBSET_DICT["24"].keys())
        elif self.num_aa_range == [2, 3, 4, 5, 6, 7, 8] or self.num_aa_range == [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            val_seq_names = list(VALIDATION_SUBSET_DICT["248"].keys())
        else:
            raise ValueError(
                f"Validation subset not defined for num_aa_range {self.num_aa_range}. "
                "Please define a new validation subset in VALIDATION_SUBSET_DICT."
            )
        # Check that the validation subset sequences are all present in the lmdb database
        missing_seq_names = [seq_name for seq_name in val_seq_names if seq_name not in all_val_seq_names]
        if missing_seq_names:
            raise ValueError(
                "Some validation subset sequence names are not present in the validation sequence names: "
                f"{missing_seq_names}. Please ensure that all subset sequence names are valid."
            )

        # Eval on the unions of the different aa lengths
        all_test_seq_names = list(test_metadata["num_samples"].keys())
        test_seq_names = {}
        if 2 in self.num_aa_range:
            test_seq_names.update(TEST_SUBSET_DICT["2"])
        if 4 in self.num_aa_range:
            test_seq_names.update(TEST_SUBSET_DICT["4"])
        if 8 in self.num_aa_range:
            test_seq_names.update(TEST_SUBSET_DICT["8"])
        if (9 in self.num_aa_range) or (10 in self.num_aa_range):
            test_seq_names.update(SCALING_SUBSET)
        if not test_seq_names:
            raise ValueError(
                f"Test subset not defined for num_aa_range {self.num_aa_range}. "
                "Please define a new test subset in TEST_SUBSET_DICT."
            )
        # Check that the test subset sequences are all present in the lmdb database
        missing_test_seq_names = [seq_name for seq_name in test_seq_names if seq_name not in all_test_seq_names]
        if missing_test_seq_names:
            raise ValueError(
                "Some test subset sequence names are not present in the test sequence names: "
                f"{missing_test_seq_names}. Please ensure that all subset sequence names are valid."
            )

        # Filter the metadata to only include the sequences in the train / validation / test sets
        for key in train_metadata.keys():
            if isinstance(train_metadata[key], dict):
                train_metadata[key] = {k: v for k, v in train_metadata[key].items() if k in train_seq_names}
        for key in val_metadata.keys():
            if isinstance(val_metadata[key], dict):
                val_metadata[key] = {k: v for k, v in val_metadata[key].items() if k in val_seq_names}
        for key in test_metadata.keys():
            if isinstance(test_metadata[key], dict):
                test_metadata[key] = {k: v for k, v in test_metadata[key].items() if k in test_seq_names}

        self.train_seq_names = train_seq_names
        self.val_seq_names = val_seq_names
        self.test_seq_names = test_seq_names

        train_max_num_particles = max(list(train_metadata["num_particles"].values()))
        val_max_num_particles = max(list(val_metadata["num_particles"].values()))
        test_max_num_particles = max(list(test_metadata["num_particles"].values()))

        tica_model_files = os.listdir(self.tica_models_path)

        self.tica_model_paths = {  # TODO horrible naming vs self.tica_models_paths
            tica_model_file.split("-")[0]: os.path.join(self.tica_models_path, tica_model_file)
            for tica_model_file in tica_model_files
        }

        assert train_max_num_particles >= val_max_num_particles, (
            "Train largest system must be greater than or equal to val largest system for pos_embed learning."
        )
        assert train_max_num_particles >= test_max_num_particles, (
            "Train largest system must be greater than or equal to test largest system for pos_embed learning."
        )
        # Need to check here so TarFlow is correctly initalized for data
        assert self.hparams.num_particles > train_max_num_particles, (
            "TarFlow num_particles must be greater than the largest system size. "
            + f"Max num particles={train_max_num_particles}. Set num_particles in data config "
            + f"to {train_max_num_particles + 1}."
        )

        pdb_paths = [
            *train_metadata["pdb_paths"].values(),
            *val_metadata["pdb_paths"].values(),
            *test_metadata["pdb_paths"].values(),
        ]
        self.pdb_dict, self.topology_dict = load_pdbs_and_topologies(pdb_paths, self.num_aa_range)
        self.encoding_dict = get_encoding_dict(self.topology_dict)

        permutation_pkl_path = "permutations.pkl"
        if os.path.exists(permutation_pkl_path):
            logging.info("Loading permutations from existing pickle file.")
            with open(permutation_pkl_path, "rb") as f:
                self.permutations_dict = pickle.load(f)
        else:
            self.permutations_dict = get_permutations_dict(self.topology_dict)
            if self.trainer.local_rank == 0:
                with open(permutation_pkl_path, "wb") as permutation_pkl_file:
                    pickle.dump(self.permutations_dict, permutation_pkl_file)

        residue_tokenization_pkl_path = "residue_tokenization.pkl"
        if os.path.exists(residue_tokenization_pkl_path):
            logging.info("Loading residue tokenization from existing pickle file.")
            with open(residue_tokenization_pkl_path, "rb") as f:
                self.residue_tokenization_dict = pickle.load(f)
        else:
            logging.info("Creating residue tokenization dictionary.")
            self.residue_tokenization_dict = get_residue_tokenization_dict(self.topology_dict)
            if self.trainer.local_rank == 0:
                with open(residue_tokenization_pkl_path, "wb") as f:
                    pickle.dump(self.residue_tokenization_dict, f)

        total_samples = 0
        weighted_vars = []
        for k, v in train_metadata["vars"].items():
            num_samples = train_metadata["num_samples"][k]
            total_samples += num_samples
            weighted_vars.append(v * num_samples)
        self.std = torch.sqrt(torch.sum(torch.tensor(weighted_vars)) / total_samples)
        logging.info(f"Standard deviation of training data: {self.std}")

        transform_list = [
            StandardizeTransform(self.std, self.hparams.num_dimensions),
            Random3DRotationTransform(self.hparams.num_dimensions),
        ]
        if self.hparams.com_augmentation:
            # Center of mass augmentation has std 1/sqrt(N) where N is the mean number of atoms
            # in the system. This is the same center of mass std deviation as the prior.

            # Compute the mean number of particles per system in the training set
            weighted_num_particles = []
            total_num_samples = 0
            for seq_name in self.train_seq_names:
                num_samples = train_metadata["num_samples"][seq_name]
                num_particles = train_metadata["num_particles"][seq_name]
                weighted_num_particles.append(num_samples * num_particles)
                total_num_samples += num_samples
            mean_num_particles = sum(weighted_num_particles) / total_num_samples

            logging.info(f"Mean number of training particles: {mean_num_particles}")

            transform_list.append(
                CenterOfMassTransform(
                    1 / math.sqrt(mean_num_particles),
                    self.hparams.num_dimensions,
                )
            )
        if self.hparams.atom_noise_augmentation_factor:
            self.atom_noise_std = (
                self.hparams.atom_noise_augmentation_factor * 0.1 / self.std
            )  # 0.1 is length in nm of N-H bond
            transform_list.append(
                AtomNoiseTransform(
                    self.atom_noise_std,
                )
            )
        else:
            self.atom_noise_std = 0.0
        transform_list = transform_list + [
            AddEncodingTransform(self.encoding_dict),
            AddPermutationsTransform(self.permutations_dict),
            AddResidueTokenizationTransform(self.residue_tokenization_dict),
            PaddingTransform(self.hparams.num_particles, self.hparams.num_dimensions),
        ]

        transforms = torchvision.transforms.Compose(transform_list)

        self.data_train = PeptideDataset(
            self.train_lmdb_path,
            seq_names=self.train_seq_names,
            num_dimensions=self.hparams.num_dimensions,
            transform=transforms,
        )

        self.data_val = PeptideDataset(
            self.val_lmdb_path,
            seq_names=self.val_seq_names,
            num_dimensions=self.hparams.num_dimensions,
            transform=transforms,
        )

        test_transform_list = [
            StandardizeTransform(self.std, self.hparams.num_dimensions),
            AddEncodingTransform(self.encoding_dict),
            AddPermutationsTransform(self.permutations_dict),
            AddResidueTokenizationTransform(self.residue_tokenization_dict),
            PaddingTransform(self.hparams.num_particles, self.hparams.num_dimensions),
        ]

        test_transforms = torchvision.transforms.Compose(test_transform_list)

        self.data_test = PeptideDataset(
            self.test_lmdb_path,
            seq_names=self.test_seq_names,
            num_dimensions=self.hparams.num_dimensions,
            transform=test_transforms,
        )

        logging.info(f"Train dataset size: {len(self.data_train)}")
        logging.info(f"Validation dataset size: {len(self.data_val)}")
        logging.info(f"Test dataset size: {len(self.data_test)}")

    def setup_potential(self, sequence: str):
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        nonbondedMethod = openmm.app.CutoffNonPeriodic
        nonbondedCutoff = 2.0 * openmm.unit.nanometer
        temperature = 310

        # Initalize forcefield systemq
        system = forcefield.createSystem(
            self.pdb_dict[sequence].topology,
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

    def prepare_eval(self, eval_sequence: str):
        if eval_sequence in self.val_seq_names:
            true_samples = self.data_val.get_seq_data(eval_sequence)
        elif eval_sequence in self.test_seq_names:
            true_samples = self.data_test.get_seq_data(eval_sequence)
        else:
            raise ValueError(
                f"Sequence {eval_sequence} not found in validation or test set. Please provide a valid sequence name."
            )

        # TODO can this be handle better? in the lmdb?
        # how to do nice plots? - i suppose plots will be a more rare occasion
        true_samples = true_samples[: self.hparams.num_eval_samples]
        true_samples = true_samples.reshape(
            true_samples.shape[0],
            -1,
        )
        true_samples = self.normalize(true_samples)
        encoding = self.encoding_dict[eval_sequence]
        permutations = self.permutations_dict[eval_sequence]
        residue_tokenization = self.residue_tokenization_dict[eval_sequence]

        potential = self.setup_potential(eval_sequence)
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encoding, energy_fn

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

        if self.hparams.do_plots:
            plot_ramachandran(
                log_image_fn,
                true_data.samples,
                self.topology_dict[sequence],
                prefix=prefix + "true",
            )
            plot_tica(
                log_image_fn,
                true_data.samples,
                self.topology_dict[sequence],
                self.tica_model_paths[sequence],
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
                continue

            logging.info(f"Evaluating {prefix + name} samples")

            data = data[: self.hparams.num_eval_samples * 2]  # slice out extra samples for those lost to symmetry

            symmetry_metrics, symmetry_change = resolve_chirality(
                true_data.samples,
                data.samples,
                self.topology_dict[sequence],
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
                        tica_model=self.tica_model_paths[sequence],
                        num_eval_samples=self.hparams.num_eval_samples,
                        prefix=prefix + name,
                        compute_distribution_distances=False,
                    )
                )
                if self.hparams.do_plots and len(data) > 32:
                    plot_ramachandran(log_image_fn, data.samples, self.topology_dict[sequence], prefix=prefix + name)
                    plot_tica(
                        log_image_fn,
                        data.samples,
                        self.topology_dict[sequence],
                        self.tica_model_paths[sequence],
                        prefix=prefix + name,
                    )

        # reduce size so plotting doesn't crash with many samples
        true_data = true_data[: self.hparams.num_eval_samples]
        proposal_data = proposal_data[: self.hparams.num_eval_samples]
        resampled_data = resampled_data[: self.hparams.num_eval_samples]
        smc_data = smc_data[: self.hparams.num_eval_samples] if smc_data is not None else None

        if self.hparams.do_plots:
            plot_energies(
                log_image_fn,
                true_data.energy,
                proposal_data.energy if len(proposal_data) > 0 else None,
                resampled_data.energy if len(resampled_data) > 0 else None,
                smc_data.energy if (smc_data is not None and len(smc_data) > 0) else None,
                prefix=prefix,
            )
            plot_atom_distances(
                log_image_fn,
                true_data.samples,
                proposal_data.samples if len(proposal_data) > 0 else None,
                resampled_data.samples if len(resampled_data) > 0 else None,
                smc_data.samples if (smc_data is not None and len(smc_data) > 0) else None,
                prefix=prefix,
            )
            plot_com_norms(
                log_image_fn,
                proposal_data.samples if len(proposal_data) > 0 else None,
                resampled_data.samples if len(resampled_data) > 0 else None,
                smc_data.samples if (smc_data is not None and len(smc_data) > 0) else None,
                prefix=prefix,
            )

        return metrics
