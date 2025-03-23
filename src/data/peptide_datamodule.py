import logging
import os
from typing import Any, Optional
import wget
import math

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm
import torch
import torchvision
from bgflow import OpenMMBridge, OpenMMEnergy
from lightning.pytorch.loggers import WandbLogger
from matplotlib.colors import LogNorm
from openmm import app

from src.data.base_datamodule import BaseDataModule
from src.data.components.encodings import ATOM_TYPE_ENCODING_DICT, AA_TYPE_ENCODING_DICT
from src.data.components.center_of_mass import CenterOfMassTransform
from src.data.components.rotation import Random3DRotationTransform
from src.data.components.transform_dataset import TransformDataset
from src.data.components.distribution_distances import (
    compute_distribution_distances,
    compute_energy_distances,
)
from src.data.components.optimal_transport import torus_wasserstein
from src.models.components.utils import (
    check_symmetry_change,
    compute_chirality_sign,
    find_chirality_centers,
)
from src.utils.data_types import SamplesData

logger = logging.getLogger(__name__) # TODO what is actually going on with the logging

class PeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "data/al3",
        npy_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        pdb_url: str = "https://osf.io/download/y7ntk/?view_only=1052300a21bd43c08f700016728aa96e",
        npy_filename: str = "AlaAlaAla_310K.npy",
        pdb_filename: str = "AlaAlaAla_310K.pdb",
        n_particles: int = 33,
        n_dimensions: int = 3,
        com_augmentation: bool = False,
        dim: int = 99,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaling: float = 1.0, # TODO
        make_iid: bool = False,
        # TODO maybe make this all just *args?
        n_train_samples: int = 100_000, # First N trajectory samples
        n_val_samples: int = 20_000, # Following M trajectory samples
        n_test_samples: int = 100_000, # Random K trajectory samples from the rest
        n_test_samples_small: int = 10_000, # Smaller subset for plotting etc
        hist_min_energy: float = -200.0, # TODO
        hist_max_energy: float = 400.0, # TODO
        num_eval_samples: int = 10_000, # TODO
    ):
        super().__init__(
            n_particles=n_particles, # TODO why some done as hparams and others not?
            n_dimensions=n_dimensions,
            dim=dim,
        )
        assert dim == n_particles * n_dimensions

        ### again why are these not hparams?

        self.scaling = scaling

        self.batch_size_per_device = batch_size

        self.adj_list = None
        self.atom_types = None

        ###

        self.npy_path = f"{self.hparams.data_dir}/{self.hparams.npy_filename}"
        self.pdb_path = f"{self.hparams.data_dir}/{self.hparams.pdb_filename}"

        # Setup transforms
        transform_list = [Random3DRotationTransform(self.n_particles, self.n_dimensions)]
        if self.hparams.com_augmentation:
            transform_list.append(CenterOfMassTransform(self.n_particles, self.n_dimensions, 1 / math.sqrt(self.n_particles))) # TODO check these values
        self.transforms = torchvision.transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.hparams.data_dir, exist_ok=True)

        def download(url, file_path):
            logger.info(f"Downloading file from {url}")
            wget.download(url, file_path)
            logger.info(f"File downloaded and saved as {file_path}")

        # Download npy file
        if not os.path.exists(self.npy_path):
            download(self.hparams.npy_url, self.npy_path)

        # Download pdb file
        if not os.path.exists(self.pdb_path):
            download(self.hparams.pdb_url, self.pdb_path)

    def setup_data(self):

        # Load the data + tensorize
        data = np.load(self.npy_path, allow_pickle=True)
        data = data.reshape(-1, self.dim)
        data = torch.tensor(data).float() / self.scaling

        if self.hparams.make_iid:
            rand_idx = torch.randperm(data.shape[0])
            data = data[rand_idx]

        # TODO for updated results remove this slicing?
        # if self.n_particles == 42:
        #     data = data[:700000]
        # elif self.n_particles == 33:
        #     data = data[:300000]

        # Zero center of mass
        data = self.zero_center_of_mass(data)

        # Split data
        train_data = data[:self.hparams.n_train_samples]
        test_data = data[self.hparams.n_train_samples:]

        # Compute std on only train data
        self.std = train_data.std()

        # Standardize the data
        train_data = self.normalize(train_data)
        test_data = self.normalize(test_data)

        # Create training dataset with transforms applied
        self.data_train = TransformDataset(train_data, transform=self.transforms) 

        # Split val and test data
        self.data_val, self.data_test = test_data[:self.hparams.n_val_samples], test_data[self.hparams.n_val_samples:]

        # Randomized ordering of val samples
        val_rng = np.random.default_rng(0)
        self.data_val = torch.tensor(val_rng.permutation(self.data_val))

        # Randomized ordering / subset of test samples
        test_rng = np.random.default_rng(1)
        self.data_test = torch.tensor(test_rng.permutation(self.data_test))[self.hparams.n_test_samples:]

        # Smaller subset for plotting 
        self.data_test_small = self.data_test[:self.hparams.n_test_samples_small]

    def setup_potential(self):

        logger.info(f"Loading pdb from {self.pdb_path}")
        self.topology = md.load_topology(self.pdb_path)
        self.pdb = app.PDBFile(self.pdb_path)
    
        # Different system configs for different datasets
        if self.n_particles != 42:
            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
            nonbondedMethod = openmm.app.CutoffNonPeriodic
            nonbondedCutoff = 2.0 * openmm.unit.nanometer
            temperature = 310
        else:
            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")
            nonbondedMethod = openmm.app.NoCutoff
            nonbondedCutoff = 0.9 * openmm.unit.nanometer
            temperature = 300

        # Initalize forcefield system
        system = forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod = nonbondedMethod,
            nonbondedCutoff = nonbondedCutoff,
            constraints=None,
        )
        
        # Initialize integrator
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )

        # Initialize potential
        self.potential = OpenMMEnergy(
            bridge=OpenMMBridge(system, integrator, platform_name="CUDA")
        )
    
    def setup_atom_types(self):
        atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
        atom_types = []
        for atom_name in self.topology.atoms:
            atom_types.append(atom_name.name[0])
        self.atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))

    def setup_adj_list(self):   
        self.adj_list = torch.from_numpy(
            np.array([(b.atom1.index, b.atom2.index) for b in self.topology.bonds], dtype=np.int32)
        )

    def setup_atom_encoding(self):
        aa_pos_encoding = []
        aa_type_encoding = []
        atom_type_encoding = []

        for i, aa in enumerate(self.topology.residues):
            for atom in aa.atoms:

                aa_pos_encoding.append(i)
                aa_type_encoding.append(AA_TYPE_ENCODING_DICT[aa.name])

                # TODO double check this with Leon
                # Standarize side-chain H atom encoding
                if atom.name[0] == "H" and atom.name[-1] in ("1", "2", "3"):
                    # For these AA the H-X-N atoms are not interchangable
                    if aa.name in (
                        "HIS",
                        "PHE",
                        "TRP",
                        "TYR"
                        ) and atom.name[:2] in (
                            "HE",
                            "HD",
                            "HZ",
                            "HH",
                            ):
                        pass
                    else:
                        atom.name = atom.name[:-1]

                # Standarize side-chain O atom encoding
                if atom.name[:2] == "OE" or atom.name[:2] == "OD":
                    atom.name = atom.name[:-1]

                atom_type_encoding.append(ATOM_TYPE_ENCODING_DICT[atom.name])

        self.atom_type_encoding = np.array(atom_type_encoding)
        self.aa_pos_encoding = np.array(aa_pos_encoding)
        self.aa_type_encoding = np.array(aa_type_encoding)

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
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        self.setup_data()
        self.setup_potential()
        self.setup_atom_types()
        self.setup_adj_list()
        self.setup_atom_encoding()

    def get_phi_psi_vectors(self, samples):
        samples = self.unnormalize(samples).cpu()
        samples = samples.reshape(-1, self.n_particles, self.n_dimensions)
        traj_samples = md.Trajectory(samples, topology=self.topology)
        phis = md.compute_phi(traj_samples)[1]
        psis = md.compute_psi(traj_samples)[1]
        return phis, psis

    def compute_ramachandran_metrics(self, true_samples, pred_samples, prefix=""):

        phis_true, psis_true = self.get_phi_psi_vectors(true_samples)
        x_true = torch.cat([torch.from_numpy(phis_true), torch.from_numpy(psis_true)], dim=1)

        phis_pred, psis_pred = self.get_phi_psi_vectors(pred_samples)
        x_pred = torch.cat([torch.from_numpy(phis_pred), torch.from_numpy(psis_pred)], dim=1)

        metrics = compute_distribution_distances(x_true, x_pred, prefix=prefix)
        metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)

        return metrics

    def plot_ramachandran(self, samples, prefix: str = "", wandb_logger: WandbLogger = None):

        phis, psis = self.get_phi_psi_vectors(samples)

        for i in range(phis.shape[1]):
            print(f"Plotting Ramachandran {i} out of {phis.shape[1]}")
            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
            )
            ticks = np.array(
                [np.exp(-6) * h.max(), np.exp(-4.0) * h.max(), np.exp(-2) * h.max(), h.max()]
            )
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            # ax.set_title("Boltzmann Generator", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im, ticks=ticks)
            # cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
            cbar.ax.set_yticklabels([6.0, 4.0, 2.0, 0.0], fontsize=25)

            cbar.ax.invert_yaxis()
            cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran/{i}", [fig])

            phi_tmp = phis[:, i]
            psi_tmp = psis[:, i]
            fig, ax = plt.subplots()
            plot_range = [-np.pi, np.pi]
            h, x_bins, y_bins, im = ax.hist2d(
                phi_tmp,
                psi_tmp,
                100,
                norm=LogNorm(),
                range=[plot_range, plot_range],
                rasterized=True,
            )
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_ticks([])
            cbar = fig.colorbar(im)  # , ticks=ticks)
            im.set_clim(vmax=samples.shape[0] // 20)
            cbar.ax.set_ylabel(f"Count, max = {int(h.max())}", fontsize=18)
            if wandb_logger is not None:
                wandb_logger.log_image(f"{prefix}/ramachandran_simple/{i}", [fig])

        return fig

    def check_and_fix_symmetry(self, true_data, pred_data, prefix=""):

        true_samples = true_data.samples
        pred_samples = pred_data.samples
    
        chirality_centers = find_chirality_centers(self.adj_list, self.atom_types)
        reference_signs = compute_chirality_sign(
            true_samples.reshape(-1, self.n_particles, 3)[[1]], chirality_centers
        )
        pred_samples = pred_samples.reshape(-1, self.n_particles, 3)
        symmetry_change = check_symmetry_change(
            pred_samples, chirality_centers, reference_signs
        )
        pred_samples[symmetry_change] *= -1
        correct_symmetry_rate = 1 - symmetry_change.sum() / len(symmetry_change)
        symmetry_change = check_symmetry_change(
            pred_samples, chirality_centers, reference_signs
        )
        uncorrectable_symmetry_rate = symmetry_change.sum() / len(symmetry_change)

        pred_data = pred_data[~symmetry_change]

        metrics = {
            prefix + "/correct_symmetry_rate": correct_symmetry_rate,
            prefix + "/uncorrectable_symmetry_rate": uncorrectable_symmetry_rate,
        }

        if uncorrectable_symmetry_rate > 0.1:
            logging.warning(
                f"Uncorrectable symmetry rate is {uncorrectable_symmetry_rate:.2f}, "
            )

        return pred_data, metrics

    def compute_all_metrics(self, true_data, pred_data, prefix: str = ""):
        """Computes all metrics between true and predicted data."""

        metrics = {}

        if len(pred_data) < 0.9 * self.hparams.num_eval_samples:
            logging.warning(r"Less than 90% of required eval samples supplied.")

        # Slice data to subset
        num_eval_samples = min(self.hparams.num_eval_samples, len(pred_data), len(true_data))
        true_data = true_data[:num_eval_samples]
        pred_data = pred_data[:num_eval_samples]
        metrics[f"num_eval_samples"] = min(num_eval_samples, len(pred_data))

        # Distribtuion distance metrics
        metrics.update(
            compute_distribution_distances(
                self.unnormalize(true_data.samples),
                self.unnormalize(pred_data.samples),
                prefix=prefix
                )
        )
        logging.info("Distance metrics computed")

        # Energy metrics
        metrics.update(
            compute_energy_distances(
                true_data.energy,
                pred_data.energy,
                prefix=prefix,
                )
        )
        metrics[f"{prefix}/mean_energy"] = pred_data.energy.mean().cpu()
        logging.info("Energy metrics computed")

        # Ramachandran metrics
        metrics.update(
            self.compute_ramachandran_metrics(
                true_data.samples,
                pred_data.samples,
                prefix=prefix
                )
        )
        logging.info("Ramachandran metrics computed")

        return metrics

    def metrics_and_plots(
        self,
        true_data: SamplesData,
        proposal_data: SamplesData,
        resampled_data: SamplesData,
        jarzynski_data: Optional[SamplesData] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics and plots at the end of an epoch."""

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        metrics = {}

        self.plot_ramachandran(true_data.samples, prefix=prefix + "ground_truth/rama")

        logging.info(f"Computing metrics for proposal data")
        proposal_data, symmetry_metrics = self.check_and_fix_symmetry(true_data, proposal_data, prefix + "proposal")
        metrics.update(symmetry_metrics)
        if len(proposal_data) == 0:
            logging.warning("No proposal samples left after symmetry correction.")
        else:
            metrics.update(self.compute_all_metrics(true_data, proposal_data, prefix + "proposal"))
            self.plot_ramachandran(proposal_data.samples, prefix=prefix + "proposal/rama")

        logging.info(f"Computing metrics for resampled data")
        resampled_data, symmetry_metrics = self.check_and_fix_symmetry(true_data, resampled_data, prefix + "resampled")
        metrics.update(symmetry_metrics)
        if len(resampled_data) == 0:
            logging.warning("No resampled samples left after symmetry correction.")
        else:
            metrics.update(self.compute_all_metrics(true_data, resampled_data, prefix + "resampled"))
            self.plot_ramachandran(resampled_data.samples, prefix=prefix + "resampled/rama")

        if jarzynski_data is not None:
            logging.info(f"Computing metrics for jarzynski data")
            jarzynski_data, symmetry_metrics = self.check_and_fix_symmetry(true_data, jarzynski_data, prefix + "jarzynski")
            metrics.update(symmetry_metrics)
            if len(jarzynski_data) == 0:
                logging.warning("No jarzynski samples left after symmetry correction.")
            else:
                metrics.update(self.compute_metrics(true_data, jarzynski_data, prefix + "jarzynski"))
                self.plot_ramachandran(jarzynski_data.samples, prefix=prefix + "jarzynski/rama")

        logging.info(f"Plotting energies")
        self.plot_energies(
            true_data.energy[self.hparams.num_eval_samples:],
            proposal_data.energy if len(proposal_data) > 0 else None,
            resampled_data.energy if len(resampled_data) > 0 else None,
            jarzynski_data.energy if (jarzynski_data is not None and len(jarzynski_data) > 0) else None,
            prefix=prefix,
        )

        logging.info(f"Plotting interatomic distances")
        self.plot_atom_distances(
            true_data.samples[self.hparams.num_eval_samples:],
            proposal_data.samples if len(proposal_data) > 0 else None,
            resampled_data.samples if len(resampled_data) > 0 else None,
            jarzynski_data.samples if (jarzynski_data is not None and len(jarzynski_data) > 0) else None,
            prefix=prefix,
        )

        return metrics