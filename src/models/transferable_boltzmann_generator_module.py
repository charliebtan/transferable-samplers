import copy
import logging
import os
import statistics as stats
import time
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import mdtraj as md
import torchmetrics
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Normal
from torchmetrics import MeanMetric
from tqdm import tqdm

import numpy as np

from src.data.components.data_types import SamplesData
from src.data.components.symmetry import get_symmetry_change, resolve_chirality
from src.models.components.ema import EMA
from src.models.components.priors import NormalDistribution
from src.models.components.smc.base_sampler import SMCSampler
from src.models.components.utils import resample

logger = logging.getLogger(__name__)


class BADNormalDistribution:
    def __init__(self, num_dimensions: int = 3, mean: float = 0.0, std: float = 1.0, mean_free: bool = False):
        self.num_dimensions = num_dimensions
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)
        self.mean_free = mean_free

    def sample(
        self, num_samples: int, num_particles: int, mask: torch.Tensor | None = None, device="cpu"
    ) -> torch.Tensor:
        x = self.distribution.sample((num_samples, num_particles, self.num_dimensions)).to(device)
        if self.mean_free:
            if mask is None:
                mask = torch.ones((num_samples, num_particles), device=device)
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]
        return x.reshape(num_samples, -1)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 2
        num_samples = x.shape[0]
        num_particles = x.shape[-1] // self.num_dimensions
        if mask is None:
            mask = torch.ones((num_samples, num_particles), device=x.device)
        if self.mean_free:
            x = x.reshape(num_samples, -1, self.num_dimensions)
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]
            x = x.reshape(num_samples, -1)

        pointwise_energy = -self.distribution.log_prob(x)
        pointwise_energy = pointwise_energy.reshape(num_samples, -1, self.num_dimensions)
        pointwise_energy = pointwise_energy * mask.unsqueeze(-1)
        pointwise_energy = pointwise_energy.reshape(num_samples, -1)
        num_particles = mask.sum(dim=-1, keepdim=True)
        # account for the pad tokens when taking the mean
        energy = pointwise_energy.sum(dim=-1, keepdims=True) / (num_particles * self.num_dimensions)

        return energy


class TransferableBoltzmannGeneratorLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        datamodule: LightningDataModule,
        smc_sampler: SMCSampler,
        sampling_config: dict,
        ema_decay: float,
        compile: bool,
        use_com_adjustment: bool = False,
        dont_fix_symmetry: bool = False,
        dont_fix_chirality: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `FlowMatchLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=("datamodule"))
        if args or kwargs:
            logger.warning(f"Unexpected arguments: {args}, {kwargs}")

        self.net = net
        if self.hparams.ema_decay > 0:
            self.net = EMA(net, decay=self.hparams.ema_decay)

        self.datamodule = datamodule

        self.smc_sampler = smc_sampler(
            log_image_fn=self.log_image,
        )

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        self.train_metrics = torchmetrics.MetricCollection({"loss": MeanMetric()}, prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.prior = NormalDistribution(
            self.datamodule.hparams.num_dimensions,
            mean_free=self.hparams.mean_free_prior,
        )

    def log_image(self, img: torch.Tensor, title: str = None) -> None:
        """Log an image to the logger.

        :param img: The image to log.
        :param title: The title of the image.
        """
        if self.loggers is not None:
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(title, [img])

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        assert len(batch["x"].shape) == 2, "molecules must be in vector format"
        loss = self.model_step(batch)
        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return {"optimizer": optimizer}

    def predict_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of samples.

        :param batch: A batch of (dummy) data.
        :return: A tuple containing the generated samples, the log probability, and the prior
            samples.
        """
        samples, log_p, prior_samples = self.batched_generate_samples(batch.shape[0])
        return samples, log_p, prior_samples

    def batched_generate_samples(
        self,
        total_size: int,
        encoding: Optional[dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        dummy_ll: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.hparams.sampling_config.batch_size
        samples = []
        log_ps = []
        prior_samples = []
        for _ in tqdm(range(total_size // batch_size)):
            s, lp, ps = self.generate_samples(batch_size, encoding=encoding, dummy_ll=dummy_ll)
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        if total_size % batch_size > 0:
            s, lp, ps = self.generate_samples(total_size % batch_size, encoding=encoding, dummy_ll=dummy_ll)
            samples.append(s)
            log_ps.append(lp)
            prior_samples.append(ps)
        samples = torch.cat(samples, dim=0)
        log_ps = torch.cat(log_ps, dim=0)
        prior_samples = torch.cat(prior_samples, dim=0)
        return samples, log_ps, prior_samples

    def generate_samples(
        self, batch_size: int, encoding: Optional[dict[str, torch.Tensor]] = None, n_timesteps: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """
        raise NotImplementedError

    @torch.no_grad()
    def eval_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        prefix: str = "val",
    ) -> None:
        loss = self.model_step(batch)
        if prefix == "val":
            self.val_metrics.update(loss)
        elif prefix == "test":
            self.test_metrics.update(loss)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        :param batch_idx: The index of the current batch.
        """
        self.eval_step(batch, batch_idx, prefix="test")

    def on_eval_epoch_end(self, metrics, prefix: str = "val") -> None:
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()
        if self.hparams.ema_decay > 0:
            if self.hparams.eval_ema:
                self.net.backup()
                self.net.copy_to_model()
                self.evaluate_all(prefix)
                self.net.restore_to_model()
            if self.hparams.eval_non_ema:
                self.evaluate_all(prefix + "/non_ema")
        else:
            self.evaluate(prefix)
        plt.close("all")

    def add_aggregate_metrics(self, metrics: dict[str, torch.Tensor], prefix: str = "val") -> dict[str, torch.Tensor]:
        """Aggregate metrics across all sequences."""

        mean_dict_list = defaultdict(list)
        median_dict_list = defaultdict(list)
        count_dict = defaultdict(int)

        # Parse and aggregate metrics along peptide sequences
        for key, value in metrics.items():
            if key.startswith(prefix):
                # Extract sequence and metric name
                parts = key.split("/")
                metric_name = "/".join(parts[2:])

                # Add to mean and median dictionaries
                mean_key = f"{prefix}/mean/{metric_name}"
                median_key = f"{prefix}/median/{metric_name}"
                count_key = f"{prefix}/count/{metric_name}"

                if isinstance(value, torch.Tensor):
                    value = value.item()
                elif isinstance(value, (int, float)):
                    value = float(value)

                mean_dict_list[mean_key].append(value)
                median_dict_list[median_key].append(value)
                count_dict[count_key] += 1

        # Compute mean and median for each metric
        mean_dict = {}
        median_dict = {}
        for key, value in mean_dict_list.items():
            mean_dict[key] = stats.mean(value)

        for key, value in median_dict_list.items():
            median_dict[key] = stats.median(value)

        metrics.update(mean_dict)
        metrics.update(median_dict)
        metrics.update(count_dict)
        return metrics

    def evaluate_all(self, prefix):
        metrics = {}
        eval_seq_names = self.datamodule.val_seq_names if prefix.startswith("val") else self.datamodule.test_seq_names
        if (prefix.startswith("test") or prefix.startswith("val")) and self.hparams.get("eval_seq_name") is not None:
            if self.hparams.eval_seq_name not in eval_seq_names:
                raise ValueError(f"{self.hparams.eval_seq_name} not in set of test sequences: {eval_seq_names}")

            if not isinstance(self.hparams.eval_seq_name, list):
                eval_seq_names = [self.hparams.eval_seq_name]
            else:
                eval_seq_names = self.hparams.eval_seq_name

        for seq_name in eval_seq_names:
            true_samples, encoding, energy_fn = self.datamodule.prepare_eval(seq_name)
            logging.info(f"Evaluating {seq_name} samples")
            metrics.update(
                self.evaluate(
                    seq_name,
                    true_samples,
                    encoding,
                    energy_fn,
                    prefix=f"{prefix}/{seq_name}",
                    proposal_generator=self.batched_generate_samples,
                )
            )

        # Aggregate metrics across all sequences
        if self.local_rank == 0:
            metric_object_list = [self.add_aggregate_metrics(metrics, prefix=prefix)]
        else:
            metric_object_list = [None]  # List must have same length for broadcast
        if self.trainer.world_size > 1:
            # Broadcast metrics to all processes - must log from all for checkpointing
            torch.distributed.broadcast_object_list(metric_object_list, src=0)
        self.log_dict(metric_object_list[0])

    @torch.no_grad()
    def evaluate(
        self, sequence, true_samples, encoding, energy_fn, prefix: str = "val", proposal_generator=None, output_dir=None
    ) -> None:
        """Generates samples from the proposal and runs SMC if enabled.
        Also computes metrics, through the datamodule function "metrics_and_plots".
        """

        true_data = SamplesData(
            self.datamodule.as_pointcloud(self.datamodule.unnormalize(true_samples)),
            energy_fn(true_samples),
        )

        # Define proposal generator
        if proposal_generator is None:
            proposal_generator = self.batched_generate_samples
            if "dummy_ll" in self.hparams and self.hparams.dummy_ll:
                proposal_generator = lambda x: self.batched_generate_samples(x, dummy_ll=True)

        # if self.hparams.sampling_config.get("leon", False):
        #     data_dim = true_samples.shape[1]

        #     data = np.load(f"result_data/Flow-Matching-2AA-wloss-9layer-128-encoding-long2_{sequence}.npz")

        #     proposal_samples = self.datamodule.normalize(torch.tensor(data["samples_np"]) / 30.0)
        #     proposal_dlog_p = torch.tensor(data["dlogp_np"])
        #     prior_samples = torch.tensor(data["latent_np"])

        #     prior_log_p = -self.prior.energy(torch.tensor(prior_samples)) * data_dim
        #     proposal_log_p = prior_log_p.flatten() - proposal_dlog_p.flatten()
        # elif self.hparams.sampling_config.get("md", False):
        #     data = np.load(
        #         f"/network/scratch/t/tanc/md-runner-scbg-baselines/data/md/{sequence}/{sequence}_310_99500/99499.npz"
        #     )
        #     proposal_samples = torch.from_numpy(data["all_positions"]).float()
        #     num_samples = proposal_samples.shape[0]
        #     proposal_samples = self.datamodule.normalize(proposal_samples.view(num_samples, -1))
        #     proposal_samples = proposal_samples[: self.hparams.sampling_config.num_samples_subset]
        #     proposal_data = SamplesData(
        #         self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples)),
        #         energy_fn(proposal_samples),
        #     )
        #     proposal_log_p = torch.zeros_like(proposal_data.energy)
        #     prior_samples = np.zeros_like(proposal_samples)
        #     logging.info(f"Proposal samples shape: {proposal_samples.shape}")

        # elif self.datamodule.hparams.num_aa_max == 2:
        #     BASE_DIR_1 = "/home/mila/t/tanc/scratch/self-consume-bg/logs/eval/multiruns/2025-05-11_22-22-44"

        #     samples_dicts = []
        #     for i in range(10):
        #         found = False
        #         for j in range(500):
        #             path1 = f"{BASE_DIR_1}/{j}/{prefix}/samples_{i}.pt"

        #             if os.path.exists(path1):
        #                 samples_dicts.append(torch.load(path1))
        #                 found = True
        #                 break

        #         if not found:
        #             raise FileNotFoundError(f"Sample file samples_{i}.pt not found in either directory.")

        #     prior_samples = torch.cat([d["prior_samples"] for d in samples_dicts], dim=0)
        #     proposal_samples = torch.cat([d["proposal_samples"] for d in samples_dicts], dim=0)
        #     proposal_log_p = torch.cat([d["proposal_log_p"] for d in samples_dicts], dim=0)

        # if True:
        #     BASE_ROOT = "/home/mila/t/tanc/scratch/self-consume-bg/logs/eval/multiruns"
        #     ALL_DATES = [
        #         "2025-05-10_02-21-17",
        #         "2025-05-11_18-51-26",
        #         "2025-05-11_18-49-32",
        #         "2025-05-11_18-55-20",
        #         "2025-05-11_18-55-02",
        #         "2025-05-11_18-52-16",
        #         "2025-05-11_01-44-04",
        #         "2025-05-11_18-55-55",
        #         "2025-05-11_18-49-08",
        #         "2025-05-11_18-55-39",
        #         "2025-05-11_18-49-54",
        #         "2025-05-13_20-21-00",
        #         "2025-05-13_20-19-58",
        #     ]

        #     samples_dicts = []

        #     for i in range(10):  # samples_0.pt to samples_9.pt
        #         found = False

        #         for j in range(500):  # folders numbered 0 through 499
        #             for date in ALL_DATES:
        #                 path = f"{BASE_ROOT}/{date}/{j}/{prefix}/samples_{i}.pt"
        #                 if os.path.exists(path):
        #                     samples_dicts.append(torch.load(path))
        #                     found = True
        #                     break  # stop checking dates for this (i, j)

        #             if found:
        #                 break  # sample i found for some j/date combo

        #         if not found:
        #             logging.warning(f"Sample file samples_{i}.pt not found in any directory.")

        #     prior_samples = torch.cat([d["prior_samples"] for d in samples_dicts], dim=0)
        #     proposal_samples = torch.cat([d["proposal_samples"] for d in samples_dicts], dim=0)
        #     proposal_log_p = torch.cat([d["proposal_log_p"] for d in samples_dicts], dim=0)

        # if not self.hparams.sampling_config.get("md", False) and not self.hparams.sampling_config.get("leon", False):
        #     self.bad_prior = BADNormalDistribution(mean_free=self.hparams.mean_free_prior)

        #     num_particles = encoding["atom_type"].size(0)
        #     data_dim = num_particles * self.datamodule.hparams.num_dimensions

        #     bad_prior_log_p = self.bad_prior.energy(prior_samples).flatten() * data_dim
        #     good_prior_log_p = self.prior.energy(prior_samples).flatten() * data_dim

        #     dlog_p = proposal_log_p.flatten() + bad_prior_log_p
        #     proposal_log_p = dlog_p - good_prior_log_p.flatten()

        if self.hparams.sample_set == "unisim":

            print(f"../scratch/unisim_pepmd_results_{self.hparams.energy_maxiter}/{sequence}/{sequence}_model_ode50_inf10000_guidance0.05.xtc")

            # Load trajectory (requires both .xtc and topology file, e.g. .pdb)
            traj = md.load_xtc(f"../scratch/unisim_pepmd_results_{self.hparams.energy_maxiter}/{sequence}/{sequence}_model_ode50_inf10000_guidance0.05.xtc", top=f"../scratch/old_test_set/raw_data/{sequence}-traj-state0.pdb")

            # Extract positions (in nanometers, shape: (n_frames, n_atoms, 3))
            prior_samples = traj.xyz  # already a NumPy array

            prior_samples = prior_samples - prior_samples.mean(axis=1, keepdims=True)  # Centering the samples
            prior_samples = torch.tensor(prior_samples, dtype=torch.float32).view(-1, prior_samples.shape[1] * prior_samples.shape[2]).to(self.device)  # Reshape to (n_frames, n_atoms * 3)

            proposal_log_p = torch.ones(prior_samples.shape[0], device=self.device)
            proposal_samples = prior_samples.clone()

        elif self.hparams.sample_set == "md":

            path = f"../scratch/md-runner-scbg-baselines-new/data/md/{sequence}/{sequence}_310_10000"

            if not os.path.exists(f"{path}/agg_3.npz"):
                for i in range(3):
                    arrays = []
                    for j in range(10_000):
                        array = np.load(f"{path}/{j}_{i}.npz", allow_pickle=True)
                        arrays.append(array["all_positions"])
                    array = np.concatenate(arrays, axis=0)
                    np.savez(f"{path}/agg_{i}.npz", all_positions=array)

            agg_arrays = [np.load(f"{path}/agg_{i}.npz", allow_pickle=True)["all_positions"] for i in range(3)]

            output_arrays = []
            budget = self.hparams.energy_maxiter  # total time budget in fs

            usage_rate = 100  # fs per frame for agg_0

            for array in agg_arrays:
                # how many frames can we afford from this array?
                max_frames = array.shape[0]
                frames_to_take = min(max_frames, budget // usage_rate)

                output_arrays.append(array[:frames_to_take])

                # reduce remaining budget
                budget -= frames_to_take * usage_rate

                print("took ", frames_to_take, "frames from array with usage rate", usage_rate, "remaining budget", budget)

                # increase time per frame by 10× for the next array
                usage_rate *= 10
            
            # Extract positions (in nanometers, shape: (n_frames, n_atoms, 3))
            prior_samples = np.concatenate(output_arrays)
            np.random.shuffle(prior_samples)  # Shuffle the samples

            prior_samples = prior_samples - prior_samples.mean(axis=1, keepdims=True)  # Centering the samples
            prior_samples = torch.tensor(prior_samples, dtype=torch.float32).view(-1, prior_samples.shape[1] * prior_samples.shape[2]).to(self.device)  # Reshape to (n_frames, n_atoms * 3)

            proposal_log_p = torch.ones(prior_samples.shape[0], device=self.device)
            proposal_samples = prior_samples.clone()

        elif self.hparams.sample_set == "bioemu":

            # Extract positions (in nanometers, shape: (n_frames, n_atoms, 3))
            prior_samples = np.load(f"../scratch/bioemu_results/{sequence}_maxiter{self.hparams.energy_maxiter}/{sequence}_md_equil.npy")

            prior_samples = prior_samples - prior_samples.mean(axis=1, keepdims=True)  # Centering the samples
            prior_samples = torch.tensor(prior_samples, dtype=torch.float32).view(-1, prior_samples.shape[1] * prior_samples.shape[2]).to(self.device)  # Reshape to (n_frames, n_atoms * 3)

            proposal_log_p = torch.ones(prior_samples.shape[0], device=self.device)
            proposal_samples = prior_samples.clone()

        logging.info(f"Prior samples shape: {prior_samples.shape}")
        logging.info(f"Proposal samples shape: {proposal_samples.shape}")
        logging.info(f"Proposal log p shape: {proposal_log_p.shape}")

        # Compute energy
        proposal_samples_energy = energy_fn(proposal_samples)

        # Remove NaN samples
        inf_mask = torch.isinf(proposal_samples_energy)

        logging.warning(f"Removing {inf_mask.sum()} inf samples from proposal samples")

        prior_samples = prior_samples[~inf_mask]
        proposal_samples = proposal_samples[~inf_mask]
        proposal_samples_energy = proposal_samples_energy[~inf_mask]
        proposal_log_p = proposal_log_p[~inf_mask]

        if not self.hparams.dont_fix_chirality:
            symmetry_metrics, symmetry_change = resolve_chirality(
                self.datamodule.as_pointcloud(self.datamodule.unnormalize(true_samples)),
                self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples)),
                self.datamodule.topology_dict[sequence],
                prefix=prefix + "/proposal",
            )
            metrics = symmetry_metrics
            if not self.hparams.dont_fix_symmetry:
                proposal_samples = proposal_samples[~symmetry_change]
                proposal_log_p = proposal_log_p[~symmetry_change]
                proposal_samples_energy = proposal_samples_energy[~symmetry_change]

            symmetry_change = get_symmetry_change(
                self.datamodule.as_pointcloud(self.datamodule.unnormalize(true_samples)),
                self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples)),
                self.datamodule.topology_dict[sequence],
            )
            proposal_samples[symmetry_change] *= -1
        else:
            metrics = {}

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.as_pointcloud(self.datamodule.unnormalize(proposal_samples)),
            proposal_samples_energy,
        )

        reweighted_data = None
        smc_data = None

        if self.local_rank == 0:
            # log dataset metrics
            metrics.update(
                self.datamodule.metrics_and_plots(
                    self.log_image,
                    sequence,
                    true_data,
                    proposal_data,
                    reweighted_data,
                    smc_data,
                    prefix=prefix,
                )
            )
        else:
            metrics = {}
        return metrics

    def on_train_epoch_start(self) -> None:
        logging.info("Train epoch start")
        self.train_metrics.reset()

    def on_validation_epoch_start(self) -> None:
        logging.info("Validation epoch start")
        self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        logging.info("Test epoch start")
        self.test_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()
        logging.info("Train epoch end")

    def on_validation_epoch_end(self):
        self.on_eval_epoch_end(self.val_metrics, "val")
        logging.info("Validation epoch end")

    def on_test_epoch_end(self) -> None:
        self.on_eval_epoch_end(self.test_metrics, "test")
        logging.info("Test epoch end")

    def on_after_backward(self) -> None:
        valid_gradients = True
        flat_grads = torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])
        global_norm = torch.norm(flat_grads, p=2)
        for _name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())

                if not valid_gradients:
                    break

        self.log("global_gradient_norm", global_norm, on_step=True, prog_bar=True)
        if not valid_gradients:
            logger.warning("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()
            return

    # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    def on_before_optimizer_step(self, optimizer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in self.trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        self.log_dict({"train/grad_norm": total_norm}, prog_bar=True)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.net, EMA):
            self.net.update_ema()

    def proposal_energy(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        raise NotImplementedError


if __name__ == "__main__":
    _ = TransferableBoltzmannGeneratorLitModule(None, None, None, None)
