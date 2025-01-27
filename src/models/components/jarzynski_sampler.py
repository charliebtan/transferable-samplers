import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

from src.utils.tbg_utils import sampling_efficiency


class JarzynskiSampler(torch.nn.Module):
    def __init__(
        self,
        source_energy,
        target_energy,
        batch_size: int = 1000,
        langevin_eps: float = 0.4,
        num_timesteps: int = 1000,
        ess_threshold: float = -1.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.source_energy = source_energy
        self.target_energy = target_energy
        self.batch_size = batch_size
        self.langevin_eps = langevin_eps
        self.num_timesteps = num_timesteps
        self.ess_threshold = ess_threshold
        self.enabled = enabled

    def plot_stepwise_energy(self, target_energy_list, interpolation_energy_list, t_list):

        stepwise_target_energy_np = np.stack(target_energy_list)
        stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for k in range(stepwise_target_energy_np.shape[1]):

            axs[0].plot(t_list, stepwise_target_energy_np[:, k], linewidth=1, alpha=0.5)
            axs[1].plot(
                t_list, stepwise_interpolation_energy_np[:, k], linewidth=1, alpha=0.5
            )

        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Target energy", fontsize=12)

        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("Interpolation energy", fontsize=12)

        plt.tight_layout()
        self.wandb_logger.log_image(f"langevin/energies", [fig])
        plt.close()

    def plot_stepwise_energy_hist(self, target_energy_list, interpolation_energy_list, t_list):

        stepwise_target_energy_np = np.stack(target_energy_list)
        stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)
        t_np = np.array(t_list)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        data = stepwise_target_energy_np
        bins = np.linspace(data.min(), data.max(), 100)
        histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
        histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
        extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
        im = axs[0].imshow(
            histograms_normalized.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(
                vmin=histograms_normalized[histograms_normalized > 0].min(),
                vmax=histograms_normalized.max(),
            ),
            cmap="inferno",
        )

        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Target energy", fontsize=12)
        fig.colorbar(im, ax=axs[0], label="Log Marginal Density")

        data = stepwise_interpolation_energy_np
        bins = np.linspace(data.min(), data.max(), 100)
        histograms = np.array([np.histogram(row, bins=bins)[0] for row in data])
        histograms_normalized = histograms / histograms.sum(axis=1, keepdims=True)
        extent = [t_np.min(), t_np.max(), bins[0], bins[-1]]
        im = axs[1].imshow(
            histograms_normalized.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(
                vmin=histograms_normalized[histograms_normalized > 0].min(),
                vmax=histograms_normalized.max(),
            ),
            cmap="inferno",
        )

        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("Interpolation energy", fontsize=12)
        fig.colorbar(im, ax=axs[1], label="Log Marginal Density")

        plt.tight_layout()
        self.wandb_logger.log_image(f"langevin/energy_histograms", [fig])
        plt.close()

    def plot_weights(self, A_list, ESS_list, t_list):

        A_np = torch.stack(A_list).cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        for k in range(A_np.shape[1]):
            axs[0].plot(t_list, A_np[:, k], linewidth=1, alpha=0.5)
        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("A", fontsize=12)

        axs[1].plot(t_list, ESS_list, linewidth=1, alpha=0.3)
        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("ESS", fontsize=12)
        axs[1].set_yscale("log")

        plt.tight_layout()
        self.wandb_logger.log_image(f"langevin/weights", [fig])
        plt.close()

    def plot_dX_t_norm(self, dX_t_norm_list, eps_list, t_list):
        

        dX_t_norm_np = np.stack(dX_t_norm_list).T

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, eps_list, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Eps", fontsize=12)
        plt.tight_layout()
        self.wandb_logger.log_image(f"langevin/eps", [fig])
        plt.close()

        fig, axs = plt.subplots(1, 1, figsize=(7.5, 5))

        for k in range(dX_t_norm_np.shape[1]):

            axs.plot(t_list, dX_t_norm_np[k], linewidth=1, alpha=0.5)

        axs.set_xlabel("Time", fontsize=12)
        axs.set_ylabel("||dX_t||", fontsize=12)
        plt.tight_layout()
        self.wandb_logger.log_image(f"langevin/dX_t_norm", [fig])
        plt.close()

    def linear_energy_interpolation(self, x, t):
        source_energy = self.source_energy(x)
        target_energy = self.target_energy(x)
        target_energy = target_energy.reshape(-1)
        assert source_energy.shape == (
            x.shape[0],
        ), f"Source energy should be a flat vector not {source_energy.shape}"
        assert target_energy.shape == (
            x.shape[0],
        ), f"Target energy should be a flat vector, not {target_energy.shape}"
        energy = (1 - t) * source_energy + t * target_energy
        return energy

    def linear_energy_interpolation_gradients(self, x, t):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True

            et = self.linear_energy_interpolation(x, t)

            assert (
                et.requires_grad
            ), "et should require grad - check the energy function for no_grad"

            # this is a bit hacky but is fine as long as
            # the energy function is defined properly and
            # doesn't mix batch items

            x_grad, t_grad = torch.autograd.grad(et.sum(), (x, t))

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"
            assert t_grad.shape == t.shape, "t_grad should have the same shape as t"

        assert x_grad is not None, "x_grad should not be None"
        assert t_grad is not None, "t_grad should not be None"

        return x_grad, t_grad

    @torch.no_grad()
    def sample(self, samples_proposal):
        if not self.enabled:
            return None, None

        num_timesteps = self.num_timesteps
        timesteps = torch.linspace(0, 1, num_timesteps + 1)

        def eps_fn(t, warmup=0.1):
            if t < warmup:
                return self.langevin_eps * t / warmup
            else:
                return self.langevin_eps

        X = samples_proposal
        A = torch.zeros(X.shape[0], device=X.device)  # the jarzynski weights

        A_list = [A]
        ESS_list = [1.0]
        t_list = [timesteps[0]]
        eps_list = [eps_fn(0.0)]

        # slice into list of batches (tensors)
        X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]

        target_energy_list = [np.concatenate([self.target_energy(X_batch).cpu() for X_batch in X_batches])]
        interpolation_energy_list = [np.concatenate([self.linear_energy_interpolation(X_batch, timesteps[0]).cpu() for X_batch in X_batches])]
        dX_t_norm_list = [torch.zeros(X.shape[0])]

        t_previous = 0.0

        for j, t in tqdm(enumerate(timesteps[:-1])):

            # slice into list of batches (tensors)
            X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
            A_batches = [A[i : i + self.batch_size] for i in range(0, A.shape[0], self.batch_size)]

            dX_t_norm_batches = []
            target_energy_batches = []
            interpolation_energy_batches = []

            dt = t - t_previous
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):

                eps = eps_fn(t)

                # get the energy gradients
                energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(
                    X_batch, t
                )

                # assert torch.allclose(energy_grad_t, - self.source_energy(X_batch) + self.target_energy(X_batch))

                # compute the updates
                dX_t = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(X_batch)

                dA_t = -energy_grad_t * dt

                assert dX_t.shape == X_batch.shape, "dX_t should have the same shape as X_batch"
                assert dA_t.shape == A_batch.shape, "dA_t should have the same shape as A_batch"

                # apply the updates to the batch in the list
                X_batches[batch_idx] = X_batch + dX_t
                A_batches[batch_idx] = A_batch + dA_t

                dX_t_norm_batches.append(dX_t.norm(dim=-1).cpu())
                target_energy_batches.append(self.target_energy(X_batch).cpu())
                interpolation_energy_batches.append(self.linear_energy_interpolation(X_batch, t).cpu())

            # cat the batches to compute global statistics
            X = torch.cat(X_batches, dim=0)
            A = torch.cat(A_batches, dim=0)

            assert A.dim() == 1, "A should be a flat vector"
            jarzynski_weights = torch.softmax(A, dim=-1)

            if X.isnan().any() or A.isnan().any() or not (j + 1) % 100 or j + 1 == num_timesteps:

                self.plot_stepwise_energy(target_energy_list, interpolation_energy_list, t_list)
                self.plot_stepwise_energy_hist(target_energy_list, interpolation_energy_list, t_list)
                self.plot_weights(A_list, ESS_list, t_list)
                self.plot_dX_t_norm(dX_t_norm_list, eps_list, t_list)

            if X.isnan().any():
                raise ValueError("X has NaNs")
            elif A.isnan().any():
                raise ValueError("A has NaNs")

            A_list.append(A)
            ESS = sampling_efficiency(A)
            ESS_list.append(ESS.cpu())

            t_list.append(t)
            eps_list.append(eps)
            dX_t_norm_list.append(np.concatenate(dX_t_norm_batches))

            target_energy_list.append(np.concatenate(target_energy_batches))
            interpolation_energy_list.append(np.concatenate(interpolation_energy_batches))

            if ESS < self.ess_threshold:
                # qmc_rand = sampler.random(n=len(A))
                # cum_prob = torch.cumsum(torch.softmax(A, dim=-1), dim=0)
                # indexes = np.searchsorted(cum_prob, qmc_rand, side="left").flatten()
                indexes = torch.multinomial(jarzynski_weights, len(A), replacement=True)
                X = X[indexes]
                A = torch.zeros_like(A)
                logging.info(f"resampling @ step {j}")

                A_list.append(A)
                ESS = sampling_efficiency(A)
                ESS_list.append(ESS.cpu())

                t_list.append(t + 1e-9)
                eps_list.append(eps)
                dX_t_norm_list.append(np.concatenate(dX_t_norm_batches))

                # slice into list of batches (tensors)
                X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]

                target_energy_list.append(np.concatenate([self.target_energy(X_batch).cpu() for X_batch in X_batches]))
                interpolation_energy_list.append(np.concatenate([self.linear_energy_interpolation(X_batch, t+1e-9).cpu() for X_batch in X_batches]))

            t_previous = t

        jarzynski_samples = X
        jarzynski_logits = A
        assert jarzynski_samples.shape == samples_proposal.shape, "shape mismatch"
        assert jarzynski_weights.dim() == 1, "jarzynski_weights should be a flat vector"
        return jarzynski_samples, jarzynski_logits
