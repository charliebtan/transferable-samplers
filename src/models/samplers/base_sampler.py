import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from tqdm import tqdm

from src.evaluation.metrics.ess import sampling_efficiency


class SMCSampler(torch.nn.Module):
    def __init__(
        self,
        log_image_fn: callable = None,
        batch_size: int = 128,
        langevin_eps: float = 1e-7,
        num_timesteps: int = 100,
        ess_threshold: float = -1.0,
        systematic_resampling: bool = False,
        adaptive_step_size: bool = False,
        warmup: float = 0.1,
        enabled: bool = False,
        do_energy_plots: bool = False,
        log_freq: int = 10,
        input_energy_cutoff: float = None,
    ):
        super().__init__()

        self.log_image_fn = log_image_fn
        self.batch_size = batch_size
        self.langevin_eps = langevin_eps
        self.num_timesteps = num_timesteps
        self.ess_threshold = ess_threshold
        self.warmup = warmup
        self.enabled = enabled
        self.do_energy_plots = do_energy_plots
        self.log_freq = log_freq
        self.input_energy_cutoff = input_energy_cutoff
        self.systematic_resampling = systematic_resampling
        self.adaptive_step_size = adaptive_step_size

    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        raise NotImplementedError

    def init_timesteps(self):
        return torch.linspace(0, 1, self.num_timesteps + 1)

    def langevin_eps_fn(self, t):
        if t < self.warmup:
            return (self.langevin_eps * t) / self.warmup
        else:
            return self.langevin_eps

    def update_step_size(self, acceptance_rate):
        if acceptance_rate > 0.6:
            self.langevin_eps = self.langevin_eps * 1.1
        elif acceptance_rate < 0.55:
            self.langevin_eps = self.langevin_eps / 1.1

    def linear_energy_interpolation(self, source_energy, target_energy, t, x):
        E_source = source_energy(x)
        E_target = target_energy(x)

        assert E_source.shape == (x.shape[0],), f"Source energy should be a flat vector not {E_source.shape}"
        assert E_target.shape == (x.shape[0],), f"Target energy should be a flat vector, not {E_target.shape}"
        energy = (1 - t) * E_source + t * E_target
        return energy

    def linear_energy_interpolation_gradients(self, source_energy, target_energy, t, x):
        t = t.repeat(x.shape[0]).to(x)

        with torch.set_grad_enabled(True):
            x.requires_grad = True
            t.requires_grad = True

            et = self.linear_energy_interpolation(source_energy, target_energy, t, x)

            # assert et.requires_grad, "et should require grad - check the energy function for no_grad"

            # this is a bit hacky but is fine as long as
            # the energy function is defined properly and
            # doesn't mix batch items
            x_grad = torch.autograd.grad(et.sum(), x)[0]

            assert x_grad.shape == x.shape, "x_grad should have the same shape as x"

        assert x_grad is not None, "x_grad should not be None"

        return x_grad.detach()

    def plot_stepwise_energy(self, target_energy_list, interpolation_energy_list, t_list):
        stepwise_target_energy_np = np.stack(target_energy_list)
        stepwise_interpolation_energy_np = np.stack(interpolation_energy_list)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        for k in range(stepwise_target_energy_np.shape[1]):
            axs[0].plot(t_list, stepwise_target_energy_np[:, k], linewidth=1, alpha=0.5)
            axs[1].plot(t_list, stepwise_interpolation_energy_np[:, k], linewidth=1, alpha=0.5)

        axs[0].set_xlabel("Time", fontsize=12)
        axs[0].set_ylabel("Target energy", fontsize=12)

        axs[1].set_xlabel("Time", fontsize=12)
        axs[1].set_ylabel("Interpolation energy", fontsize=12)

        plt.tight_layout()
        self.log_image_fn(fig, "langevin/energies")
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
        self.log_image_fn(fig, "langevin/energy_histograms")
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
        self.log_image_fn(fig, "langevin/weights")
        plt.close()

    def plot_eps(self, eps_list, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, eps_list, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Eps", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/eps")
        plt.close()

    def plot_acceptance_rate(self, acceptance_rate_list, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, acceptance_rate_list, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Acceptance Rate", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/acceptance-rate")
        plt.close()

    def plot_particle_survival(self, survived_linages, t_list):
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        ax.plot(t_list, survived_linages, linewidth=1, alpha=0.5)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Survived Linages (%)", fontsize=12)
        plt.tight_layout()
        self.log_image_fn(fig, "langevin/linage-survived")
        plt.close()

    @torch.no_grad()
    def resample(self, x, logw):
        if self.systematic_resampling:
            # systematic resampling
            N = len(logw)
            w = torch.softmax(logw, dim=-1)
            c = torch.cumsum(w, dim=0)
            u = torch.rand(1, device=x.device) / N
            indexes = torch.searchsorted(c, u + torch.arange(N, device=x.device) / N)
        else:
            # smc weights
            w = torch.softmax(logw, dim=-1)
            # multinomial resampling
            indexes = torch.multinomial(w, len(x), replacement=True)

        return x[indexes], indexes

    @torch.no_grad()
    def sample(self, proposal_samples, source_energy, target_energy):
        if not self.enabled:
            return None, None

        # Filter samples based on target energy cutoff
        if self.input_energy_cutoff is not None:
            proposal_samples_energy = target_energy(proposal_samples)
            proposal_samples = proposal_samples[proposal_samples_energy < self.input_energy_cutoff]
            logging.info("Clipping energies")

        num_timesteps = self.num_timesteps
        timesteps = self.init_timesteps()

        X = proposal_samples
        A = torch.ones(X.shape[0], device=X.device)  # the smc weights

        A_list = [A]
        ESS_list = [1.0]
        t_list = [timesteps[0]]
        eps_list = [self.langevin_eps_fn(0.0)]
        acceptance_rate_list = [torch.tensor(1.0)]
        survived_linages = [torch.tensor(1.0)]

        if self.do_energy_plots:
            # slice into list of batches (tensors)
            X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]

            target_energy_list = [np.concatenate([target_energy(X_batch).cpu() for X_batch in X_batches])]
            interpolation_energy_list = [
                np.concatenate(
                    [
                        self.linear_energy_interpolation(source_energy, target_energy, timesteps[0], X_batches[i]).cpu()
                        for i in range(len(X_batches))
                    ]
                )
            ]

        t_previous = 0.0
        particle_ids = torch.arange(X.shape[0])
        for j, t in tqdm(enumerate(timesteps[:-1])):
            logging.info(f"Outer loop iteration {j}")

            # slice into list of batches (tensors)
            X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
            A_batches = [A[i : i + self.batch_size] for i in range(0, A.shape[0], self.batch_size)]

            target_energy_batches = []
            interpolation_energy_batches = []
            batch_acceptance_rate_list = []

            dt = t - t_previous
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):
                if X_batch.isnan().any():
                    raise ValueError("X contains NaNs")

                # Update coordinates and weights according to mcmc kernel
                X_batch, A_batch, acceptance_rate = self.mcmc_kernel(
                    source_energy=source_energy, target_energy=target_energy, t=t, x=X_batch, logw=A_batch, dt=dt
                )

                # apply the updates to the batch in the list
                X_batches[batch_idx] = X_batch
                A_batches[batch_idx] = A_batch

                if self.do_energy_plots:
                    target_energy_batches.append(target_energy(X_batch).cpu())
                    interpolation_energy_batches.append(
                        self.linear_energy_interpolation(source_energy, target_energy, t, X_batch).cpu()
                    )

                batch_acceptance_rate_list.append(acceptance_rate.view(-1))

            # cat the batches to compute global statistics
            X = torch.cat(X_batches, dim=0)
            A = torch.cat(A_batches, dim=0)
            acceptance_rate = torch.cat(batch_acceptance_rate_list, dim=0).mean()

            if self.adaptive_step_size:
                self.update_step_size(acceptance_rate)

            assert A.dim() == 1, "A should be a flat vector"

            if X.isnan().any() or A.isnan().any() or not (j + 1) % self.log_freq or j + 1 == num_timesteps:
                if self.do_energy_plots:
                    self.plot_stepwise_energy(target_energy_list, interpolation_energy_list, t_list)
                    self.plot_stepwise_energy_hist(target_energy_list, interpolation_energy_list, t_list)

                self.plot_weights(A_list, ESS_list, t_list)
                self.plot_eps(eps_list, t_list)
                self.plot_acceptance_rate(acceptance_rate_list, t_list)
                self.plot_particle_survival(survived_linages, t_list)

            if X.isnan().any():
                raise ValueError("X has NaNs")
            elif A.isnan().any():
                raise ValueError("A has NaNs")

            A_list.append(A)
            ESS = sampling_efficiency(A)
            ESS_list.append(ESS.cpu())
            acceptance_rate_list.append(acceptance_rate.cpu())
            unique_ratio = particle_ids.unique().numel() / len(particle_ids)
            survived_linages.append(unique_ratio)

            t_list.append(t)
            # log epsilon step size from langevin
            eps = self.langevin_eps_fn(t)
            eps_list.append(eps)

            if self.do_energy_plots:
                target_energy_list.append(np.concatenate(target_energy_batches))
                interpolation_energy_list.append(np.concatenate(interpolation_energy_batches))

            if ESS < self.ess_threshold and not j + 1 == num_timesteps:
                # qmc_rand = sampler.random(n=len(A))
                # cum_prob = torch.cumsum(torch.softmax(A, dim=-1), dim=0)
                # indexes = np.searchsorted(cum_prob, qmc_rand, side="left").flatten()
                X, indexes = self.resample(x=X, logw=A)
                A = torch.ones_like(A)
                logging.info(f"resampling @ step {j}")

                particle_ids = particle_ids[indexes.cpu()]
                A_list.append(A)
                ESS = sampling_efficiency(A)
                ESS_list.append(ESS.cpu())

                t_list.append(t + 1e-9)
                eps_list.append(eps)
                acceptance_rate_list.append(acceptance_rate.cpu())
                unique_ratio = particle_ids.unique().numel() / len(particle_ids)
                survived_linages.append(unique_ratio)

                # slice into list of batches (tensors)
                X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]

                if self.do_energy_plots:
                    target_energy_list.append(np.concatenate([target_energy(X_batch).cpu() for X_batch in X_batches]))
                    interpolation_energy_list.append(
                        np.concatenate(
                            [
                                self.linear_energy_interpolation(source_energy, target_energy, t + 1e-9, X_batch).cpu()
                                for X_batch in X_batches
                            ]
                        )
                    )

            t_previous = t

        # Final resampling
        X, indexes = self.resample(x=X, logw=A)
        particle_ids = particle_ids[indexes.cpu()]
        unique_ratio = particle_ids.unique().numel() / len(particle_ids)
        logging.info(f"resampling @ step {j}")
        logging.info(f"Fraction of Original Samples: {unique_ratio} %")

        smc_samples = X
        smc_logits = A
        assert smc_samples.shape == proposal_samples.shape, "shape mismatch"
        assert smc_logits.dim() == 1, "smc_weights should be a flat vector"
        return smc_samples, smc_logits
