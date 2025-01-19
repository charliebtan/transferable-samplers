import math

import torch
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
        ess_threshold: float = 0.5,
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

    def linear_energy_interpolation(self, x, t):
        source_energy = self.source_energy(x)
        target_energy = self.target_energy(x)
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
        # TODO I think I should test with a simple energy function and make
        # sure I am getting the correct energies etc
        num_timesteps = self.num_timesteps
        eps = self.langevin_eps

        X = samples_proposal
        A = torch.zeros(X.shape[0], device=X.device)  # the jarzynski weights

        timesteps = torch.linspace(0.0, 1, num_timesteps + 1)
        dt = 1 / num_timesteps

        A_list = [A]
        ESS_list = []

        # slice into list of batches (tensors)
        X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
        A_batches = [A[i : i + self.batch_size] for i in range(0, A.shape[0], self.batch_size)]

        for j, t in tqdm(enumerate(timesteps[:-1])):
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):
                # get the energy gradients
                energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(
                    X_batch, t
                )

                # compute the updates
                dX_t = -eps * energy_grad_x * dt + math.sqrt(2 * eps * dt) * torch.randn_like(
                    X_batch
                )
                dA_t = -energy_grad_t * dt
                assert dX_t.shape == X_batch.shape, "dX_t should have the same shape as X_batch"
                assert dA_t.shape == A_batch.shape, "dA_t should have the same shape as A_batch"

                # apply the updates to the batch in the list
                X_batches[batch_idx] = X_batch + dX_t
                A_batches[batch_idx] = A_batch + dA_t

                if X_batches[batch_idx].isnan().any():
                    raise ValueError("X_batch has NaNs")
                if A_batches[batch_idx].isnan().any():
                    raise ValueError("A_batch has NaNs")

            # cat the batches to compute global statistics
            X = torch.cat(X_batches, dim=0)
            A = torch.cat(A_batches, dim=0)

            assert A.dim() == 1, "A should be a flat vector"
            jarzynski_weights = torch.softmax(A, dim=-1)
            if j % 100 == 0:
                print("energy", j, self.target_energy(X).mean())

            A_list.append(A)
            ESS = sampling_efficiency(A)
            ESS_list.append(ESS)

            if ESS < self.ess_threshold:
                # qmc_rand = sampler.random(n=len(A))
                # cum_prob = torch.cumsum(torch.softmax(A, dim=-1), dim=0)
                # indexes = np.searchsorted(cum_prob, qmc_rand, side="left").flatten()
                indexes = torch.multinomial(jarzynski_weights, len(A), replacement=True)
                X = X[indexes]
                A = torch.zeros_like(A)
            if j % 1000 == 0:
                pass
                # print("energy", j, target_energy(X))

        jarzynski_samples = X
        jarzynski_weights = torch.softmax(A, dim=-1)
        assert jarzynski_samples.shape == samples_proposal.shape, "shape mismatch"
        assert jarzynski_weights.dim() == 1, "jarzynski_weights should be a flat vector"
        return jarzynski_samples, jarzynski_weights
