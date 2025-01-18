import math

import torch
from src.utils.tbg_utils import sampling_efficiency
from tqdm import tqdm
from src.models.components.jarzynski_sampler import JarzynskiSampler


class FastJarzynskiSampler(JarzynskiSampler):
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

        timesteps = torch.linspace(0, 1, num_timesteps + 1)
        dt = 1 / num_timesteps

        A_list = [A]
        ESS_list = []

        # slice into list of batches (tensors)
        X_batches = [X[i : i + self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
        A_batches = [A[i : i + self.batch_size] for i in range(0, A.shape[0], self.batch_size)]

        for j, t in tqdm(enumerate(timesteps[:-1])):
            for batch_idx, (X_batch, A_batch) in enumerate(zip(X_batches, A_batches)):
                # get the energy gradients
                x = X_batch
                with torch.enable_grad():
                    x.requires_grad = True
                    target_energy = self.target_energy(x)
                    x_grad = torch.autograd.grad(target_energy.sum(), x)[0].detach()
                # compute the updates
                dX_t = -eps * x_grad * dt + math.sqrt(
                    2 * eps * dt
                ) * torch.randn_like(X_batch)
                # This is a hack
                dA_t = torch.zeros_like(A_batch)

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
