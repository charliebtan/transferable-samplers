import math
from typing import Any, Dict, Tuple

import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm

from src.models.components.wrappers import torchdyn_wrapper
from src.models.proposal_flow_module import ProposalFlowLitModule
from src.utils.dw4_plots import TARGET
from src.utils.tbg_utils import kish_effective_sample_size


class FlowMatchLitModule(ProposalFlowLitModule):
    """

    TODO - Add a description.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `ProposalFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, compile)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net(t, x)

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        x1 = batch
        x0 = self.prior.sample(x1.shape[0]).to(x1.device)
        t = torch.rand(
            x1.shape[0], 1, device=x1.device
        )  # should this be generated here or elsewhere?

        xt = (1.0 - t) * x0 + t * x1
        vt_ref = x1 - x0

        vt_pred = self.forward(t, xt)
        loss = self.criterion(vt_pred, vt_ref)
        return loss

    def flow(self, x: torch.Tensor, reverse=False) -> torch.Tensor:
        dlog_p_init = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        node = NeuralODE(
            torchdyn_wrapper(self.net),
            atol=1e-4,
            rtol=1e-4,
            solver="dopri5",
            sensitivity="adjoint",
        )

        traj = node.trajectory(
            torch.cat([x, dlog_p_init], dim=-1),
            t_span=t_span,
        )

        dlog_p = traj[-1][..., -1]
        x = traj[-1][..., :-1]

        return x, dlog_p

    def generate_samples(
        self, batch_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        prior_samples = self.prior.sample(batch_size).to(device)
        prior_log_p = -self.prior.energy(prior_samples)

        with torch.no_grad():
            samples, dlog_p = self.flow(prior_samples, reverse=False)

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p, prior_samples

    def prior_energy_from_proposal(self, x: torch.Tensor) -> torch.Tensor:
        # x is considered to be a sample from the proposal distribution
        x0, dlogp = self.flow(x, reverse=True)
        return -(-self.prior.energy(x0).view(-1) - dlogp.view(-1))

    def linear_energy_interpolation(self, x, t):
        energy = (1 - t) * self.prior_energy_from_proposal(x) + t * TARGET.energy(x).view(-1)
        assert energy.shape == (
            x.shape[0],
        ), "Energy should be a flat vector, one value per sample"
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
    def jarzyinski_process(self, samples_proposal, log_p_proposal):
        # TODO I think I should test with a simple energy function and make sure I am getting the correct energies etc

        X = samples_proposal

        eps = 0.001
        num_timesteps = 100  # TODO should default to 1000

        A = torch.zeros(X.shape[0], device=X.device)  # the jarzynski weights

        timesteps = torch.linspace(0, 1, num_timesteps + 1)
        dt = 1 / num_timesteps

        A_list = [A]
        ESS_list = []

        for j, t in tqdm(enumerate(timesteps[:-1])):
            # get the energy gradients
            energy_grad_x, energy_grad_t = self.linear_energy_interpolation_gradients(X, t)

            # compute the updates
            dX_t = -eps * energy_grad_x * dt + math.sqrt(2 * eps * dt) * torch.randn_like(X)
            dA_t = -energy_grad_t * dt

            assert dX_t.shape == X.shape, "dX_t should have the same shape as X"
            assert dA_t.shape == A.shape, "dA_t should have the same shape as A"

            # apply the updates
            X = X + dX_t
            A = A + dA_t

            A_list.append(A)
            ESS = kish_effective_sample_size(torch.softmax(A, dim=-1)).item() / len(A)
            ESS_list.append(ESS)

            if ESS < -1:
                # qmc_rand = sampler.random(n=len(A))
                # cum_prob = torch.cumsum(torch.softmax(A, dim=-1), dim=0)
                # indexes = np.searchsorted(cum_prob, qmc_rand, side="left").flatten()
                indexes = torch.multinomial(torch.softmax(A, dim=-1), len(A), replacement=True)
                X = X[indexes]
                A = torch.zeros_like(A)
            if j % 1000 == 0:
                pass
                # print("energy", j, target_energy(X))

        jarzynski_samples = X
        jarzynski_weights = torch.softmax(A, dim=-1)

        return jarzynski_samples, jarzynski_weights


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
