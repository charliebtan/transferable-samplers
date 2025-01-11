import math
from typing import Any, Dict, Tuple

import torch
from torchdyn.core import NeuralODE

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule
from src.models.components.wrappers import TorchdynWrapper


class ShortcutLitModule(BoltzmannGeneratorLitModule):
    """

    TODO - Add a description.

    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        jarzynski_batch_size: int,  # TODO bit weird this is here but main generation done by data module
        sigma: float = 0.0,
        M: int = 128,
        bootstrap_every: int = 8,
        sampling_d_base: int = None,
    ) -> None:
        """Initialize a `ProposalFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(net, optimizer, scheduler, compile)

    def forward(self, t: torch.Tensor, x: torch.Tensor, d_base: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net(t, x, d_base=d_base.to(x.device))

    def get_xt(self, x0, x1, t):
        t = t.view(-1, 1)

        mu_t = (1.0 - t) * x0 + t * x1

        if not self.hparams.sigma == 0.0:
            noise = self.prior.sample(x1.shape[0]).to(x1.device)
            xt = mu_t + self.hparams.sigma * noise
        else:
            xt = mu_t

        return xt

    def get_flow_targets(self, x0, x1):
        vt_flow = x1 - x0
        return vt_flow

    def sample_bootstrap_d_base(self, batch_size, device):
        self.log2_sections = int(math.log2(self.hparams.M))

        assert (
            batch_size > self.log2_sections
        ), "known issue: batch_size must be greater than M otherwise only returns zeros"

        dt_range = torch.arange(
            self.log2_sections, device=device, dtype=torch.int32
        )  # [0, 1, 2, ..., log2_sections-1]
        dt_array = self.log2_sections - 1 - dt_range  # e.g. [log2_sections-1, ..., 0]
        repeated = dt_array.repeat(batch_size // self.log2_sections)
        needed = batch_size - repeated.shape[0]
        dt_base = torch.cat(
            [repeated, torch.zeros(needed, device=device, dtype=torch.int32)], dim=0
        )

        d_base = dt_base.to(torch.float32)  # cast to float to match usage below

        return d_base

    def sample_t(self, d_base):
        # time has to be sampled carefully to align with the discretization

        batch_size = d_base.shape[0]
        device = d_base.device

        d_sections = 2.0**d_base

        # We want to sample t ~ Uniform{0, dt_sections[i]} (integer), then divide by dt_sections[i].
        # This is somewhat trickier to do in a single vectorized call in PyTorch
        # (because each batch element can have a different max).
        # We'll do it in a loop for clarity:

        t_list = []
        for i in range(batch_size):
            maxval = int(d_sections[i].item())  # dt_sections[i] is float, convert to int
            # If maxval == 0 for some reason, clamp to 1 to avoid errors
            if maxval < 1:
                maxval = 1
            # Sample an integer in [0, maxval)
            t_i = torch.randint(low=0, high=maxval, size=(1,), device=device, dtype=torch.int64)
            t_list.append(t_i)

        t = torch.cat(t_list, dim=0).to(torch.float32)  # shape [bootstrap_batchsize]
        t = t / d_sections  # elementwise scale to [0, 1]

        return t

    def get_bootstrap_targets(self, xt, t, d_base):
        t = t.view(-1, 1)
        d_base = d_base.view(-1, 1)

        # d_base is log2(1/d) where
        # d is the size of the step you want the target for
        # so the target is generated with two steps of size d/2
        d_half_base = d_base + 1
        d_half = 1 / 2.0**d_half_base

        with torch.no_grad():
            vb1 = self.forward(t, xt, d_half_base)

        t2 = t + d_half
        xt2 = xt + d_half * vb1
        xt2 = torch.clamp(xt2, -4.0, 4.0)

        assert xt2.shape == xt.shape, "xt2 shape not as expected, check for broadcasting errors"

        with torch.no_grad():
            vb2 = self.forward(t2, xt2, d_half_base)

        vt_bst = 0.5 * (vb1 + vb2)
        vt_bst = torch.clamp(vt_bst, -4.0, 4.0)

        return vt_bst

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        batch = batch[:64]

        bst_batch_size = batch.shape[0] // self.hparams.bootstrap_every
        flow_batch_size = batch.shape[0] - bst_batch_size

        # sample d_base (e.g 3 for d=1/(2**3) for each bootstrap batch element)
        # where d is the size of the step you want the target for
        d_base = self.sample_bootstrap_d_base(bst_batch_size, batch.device)

        # concat log2(M) (e.g 7 for d=1/(2**7) = d/128 to d_base
        # corresponding to non-shortcut steps / non bootstrap batch elements
        d_base = torch.cat(
            [
                d_base,
                torch.ones([flow_batch_size], device=batch.device) * math.log2(self.hparams.M),
            ]
        )

        # sample t for both flow and bootstrap batch elements
        t = self.sample_t(d_base)

        # samples prior and xt for both flow and bootstrap batch elements
        batch_prior = self.prior.sample(batch.shape[0]).to(batch.device)
        xt = self.get_xt(batch_prior, batch, t)

        # get the targets for the flow batch elements
        vt_flow = self.get_flow_targets(batch_prior[-flow_batch_size:], batch[-flow_batch_size:])

        # get the targets for the bootstrap batch elements
        vt_bst = self.get_bootstrap_targets(
            xt[:bst_batch_size], t[:bst_batch_size], d_base[:bst_batch_size]
        )

        assert torch.all(
            d_base[-flow_batch_size:] == math.log2(self.hparams.M)
        ), "d_base not as expected, SHOULD be all log2(M) for flow batch elements, there is probably an error in the batch slicing"

        # get network output
        vt_pred = self.forward(t, xt, d_base)

        # comptute losses on slices of network output
        loss_flow = self.criterion(vt_pred[-flow_batch_size:], vt_flow)
        loss_bst = self.criterion(vt_pred[:bst_batch_size], vt_bst)

        return loss_flow + loss_bst

    def flow(self, x: torch.Tensor, reverse=False) -> torch.Tensor:
        n_timesteps = 2**self.hparams.sampling_d_base

        dlog_p_init = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = (
            torch.linspace(1, 0, n_timesteps + 1)
            if reverse
            else torch.linspace(0, 1, n_timesteps + 1)
        )

        d_base = torch.tensor(
            [self.hparams.sampling_d_base], device=x.device
        )  # batch dims required by EGNN architecture

        node = NeuralODE(TorchdynWrapper(self.net, d_base=d_base), solver="euler")

        traj = node.trajectory(
            torch.cat([x, dlog_p_init], dim=-1),
            t_span=t_span,
        )

        dlog_p = traj[-1][..., -1]
        x = traj[-1][..., :-1]

        return x, dlog_p

    def generate_samples(
        self,
        batch_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        assert (
            self.hparams.sampling_d_base is not None
        ), "sampling_d must be set to generate samples"

        prior_samples = self.prior.sample(batch_size).to(device)
        prior_log_p = -self.prior.energy(prior_samples)

        with torch.no_grad():
            samples, dlog_p = self.flow(prior_samples, reverse=False)

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p, prior_samples


if __name__ == "__main__":
    _ = ShortcutLitModule(None, None, None, None)
