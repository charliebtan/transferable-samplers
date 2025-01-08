from typing import Any, Dict, Tuple

import torch
from torchdyn.core import NeuralODE

from src.models.components.wrappers import cnf_wrapper, torchdyn_wrapper
from src.models.proposal_flow_module import ProposalFlowLitModule


class ShortcutLitModule(ProposalFlowLitModule):
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

    def get_targets(self, samples_data, force_t=None, force_dt=None):
        device = samples_data.device

        samples_prior = self.prior.sample((samples_data.shape[0],)).to(device)
        batch_size = samples_data.shape[0]

        # -----------------------------------------------------------------
        #  1) =========== Sample dt. ============
        # -----------------------------------------------------------------
        bootstrap_batchsize = batch_size // self.hparams.bootstrap_every
        log2_sections = int(torch.log2(self.hparams.M))

        dt_range = torch.arange(
            log2_sections, device=device, dtype=torch.int32
        )  # [0, 1, 2, ..., log2_sections-1]
        dt_array = log2_sections - 1 - dt_range  # e.g. [log2_sections-1, ..., 0]
        repeated = dt_array.repeat(bootstrap_batchsize // log2_sections)
        needed = bootstrap_batchsize - repeated.shape[0]
        dt_base = torch.cat(
            [repeated, torch.zeros(needed, device=device, dtype=torch.int32)], dim=0
        )

        force_dt_vec = (
            torch.ones(bootstrap_batchsize, device=device, dtype=torch.float32) * force_dt
        )
        dt_base = dt_base.to(torch.float32)  # cast to float to match usage below
        dt_base = torch.where(
            force_dt_vec != -1, force_dt_vec, dt_base
        )  # if force_dt != -1, use that

        # dt = 1 / (2^(dt_base))
        dt = 1.0 / (2.0**dt_base)

        dt_base_bootstrap = dt_base + 1.0
        dt_bootstrap = dt / 2.0

        # -----------------------------------------------------------------
        #  2) =========== Sample t. ============
        # -----------------------------------------------------------------
        dt_sections = 2.0**dt_base

        # We want to sample t ~ Uniform{0, dt_sections[i]} (integer), then divide by dt_sections[i].
        # This is somewhat trickier to do in a single vectorized call in PyTorch
        # (because each batch element can have a different max).
        # We'll do it in a loop for clarity:

        t_list = []
        for i in range(bootstrap_batchsize):
            maxval = int(dt_sections[i].item())  # dt_sections[i] is float, convert to int
            # If maxval == 0 for some reason, clamp to 1 to avoid errors
            if maxval < 1:
                maxval = 1
            # Sample an integer in [0, maxval)
            t_i = torch.randint(low=0, high=maxval, size=(1,), device=device, dtype=torch.int64)
            t_list.append(t_i)

        t = torch.cat(t_list, dim=0).to(torch.float32)  # shape [bootstrap_batchsize]
        t = t / dt_sections  # elementwise scale to [0, 1]

        # force_t logic
        force_t_vec = torch.ones(bootstrap_batchsize, device=device, dtype=torch.float32) * force_t
        t = torch.where(force_t_vec != -1, force_t_vec, t)
        t_full = t.view(-1, 1)

        # -----------------------------------------------------------------
        #  3) =========== Generate Bootstrap Targets ============
        # -----------------------------------------------------------------

        x_1 = samples_data[: self.hparams.bootstrap_batchsize]
        x_0 = samples_prior[: self.hparams.bootstrap_batchsize]
        x_t = (1.0 - (1.0 - 1e-5) * t_full) * x_0 + t_full * x_1

        with torch.no_grad():
            v_b1 = self.forward(torch.cat([x_t, t[:, None], dt_base_bootstrap[:, None]], dim=-1))

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap.view(-1, 1) * v_b1
        x_t2 = torch.clamp(x_t2, -4.0, 4.0)

        with torch.no_grad():
            v_b2 = self.forward(torch.cat([x_t2, t2[:, None], dt_base_bootstrap[:, None]], dim=-1))

        v_target = 0.5 * (v_b1 + v_b2)
        v_target = torch.clamp(v_target, -4.0, 4.0)

        bst_v = v_target
        bst_dt = dt_base
        bst_t = t
        bst_xt = x_t

        # -----------------------------------------------------------------
        #  4) =========== Generate Flow-Matching Targets ============
        # -----------------------------------------------------------------

        # Sample t uniformly in [0, denoise_timesteps), then / denoise_timesteps
        t_rand = torch.randint(0, self.hparams.M, (samples_data.shape[0],), device=device)
        t_float = t_rand.to(torch.float32) / self.hparams.M

        force_t_vec = (
            torch.ones(samples_data.shape[0], device=device, dtype=torch.float32) * force_t
        )
        t_float = torch.where(force_t_vec != -1, force_t_vec, t_float)
        t_full = t_float.view(-1, 1)

        # x_0 ~ N(0, 1)
        x_0 = samples_prior
        x_1 = samples_data

        # x_t = (1 - alpha * t) * x_0 + t * x_1  (with alpha=1-1e-5 in your code)
        x_t_flow = (1.0 - (1.0 - 1e-5) * t_full) * x_0 + t_full * x_1
        v_t_flow = x_1 - (1.0 - 1e-5) * x_0

        dt_flow = int(torch.log2(self.hparams.M))
        dt_base_flow = (
            torch.ones(samples_data.shape[0], device=device, dtype=torch.int32) * dt_flow
        )

        # -----------------------------------------------------------------
        #  5) =========== Merge Flow + Bootstrap =============
        # -----------------------------------------------------------------
        bst_size = batch_size // self.hparams.bootstrap_every
        bst_size_data = batch_size - bst_size

        # Combine the bootstrap slices with the flow slices
        x_t_final = torch.cat([bst_xt, x_t_flow[-bst_size_data:]], dim=0)
        t_final = torch.cat([bst_t, t_float[-bst_size_data:]], dim=0)
        dt_base_final = torch.cat([bst_dt, dt_base_flow[-bst_size_data:]], dim=0)
        v_t_final = torch.cat([bst_v, v_t_flow[-bst_size_data:]], dim=0)

        # TODO do we want to log these?
        # info["bootstrap_ratio"] = torch.mean((dt_base_final != dt_flow).float())
        # info["v_magnitude_bootstrap"] = torch.sqrt(torch.mean(bst_v**2))
        # info["v_magnitude_b1"] = torch.sqrt(torch.mean(v_b1**2))
        # info["v_magnitude_b2"] = torch.sqrt(torch.mean(v_b2**2))

        return x_t_final, v_t_final, t_final, dt_base_final

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        xt, vt_ref, t, dt = self.get_targets(batch)
        vt_pred = self.forward(t, xt, dt)
        loss = self.criterion(vt_pred, vt_ref)

        return loss

    def generate_samples(
        self, batch_size: int, n_timesteps: int = 100, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        node = NeuralODE(
            cnf_wrapper(self.net),
            atol=1e-3,
            rtol=1e-3,
            solver="dopri5",
            sensitivity="adjoint",
        )

        prior_samples = self.prior.sample((batch_size,)).to(device)
        prior_log_p = self.prior.log_prob(prior_samples)

        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([prior_samples, prior_log_p[:, None]], dim=-1),
                t_span=torch.linspace(0, 1, n_timesteps),
            )

        log_p = traj[-1][..., -1]
        samples = traj[-1][..., :-1].reshape(batch_size, 4, -1)  # TODO hardcode

        return samples, log_p, prior_samples


if __name__ == "__main__":
    _ = ShortcutLitModule(None, None, None, None)
