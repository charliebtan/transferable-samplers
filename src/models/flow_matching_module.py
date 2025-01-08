from typing import Any, Dict, Tuple

import torch
from torchdyn.core import NeuralODE

from src.models.components.wrappers import torchdyn_wrapper
from src.models.proposal_flow_module import ProposalFlowLitModule


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
        return self.net(x, t)

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
            torchdyn_wrapper(self.net),
            atol=1e-4,
            rtol=1e-4,
            solver="dopri5",
            sensitivity="adjoint",
        )

        prior_samples = self.prior.sample(batch_size).to(device)
        prior_log_p = -self.prior.energy(prior_samples)

        dlog_p_init = torch.zeros_like(prior_log_p)

        with torch.no_grad():
            traj = node.trajectory(
                torch.cat([prior_samples, dlog_p_init], dim=-1),
                t_span=torch.linspace(0, 1, 2),
            )

        dlog_p = traj[-1][..., -1]
        samples = traj[-1][..., :-1]

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p, prior_samples


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
