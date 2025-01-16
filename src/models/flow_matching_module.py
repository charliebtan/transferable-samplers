import copy
from typing import Optional, Tuple

import torch
from torchdyn.core import NeuralODE
from tqdm import tqdm

from src.models.boltzmann_generator_module import BoltzmannGeneratorLitModule
from src.models.components.wrappers import TorchdynWrapper, torch_wrapper


class FlowMatchLitModule(BoltzmannGeneratorLitModule):
    """

    TODO - Add a description.

    """

    def __init__(self, sigma: float = 0.0, *args, **kwargs) -> None:
        """Initialize a `ProposalFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(*args, **kwargs)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net(t, x)

    def get_xt(self, x0, x1, t):
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

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch:

        :return: - A tensor of losses.
        """

        t = torch.rand(batch.shape[0], 1, device=batch.device)
        batch_prior = self.prior.sample(batch.shape[0]).to(batch.device)

        xt = self.get_xt(batch_prior, batch, t)
        vt_flow = self.get_flow_targets(batch_prior, batch)

        vt_pred = self.forward(t, xt)
        loss = self.criterion(vt_pred, vt_flow)

        return loss

    def flow(self, x: torch.Tensor, reverse=False) -> torch.Tensor:
        dlog_p_init = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        node = NeuralODE(
            TorchdynWrapper(copy.deepcopy(self.net)),
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
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        prior_samples = self.prior.sample(batch_size).to(self.device)
        prior_log_p = -self.prior.energy(prior_samples)

        with torch.no_grad():
            samples, dlog_p = self.flow(prior_samples, reverse=False)

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p, prior_samples

    def generate_samples_no_ll(self, batch_size) -> torch.Tensor:
        x_0 = self.prior.sample(batch_size).to(self.device)
        t_span = torch.linspace(0, 1, 2)
        node = NeuralODE(
            torch_wrapper(self.net),
            atol=1e-4,
            rtol=1e-4,
            solver="dopri5",
            sensitivity="adjoint",
        )
        x = node.trajectory(x_0, t_span=t_span)[-1]
        return x, x_0

    def batched_generate_samples_no_ll(
        self, total_size: int, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = []
        prior_samples = []
        for _ in tqdm(range(total_size // batch_size)):
            s, ps = self.generate_samples_no_ll(batch_size)
            samples.append(s)
            prior_samples.append(ps)
        if total_size % batch_size > 0:
            s, ps = self.generate_samples_no_ll(total_size % batch_size)
            samples.append(s)
            prior_samples.append(ps)
        samples = torch.cat(samples, dim=0)
        prior_samples = torch.cat(prior_samples, dim=0)
        return samples, prior_samples


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
