import copy
import logging
from typing import Optional

import torch
from torchdyn.core import NeuralODE

from src.models.components.wrappers import TorchdynWrapper, torch_wrapper
from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule

logger = logging.getLogger(__name__)


class FlowMatchLitModule(TransferableBoltzmannGeneratorLitModule):
    """

    TODO - Add a description.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a `ProposalFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(*args, **kwargs)
        self.nfe = 0
        self.num_integrations = 0
        self.eps = 1e-1
        if "strict_loading" in kwargs:
            self.strict_loading = kwargs["strict_loading"]

    def forward(self, t: torch.Tensor, x: torch.Tensor, encoding, mask) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x:
        :param t:
        :return: dx
        """
        return self.net(t, x, encoding=encoding, node_mask=mask)

    def get_xt(self, x0, x1, t):
        return (1.0 - t) * x0 + t * x1

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

        x1 = batch["x"]

        encoding = batch["encoding"]
        mask = batch.get("mask", None)

        num_samples = x1.shape[0]
        assert not x1.shape[1] % self.datamodule.hparams.num_dimensions, (
            f"x1 should be a multiple of {self.datamodule.hparams.num_dimensions}"
        )
        num_tokens = x1.shape[1] // self.datamodule.hparams.num_dimensions

        t = torch.rand(num_samples, 1, device=x1.device)
        prior_samples = self.prior.sample(num_samples, num_tokens, mask, device=x1.device)

        xt = self.get_xt(prior_samples, x1, t)
        vt_flow = self.get_flow_targets(prior_samples, x1)

        vt_pred = self.forward(t, xt, encoding=encoding, mask=mask)
        loss = self.criterion(vt_pred, vt_flow)

        return loss

    def test_integrators(self) -> torch.Tensor:
        x = self.prior.sample(self.hparams.sampling_config.batch_size).to(self.device)
        integrators = [
            "exact",
            "exact_no_functional",
            "hutch_rademacher",
            "hutch_gaussian",
        ]
        logger.info("Testing integrators")
        self.hparams.div_estimator = "exact"
        self.nfe = 0
        self.hparams.n_eps = 1
        base_x, base_dlog_p = self.flow(x, reverse=False)
        for integrator in integrators:
            logger.info(f"Testing integrator {integrator}")
            self.hparams.div_estimator = integrator
            for n in [1, 2, 4, 8, 16, 32]:
                if integrator.startswith("exact") and n > 1:
                    continue
                self.hparams.n_eps = n
                x, dlog_p = self.flow(x, reverse=False)
                self.log_dict(
                    {
                        f"test_integrators/{integrator}_{n}/x_err": torch.norm(base_x - x),
                        f"test_integrators/{integrator}_{n}/dlog_p_err": torch.norm(base_dlog_p - dlog_p),
                        f"test_integrators/{integrator}_{n}/nfe": self.nfe / (max(self.num_integrations, 1e-4)),
                    }
                )
                logger.info(
                    f"estimator: {integrator} n: {n}, x_err: {torch.norm(base_x - x)}, "
                    f"dlog_p_err: {torch.norm(base_dlog_p - dlog_p)}, "
                    f"nfe: {self.nfe / max(self.num_integrations, 1e-4)}"
                )
                self.nfe = 0

    @torch.no_grad()
    def flow(self, x: torch.Tensor, encoding, reverse=False, dummy_ll=False) -> torch.Tensor:
        dlog_p = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        eval_net = copy.deepcopy(self.net)  # copied needed - ask alex
        eval_net.eval_encoding = encoding  # yes its a hack - fight me (charlie)

        print(eval_net.eval_encoding)
        breakpoint()

        if self.hparams.div_estimator == "ito":
            x_ito, dlog_p_ito = self.sde_integrate(x, reverse=reverse)
            # prior_log_p = -self.prior.energy(x)
            # logp_ito = prior_log_p.squeeze() + dlog_p_ito
            # x = x_ito
            # reverse = True
            # t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)
            return x_ito, dlog_p_ito

        if dummy_ll:
            wrapped_net = torch_wrapper(eval_net)
        else:
            wrapped_net = TorchdynWrapper(
                eval_net,
                div_estimator=self.hparams.div_estimator,
                logp_tol_scale=self.hparams.logp_tol_scale,
                n_eps=self.hparams.n_eps,
            )

        node = NeuralODE(
            wrapped_net,
            atol=self.hparams.atol,
            rtol=self.hparams.rtol,
            solver="dopri5",
            sensitivity="adjoint",
        )
        if not dummy_ll:
            x = torch.cat([x, dlog_p], dim=-1)
        x = node.trajectory(x, t_span=t_span)[-1]
        self.nfe += wrapped_net.nfe
        self.num_integrations += 1
        wrapped_net.nfe = 0
        if not dummy_ll:
            dlog_p = x[..., -1] * self.hparams.logp_tol_scale
            x = x[..., :-1]
        # logp = (-self.prior.energy(x).view(-1) - dlog_p.view(-1))

        return x, dlog_p

    def euler_maruyama_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        step: int,
        batch_size: int,
    ):
        vt = self.net(t, x, d_base=None)

        # if t.dim() == 0:
        # # repeat the same time for all points if we have a scalar time
        # t = t * torch.ones_like(x).to(x.device)

        sigma_t_squared = 2 * (1 - t) / torch.clip(t, min=self.eps)
        sigma_t_squared = 2 * (1 - t) / torch.clip(t, min=0.1)
        sigma_t = sigma_t_squared**0.5

        # st is correct we checked
        st = vt + sigma_t_squared * (t * vt - x) / torch.clip(1 - t, min=self.eps) / 2
        eps_t = torch.randn_like(x)
        noise_t = sigma_t * eps_t * (dt**0.5)
        dxt = st * dt + noise_t

        score_t = (t * vt - x) / torch.clip(1 - t, min=self.eps)
        a = -x.shape[-1] / torch.clip(t, min=self.eps) * dt
        b = (sigma_t_squared / 2 * score_t * score_t).sum(-1) * dt
        c = (score_t * noise_t).sum(-1)

        dlogp = a + b + c
        # Update the state
        x_next = x + dxt
        return x_next, dlogp

    def sde_integrate(self, x0: torch.Tensor, reverse=False) -> torch.Tensor:
        batch_size = x0.shape[0]
        time_range = 1.0
        start_time = 1.0 if reverse else 0.0
        end_time = 1.0 - start_time
        if self.num_integrations == 0:
            num_integration_steps = 1000
        else:
            num_integration_steps = self.num_integrations
        times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]
        x = x0

        x0.requires_grad = True
        samples = []
        dlogp_sum = 0

        for step, t in enumerate(times):
            x, dlogp = self.euler_maruyama_step(t, x, time_range / num_integration_steps, step, batch_size)
            dlogp_sum += dlogp
            samples.append(x)

        samples = torch.stack(samples)
        return x, dlogp_sum

    def proposal_energy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, dlogp = self.flow(x, reverse=True)
        return -(-self.prior.energy(x).view(-1) - dlogp.view(-1))

    def evaluate(
        self, sequence, true_samples, encoding, energy_fn, prefix: str = "val", proposal_generator=None, output_dir=None
    ):
        logger.info(f"has test_integrators {hasattr(self.hparams, 'test_integrators')}")
        if True and hasattr(self.hparams, "test_integrators"):
            self.test_integrators()
            return {}
        results = super().evaluate(sequence, true_samples, encoding, energy_fn, prefix, proposal_generator, output_dir)

        self.log(f"{prefix}/nfe", self.nfe / (max(self.num_integrations, 1e-4)))
        self.nfe = 0
        self.num_integrations = 0
        return results

    @torch.no_grad()
    def generate_samples(
        self,
        batch_size: int,
        encoding: Optional[dict[str, torch.Tensor]] = None,
        dummy_ll: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        num_particles = encoding["atom_type"].size(0)
        data_dim = num_particles * self.datamodule.hparams.num_dimensions

        local_batch_size = batch_size // self.trainer.world_size
        prior_samples = self.prior.sample(local_batch_size, num_particles).to(self.device)

        # need to rescale to the "sum" of the log p (the prior returns the position-wise mean)
        prior_log_p = -self.prior.energy(prior_samples) * data_dim

        if encoding is not None:
            encoding = {
                key: tensor.unsqueeze(0).repeat(local_batch_size, 1).to(self.device) for key, tensor in encoding.items()
            }

        with torch.no_grad():
            samples, dlog_p = self.flow(prior_samples, encoding=encoding, reverse=False, dummy_ll=dummy_ll)
            samples = self.all_gather(samples).reshape(batch_size, -1)
            dlog_p = self.all_gather(dlog_p).reshape(-1, *dlog_p.shape[1:])
            prior_log_p = self.all_gather(prior_log_p).reshape(-1, *prior_log_p.shape[1:])
            prior_samples = self.all_gather(prior_samples).reshape(-1, *prior_samples.shape[1:])

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p, prior_samples

    @torch.no_grad()
    def batched_generate_samples_no_ll(
        self, total_size: int, batch_size: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().batched_generate_samples(total_size, batch_size, dummy_ll=True)


if __name__ == "__main__":
    _ = FlowMatchLitModule(None, None, None, None)
