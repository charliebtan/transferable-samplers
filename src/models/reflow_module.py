from src.models.flow_matching_module import FlowMatchLitModule
import torch


class ReflowModule(FlowMatchLitModule):
    def __init__(
        self,
        base_flow_ckpt_path: str,
        *args,
        **kwargs,
    ) -> None:
        if base_flow_ckpt_path is None:
            raise ValueError("base_flow_ckpt_path must be provided")
        super().__init__(*args, **kwargs)
        self.base_flow = FlowMatchLitModule.load_from_checkpoint(
            base_flow_ckpt_path,
            datamodule=self.datamodule,
        )
        self.samples = None
        self.prior_samples = None

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # sample the prior and get xt
        if self.samples is None:
            self.samples, self.prior_samples = self.base_flow.batched_generate_samples_no_ll(
                self.hparams.num_reflow_samples, batch_size=self.hparams.reflow_batch_size
            )
            self.evaluate(prefix="base_flow", generator=self.base_flow.batched_generate_samples)
        # Sample random indices from length of samples
        idx = torch.randint(0, self.samples.shape[0], (batch.shape[0],), device=self.device)
        batch_prior = self.prior_samples[idx]
        batch_target = self.samples[idx]
        x_pred, _ = self.net.forward(batch_prior)
        loss = self.criterion(x_pred, batch_target)

        return loss
    pass
