import os

import torch
from src.models.flow_matching_module import FlowMatchLitModule


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
            atol=self.hparams.atol,
            rtol=self.hparams.rtol,
            div_estimator=self.hparams.div_estimator,
            logp_tol_scale=self.hparams.logp_tol_scale,
        )
        self.samples = None
        self.prior_samples = None

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # sample the prior and get xt
        output_dir = self.hparams.base_flow_ckpt_path
        # strip the file from output_dir
        output_dir = output_dir[: output_dir.rfind("/")]
        if self.samples is None:
            if os.path.exists(f"{output_dir}/samples.pt"):
                self.samples, _, self.prior_samples = torch.load(f"{output_dir}/samples.pt")
            else:
                self.samples, _, self.prior_samples = (
                    self.base_flow.batched_generate_samples_no_ll(
                        self.hparams.num_reflow_samples, batch_size=self.hparams.reflow_batch_size
                    )
                )
                self.evaluate(
                    prefix="base_flow",
                    generator=self.base_flow.batched_generate_samples_no_ll,
                    output_dir=output_dir,
                )
        # Sample random indices from length of samples
        idx = torch.randint(0, self.samples.shape[0], (batch.shape[0],), device=self.device)
        batch_prior = self.prior_samples[idx]
        batch_target = self.samples[idx]
        vt_target = batch_target - batch_prior
        t = torch.zeros(batch.shape[0], 1, device=batch.device)
        vt_pred = self.forward(t, batch_prior)
        loss = self.criterion(vt_pred, vt_target)
        return loss

    def flow(self, x: torch.Tensor, reverse=False, dummy_ll=True) -> torch.Tensor:
        dlog_p_init = torch.zeros((x.shape[0], 1), device=x.device)
        t = torch.zeros(x.shape[0], 1, device=x.device)
        vt_pred = self.forward(t, x)
        x = x + vt_pred
        # Inaccurate dlog_p
        return x, dlog_p_init
