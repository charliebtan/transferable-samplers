import torch


class TorchdynWrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format with additional dimension representing the change
    in likelihood over time."""

    def __init__(self, model, d_base: int = None, div_estimator="exact"):
        super().__init__()
        self.model = model
        self.d_base = d_base

        if div_estimator == "exact":
            self.div_fn = self.div_fn_exact
        else:
            self.div_fn = self.div_fn_hutch
            if div_estimator == "hutch_guassian":
                self.eps_fn = lambda x: torch.randn_like(x)
            elif div_estimator == "hutch_rademacher":
                self.eps_fn = lambda x: torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(
                    f"likelihood estimator {div_estimator} is not implemented"
                )

    def div_fn_hutch(self, t, x):
        """Hutchingson's trace estimator for the divergence of the vector field.

        Using Rademacher random variables for epsilons.
        """

        eps = self.eps_fn(x)

        def vecfield(y):
            y = y.view(1, -1)  # batch dims required by EGNN architecture
            if self.d_base is not None:
                d_base_vec = torch.ones(y.shape[0]) * self.d
                return self.model(t, y, d_base=d_base_vec)
            else:
                return self.model(t, y, d_base=self.d_base)

        _, vjpfunc = torch.func.vjp(vecfield, x)
        return (vjpfunc(eps)[0] * eps).sum()

    def div_fn_exact(self, t, x):
        def vecfield(y):
            y = y.view(1, -1)  # batch dims required by EGNN architecture
            if self.d_base is not None:
                d_base_vec = torch.ones(y.shape[0], device=y.device) * self.d_base
                return self.model(t, y, d_base=d_base_vec).flatten()
            else:
                return self.model(t, y, d_base=self.d_base).flatten()

        J = torch.func.jacrev(vecfield)

        return torch.trace(J(x))

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]  # remove the divergence estimate

        if self.d_base is not None:
            d_base_vec = torch.ones(x.shape[0], device=x.device) * self.d_base
            dx = self.model(t, x, d_base=d_base_vec)
        else:
            dx = self.model(t, x, d_base=self.d_base)
        dlog_p = -torch.vmap(self.div_fn, in_dims=(None, 0), randomness="different")(
            torch.tensor([t], device=x.device), x
        )

        return torch.cat([dx, dlog_p[:, None]], dim=-1).detach()
