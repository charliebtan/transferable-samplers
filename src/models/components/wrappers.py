import torch


def exact_div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = torch.func.jacrev(u)
    return lambda x, *args: torch.trace(J(x))


def div_fn_hutch_trace(u):
    def div_fn(x, eps):
        _, vjpfunc = torch.func.vjp(u, x)
        return (vjpfunc(eps)[0] * eps).sum()

    return div_fn


class cnf_wrapper(torch.nn.Module):
    """Wraps model to a torchdyn compatible CNF format.

    Appends an additional dimension representing the change in likelihood over time.
    """

    def __init__(self, model, likelihood_estimator="exact"):
        super().__init__()
        self.model = model
        self.div_fn, self.eps_fn = self.get_div_and_eps(likelihood_estimator)

    def get_div_and_eps(self, likelihood_estimator):
        if likelihood_estimator == "exact":
            return exact_div_fn, None
        if likelihood_estimator == "hutch_gaussian":
            return div_fn_hutch_trace, torch.randn_like
        if likelihood_estimator == "hutch_rademacher":

            def eps_fn(x):
                return torch.randint_like(x, low=0, high=2).float() * 2 - 1.0

            return div_fn_hutch_trace, eps_fn
        raise NotImplementedError(
            f"likelihood estimator {likelihood_estimator} is not implemented"
        )

    def forward(self, t, x, *args, **kwargs):
        t = t.squeeze()
        x = x[..., :-1]

        def vecfield(y):
            return self.model(torch.cat([y, t[None]]))

        if self.eps_fn is None:
            div = torch.vmap(self.div_fn(vecfield))(x)
        else:
            div = torch.vmap(self.div_fn(vecfield))(x, self.eps_fn(x))
        dx = self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))
        return torch.cat([dx, -div[:, None]], dim=-1)


class torchdyn_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format with additional dimension representing the change
    in likelihood over time."""

    def __init__(self, model, d: int = None):
        super().__init__()
        self.model = model
        self.d = d

    def div_fn(self, t, x):
        """Hutchingson's trace estimator for the divergence of the vector field.

        Using Rademacher random variables for epsilons.
        """

        x = x.view(1, -1)  # batch_dim needed for the EGNN architecture

        eps = torch.randint_like(x, low=0, high=2).float() * 2 - 1.0

        def vecfield(y):
            d_vec = torch.ones(y.shape[0]) * self.d
            return self.model(y, t, d=d_vec)

        _, vjpfunc = torch.func.vjp(vecfield, x)
        return (vjpfunc(eps)[0] * eps).sum()

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]  # remove the divergence estimate

        d_vec = torch.ones(x.shape[0]) * self.d
        dx = self.model(x, t, d=d_vec)
        ddiv = -torch.vmap(self.div_fn, in_dims=(None, 0), randomness="different")(
            torch.tensor([t]), x
        )

        return torch.cat([dx, ddiv[:, None]], dim=-1)
