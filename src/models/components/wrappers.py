import torch

class torchdyn_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format with 
    additional dimension representing the change in likelihood over time."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def div_fn(self, t, x):
        """Hutchingson's trace estimator for the divergence of the vector field.
        Using Rademacher random variables for epsilons.
        """

        x = x.view(1, -1) # batch_dim needed for the architecture

        eps =  torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
        def vecfield(y):
            return self.model(t, y)
        _, vjpfunc = torch.func.vjp(vecfield, x)
        return (vjpfunc(eps)[0] * eps).sum()

    def forward(self, t, x, *args, **kwargs):

        x = x[..., :-1]

        dx = self.model(t, x)
        ddiv = - torch.vmap(self.div_fn, in_dims=(None, 0), randomness="different")(torch.tensor([t]), x)

        return torch.cat([dx, ddiv[:, None]], dim=-1)
