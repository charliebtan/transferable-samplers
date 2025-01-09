import torch
from zuko.distributions import DiagNormal
from zuko.flows.autoregressive import MaskedAutoregressiveTransform
from zuko.lazy import Flow, UnconditionalDistribution
from zuko.transforms import (CircularShiftTransform, ComposedTransform,
                             MonotonicRQSTransform)


class OurMAF(Flow):
    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        base=None,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]
        if base is None:
            base = UnconditionalDistribution(
                DiagNormal,
                torch.zeros(features),
                torch.ones(features),
                buffer=True,
            )
        super().__init__(transforms, base)


class OurNSF(OurMAF):
    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        base=None,
        **kwargs,
    ):
        if base is None:
            base = UnconditionalDistribution(
                MyMeanFreeNormalDistribution,
                8,
                4,
                buffer=True,
            )

        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            base=base,
            **kwargs,
        )


# # now set up a prior
# prior =  MeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()


class MyMeanFreeNormalDistribution(torch.distributions.MultivariateNormal):
    """Mean-free normal distribution."""

    def __init__(self, dim, n_particles, std=1.0, two_event_dims=True):
        loc = torch.zeros(dim)
        scale_tril = torch.eye(dim)
        super().__init__(loc=loc, scale_tril=scale_tril)
        self._two_event_dims = two_event_dims
        self._dim = dim
        self._n_particles = n_particles
        self._spacial_dims = dim // n_particles
        self._std = std

    def log_prob(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = self._remove_mean(x).view(-1, self._dim)
        # TODO: make consistent
        return -0.5 * x.pow(2).sum(dim=-1, keepdim=True) / self._std**2

    def sample(self, n_samples, temperature=1.0):
        x = torch.ones(
            (n_samples, self._n_particles, self._spacial_dims),
            dtype=self._std.dtype,
            device=self._std.device,
        ).normal_(mean=0, std=self._std)
        x = self._remove_mean(x)
        if not self._two_event_dims:
            x = x.view(-1, self._dim)
        return x

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x


if __name__ == "__main__":
    dim = 8
    n_particles = 4

    prior = MyMeanFreeNormalDistribution(dim, n_particles, two_event_dims=False).cuda()
    maf = OurMAF(10)
