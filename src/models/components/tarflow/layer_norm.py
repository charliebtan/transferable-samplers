import torch
from torch.nn import functional as F


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class AdaptiveLayerNorm(torch.nn.Module):
    """Adaptive layer norm layer, where scales and biases are learned from some
    conditioning variables."""

    def __init__(self, *, channels: int, channels_cond: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_cond = torch.nn.LayerNorm(channels_cond)

        self.to_gamma = torch.nn.Sequential(torch.nn.Linear(channels_cond, channels), torch.nn.Sigmoid())

        self.to_beta = torch.nn.Linear(channels_cond, channels, bias=False)

    def forward(self, x, cond, mask):
        """
        Args:
            x: input representation, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Representation after adaptive layer norm, shape as input representation [*, dim].
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        out = normed * gamma + beta
        return out * mask[..., None]


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class AdaptiveLayerNormOutputScale(torch.nn.Module):
    """Adaptive scaling of a representation given conditioning variables."""

    def __init__(self, *, channels, channels_cond, adaln_zero_bias_init_value=-2.0):
        super().__init__()

        adaln_zero_gamma_linear = torch.nn.Linear(channels_cond, channels)
        torch.nn.init.zeros_(adaln_zero_gamma_linear.weight)
        torch.nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = torch.nn.Sequential(adaln_zero_gamma_linear, torch.nn.Sigmoid())

    def forward(self, x, cond, mask):
        """
        Args:
            x: input sequence, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Scaled input, shape [*, dim].
        """
        gamma = self.to_adaln_zero_gamma(cond)  # [*, dim]
        return x * gamma * mask[..., None]


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class Transition(torch.nn.Module):
    """Transition layer."""

    def __init__(self, channels, expansion_factor=4, layer_norm=False):
        super().__init__()

        channels_inner = int(channels * expansion_factor)

        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.ln = torch.nn.LayerNorm(channels)

        self.swish_linear = torch.nn.Sequential(
            torch.nn.Linear(channels, channels_inner * 2, bias=False),
            SwiGLU(),
        )
        self.linear_out = torch.nn.Linear(channels_inner, channels, bias=False)

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            mask: binary, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        if self.use_layer_norm:
            x = self.ln(x)
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]
