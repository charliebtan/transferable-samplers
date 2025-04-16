import torch
import torch.nn as nn
from adaptive_layer_norm import AdaptiveLayerNorm, AdaptiveLayerNormOutputScale, Transition


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}

    def forward_spda(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum("bmhd,bnhd->bmnh", q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum("bmnh,bnhd->bmhd", attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
        **kwargs,
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MultiHeadAttentionADALN(nn.Module):
    """
    Adapted from Proteina:
        https://github.com/NVIDIA-Digital-Bio/proteina/blob/main/proteinfoundation/nn/protein_transformer.py

    Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output.
    """

    def __init__(
        self,
        channels: int = 128,
        head_channels: int = 64,
        use_qkln: bool = False,
        dropout: float = 0.0,
        expansion: int = 4,
        sample: bool = False,
    ):
        super().__init__()

        assert channels % head_channels == 0, "in_channels must be divisible by head_channels"
        self.sample = sample
        self.adaln = AdaptiveLayerNorm(channels=channels, channels_cond=channels)
        self.mha = AttentionBlock(channels=channels, head_channels=head_channels, expansion=expansion)
        self.scale_output = AdaptiveLayerNormOutputScale(channels=channels, channels_cond=channels)

    def forward(self, x, cond, mask, attn_mask, attn_temp: float = 1.0, which_cache: str = "cond"):
        """
        Args:
            x: Input sequence representation, shape [b, n, channels]
            cond: Conditioning variables, shape [b, n, channels]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, channels].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, attn_mask, attn_temp, which_cache=which_cache)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]  # [b, n, channels]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling applied to output."""

    def __init__(self, channels, channels_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(channels=channels, channels_cond=channels_cond)
        self.transition = Transition(channels=channels, expansion_factor=expansion_factor, layer_norm=False)
        self.scale_output = AdaptiveLayerNormOutputScale(channels=channels, channels_cond=channels_cond)

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class AdaptiveAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        channels: int = 128,
        head_channels: int = 64,
        residual_mha: bool = False,
        residual_transition: bool = False,
        use_qkln: bool = True,
        dropout=0.0,
        expansion=4,
        sample: bool = False,
    ):
        super().__init__()

        assert channels % head_channels == 0, "in_channels must be divisible by head_dim"
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition
        self.sample = sample

        self.mha = MultiHeadAttentionADALN(
            channels=channels,
            head_channels=head_channels,
            use_qkln=use_qkln,
            dropout=dropout,
            expansion=expansion,
        )

        self.transition = TransitionADALN(channels=channels, channels_cond=channels, expansion_factor=expansion)

    def _apply_mha(self, x, cond, mask, attn_mask=None, attn_temp: float = 1.0, which_cache: str = "cond"):
        x_attn = self.mha(x, cond, mask, attn_mask, attn_temp, which_cache=which_cache)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, cond, mask=None, attn_mask=None, attn_temp: float = 1.0, which_cache: str = "cond"):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        x = x * mask[..., None]
        x = self._apply_mha(x, cond, mask, attn_mask=attn_mask, attn_temp=attn_temp, which_cache=which_cache)
        # x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    num_atoms = 10
    channels = 128
    num_heads = 8

    from embed import ConditionalEmbedder

    x = torch.randn((batch_size, num_atoms, channels))
    atom_type = torch.randint(0, 54, (batch_size, num_atoms))
    aa_type = torch.randint(0, 20, (batch_size, num_atoms))
    aa_pos = torch.randint(0, 2, (batch_size, num_atoms))
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)

    embedder = ConditionalEmbedder(channels=channels)
    cond = embedder(atom_type, aa_type, aa_pos, mask)

    adapt_layernorm = AdaptiveAttnAndTransition(in_channels=channels, head_channels=64)
    attn_mask = torch.tril(torch.ones((num_atoms, num_atoms), dtype=torch.bool))
    output = adapt_layernorm(x, cond, mask, attn_mask)
    print(output.shape)  # Should be [batch_size, num_atoms, channels]

    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"

    num_atoms2 = 5
    x2 = torch.randn((batch_size, num_atoms2, channels))
    atom_type2 = torch.randint(0, 54, (batch_size, num_atoms2))
    aa_type2 = torch.randint(0, 20, (batch_size, num_atoms2))
    aa_pos2 = torch.randint(0, 2, (batch_size, num_atoms2))
    mask2 = torch.ones((batch_size, num_atoms2), dtype=torch.bool)

    # pad atom_type2, aa_type2, aa_pos2 to match num_atoms
    x2 = torch.concat([x2, torch.zeros((batch_size, num_atoms - num_atoms2, channels))], dim=1)
    atom_type2 = torch.concat([atom_type2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    aa_type2 = torch.concat([aa_type2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    aa_pos2 = torch.concat([aa_pos2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    mask2 = torch.concat([mask2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.bool)], dim=1)

    # concat x and x2, atom_type and atom_type2, etc
    x = torch.concat([x, x2], dim=0)
    atom_type = torch.concat([atom_type, atom_type2], dim=0)
    aa_type = torch.concat([aa_type, aa_type2], dim=0)
    aa_pos = torch.concat([aa_pos, aa_pos2], dim=0)
    mask = torch.concat([mask, mask2], dim=0)
    cond = embedder(atom_type, aa_type, aa_pos)
    attn_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool))
    attn_mask = attn_mask[None, ...].repeat(x.shape[0], 1, 1)
    attn_mask = mask[..., None] * attn_mask
    attn_mask = attn_mask.unsqueeze(1)
    output = adapt_layernorm(x, cond, mask, attn_mask)

    print(output.shape)  # Should be [batch_size*2, num_atoms, channels]
    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"
