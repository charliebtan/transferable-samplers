import torch
import torch.nn as nn
from layer_norm import AdaptiveLayerNorm, AdaptiveLayerNormOutputScale, Transition


class ConditionalEmbedder(nn.Module):
    def __init__(
        self, channels: int = 128, num_atom_emb: int = 55, num_residue_emb: int = 21, num_residue_pos: int = 24
    ):
        super().__init__()

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=channels)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=channels)
        self.residue_pos_embed = nn.Embedding(num_embeddings=num_residue_pos, embedding_dim=channels)

        self.mlp = nn.Sequential(nn.Linear(3 * channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, atom_type, aa_type, aa_pos, mask=None):
        if mask is None:
            mask = torch.ones_like(atom_type, dtype=torch.bool)

        atom_emb = self.atom_embed(atom_type)
        residue_emb = self.residue_embed(aa_type)
        pos_embed = self.residue_pos_embed(aa_pos)

        x = torch.concat([atom_emb, residue_emb, pos_embed], dim=-1)
        return self.mlp(x) * mask[..., None]  # [b, n, channels]


class MultiHeadAttention(torch.nn.Module):
    """Typical multi-head self-attention attention using pytorch's module."""

    def __init__(self, channels: int = 128, num_heads: int = 8, dropout=0.0, use_qkln: bool = False):
        super().__init__()

        self.to_q = torch.nn.Linear(channels, channels)
        self.to_kv = torch.nn.Linear(channels, 2 * channels, bias=False)
        self.q_layer_norm = nn.LayerNorm(channels) if use_qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(channels) if use_qkln else nn.Identity()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence, shape [b, n, channels]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence, shape [b, n, channels]
        """
        query = self.to_q(x)  # [b, n, channels]
        key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, channels]
        query = self.q_layer_norm(query)
        key = self.k_layer_norm(key)
        return (
            self.mha(
                query=query,
                key=key,
                value=value,
                key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
                need_weights=False,
                is_causal=False,
            )[0]
            * mask[..., None]
        )  # [b, n, channels]


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
        num_heads: int = 8,
        use_qkln: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.adaln = AdaptiveLayerNorm(channels=channels, channels_cond=channels)
        self.mha = MultiHeadAttention(channels=channels, num_heads=num_heads, dropout=dropout, use_qkln=use_qkln)
        self.scale_output = AdaptiveLayerNormOutputScale(channels=channels, channels_cond=channels)

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, channels]
            cond: Conditioning variables, shape [b, n, channels]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, channels].
        """

        x = self.adaln(x, cond, mask)
        x = self.mha(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]  # [b, n, channels]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, channels, channels_cond, expansion_factor=4):
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


class MultiheadAttnAndTransition(torch.nn.Module):
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
        num_heads: int = 8,
        residual_mha: bool = True,
        residual_transition: bool = True,
        use_qkln: bool = True,
        dropout=0.0,
        expansion_factor=4,
    ):
        super().__init__()

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mha = MultiHeadAttentionADALN(
            channels=channels,
            num_heads=num_heads,
            use_qkln=use_qkln,
            dropout=dropout,
        )

        self.transition = TransitionADALN(channels=channels, channels_cond=channels, expansion_factor=expansion_factor)

    def _apply_mha(self, x, cond, mask):
        x_attn = self.mha(x, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        x = self._apply_mha(x, cond, mask)
        x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    num_atoms = 10
    channels = 128
    num_heads = 8

    x = torch.randn((batch_size, num_atoms, channels))
    atom_type = torch.randint(0, 55, (batch_size, num_atoms))
    aa_type = torch.randint(0, 21, (batch_size, num_atoms))
    aa_pos = torch.randint(0, 24, (batch_size, num_atoms))
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)

    embedder = ConditionalEmbedder(channels=channels)
    cond = embedder(atom_type, aa_type, aa_pos, mask)

    adapt_layernorm = MultiheadAttnAndTransition(channels=channels, num_heads=num_heads)
    output = adapt_layernorm(x, cond, mask)
    print(output.shape)  # Should be [batch_size, num_atoms, channels]

    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"

    num_atoms2 = 5
    x2 = torch.randn((batch_size, num_atoms2, channels))
    atom_type2 = torch.randint(0, 55, (batch_size, num_atoms2))
    aa_type2 = torch.randint(0, 21, (batch_size, num_atoms2))
    aa_pos2 = torch.randint(0, 24, (batch_size, num_atoms2))
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
    output = adapt_layernorm(x, cond, mask)

    print(output.shape)  # Should be [batch_size*2, num_atoms, channels]
    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"
