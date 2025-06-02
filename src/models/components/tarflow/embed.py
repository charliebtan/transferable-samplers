import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_size, div_value, max_len=20):
        # max_len = 20 for long-term goal of oligopeptides
        super().__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.div_value = div_value

        assert self.embed_size % 2 == 0, "embed_size must be even."

    def forward(self, indices):
        # TODO fix docstring
        """Creates sine / cosine positional embeddings from a prespecified indices.

        Args:
            indices: offsets of size [..., N_edges] of type integer
            max_len: maximum length.
            embed_size: dimension of the embeddings to create

        Returns:
            positional embedding of shape [N, embed_size]
        """
        if not self.div_value:
            K = torch.arange(self.embed_size // 2, device=indices.device)
            pos_embedding_sin = torch.sin(
                indices[..., None] * math.pi / (self.max_len ** (2 * K[None] / self.embed_size))
            )
            pos_embedding_cos = torch.cos(
                indices[..., None] * math.pi / (self.max_len ** (2 * K[None] / self.embed_size))
            )
            pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
        else:
            position = indices.float().unsqueeze(-1)  # Shape: [..., 1]
            div_term = torch.exp(
                torch.arange(0, self.embed_size, 2, device=indices.device)
                * -(math.log(self.div_value) / self.embed_size)
            )  # Shape: [embed_size // 2]
            sinusoid_inp = position * div_term  # Shape: [..., embed_size // 2]
            pos_embedding = torch.zeros(*sinusoid_inp.shape[:-1], self.embed_size, device=indices.device)
            pos_embedding[..., 0::2] = torch.sin(sinusoid_inp)
            pos_embedding[..., 1::2] = torch.cos(sinusoid_inp)
        return pos_embedding


class ConditionalEmbedder(nn.Module):
    def __init__(
        self, channels: int = 128, num_atom_emb: int = 54, num_residue_emb: int = 20, sinusoid_div_value: float = 0.0
    ):
        """
        Input the value of the atom type, residue type, and residue position WITHOUT counting the padding token
        """

        super().__init__()

        num_atom_emb += 1  # for padding
        num_residue_emb += 1

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=channels)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=channels)
        self.residue_pos_embed = SinusoidalEmbedding(embed_size=channels, div_value=sinusoid_div_value)
        self.seq_len_embed = SinusoidalEmbedding(embed_size=channels, div_value=sinusoid_div_value)

        self.mlp = nn.Sequential(nn.Linear(4 * channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, atom_type, aa_type, aa_pos, seq_len, mask=None):
        if mask is None:
            mask = torch.ones_like(atom_type, dtype=torch.bool)

        atom_emb = self.atom_embed(atom_type)
        residue_emb = self.residue_embed(aa_type)
        pos_embed = self.residue_pos_embed(aa_pos)

        num_tokens = atom_type.shape[1]
        # seq_len is of shape [b, 1], once embeded it will be [b, 1, channels]
        # so we expand it to [b, n, channels] to be concatenated with the other embeddings
        seq_len_embed = self.seq_len_embed(seq_len).expand(-1, num_tokens, -1)

        x = torch.concat([atom_emb, residue_emb, pos_embed, seq_len_embed], dim=-1)
        return self.mlp(x) * mask[..., None]  # [b, n, channels]
