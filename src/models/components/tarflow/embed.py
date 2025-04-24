import math

import torch
import torch.nn as nn


class SinusodialEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        self.embed_size = embed_size
        self.max_len = max_len

    def forward(self, indices):
        """Creates sine / cosine positional embeddings from a prespecified indices.

        Args:
            indices: offsets of size [..., N_edges] of type integer
            max_len: maximum length.
            embed_size: dimension of the embeddings to create

        Returns:
            positional embedding of shape [N, embed_size]
        """
        K = torch.arange(self.embed_size // 2, device=indices.device)
        pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (self.max_len ** (2 * K[None] / self.embed_size))).to(
            indices.device
        )
        pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (self.max_len ** (2 * K[None] / self.embed_size))).to(
            indices.device
        )
        pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
        return pos_embedding


class ConditionalEmbedder(nn.Module):
    def __init__(self, channels: int = 128, num_atom_emb: int = 54, num_residue_emb: int = 20):
        """
        Input the value of the atom type, residue type, and residue position WITHOUT counting the padding token
        """

        super().__init__()

        num_atom_emb += 1  # for padding
        num_residue_emb += 1

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=channels)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=channels)
        self.residue_pos_embed = SinusodialEmbedding(embed_size=channels)
        self.seq_len_embed = SinusodialEmbedding(embed_size=channels)

        self.mlp = nn.Sequential(nn.Linear(4 * channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, atom_type, aa_type, aa_pos, seq_len, mask=None):
        if mask is None:
            mask = torch.ones_like(atom_type, dtype=torch.bool)

        num_atoms = atom_type.shape[-1]
        atom_emb = self.atom_embed(atom_type)
        residue_emb = self.residue_embed(aa_type)
        pos_embed = self.residue_pos_embed(aa_pos)
        seq_len_embed = self.seq_len_embed(seq_len).expand(-1, num_atoms, -1)

        x = torch.concat([atom_emb, residue_emb, pos_embed, seq_len_embed], dim=-1)
        return self.mlp(x) * mask[..., None]  # [b, n, channels]
