import torch
import torch.nn as nn


class ConditionalEmbedder(nn.Module):
    def __init__(
        self, channels: int = 128, num_atom_emb: int = 54, num_residue_emb: int = 20, num_residue_pos: int = 2
    ):
        """
        Input the value of the atom type, residue type, and residue position WITHOUT counting the padding token
        """

        super().__init__()

        num_atom_emb += 1  # for padding
        num_residue_emb += 1
        num_residue_pos += 1

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
