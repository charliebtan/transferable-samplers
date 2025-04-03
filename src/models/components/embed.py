import torch
import torch.nn as nn


class ConditionalEmbedder(nn.Module):
    def __init__(
        self, channels: int = 128, num_atom_emb: int = 54, num_residue_emb: int = 20, num_residue_pos: int = 23
    ):
        super().__init__()

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=channels)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=channels)
        self.residue_pos_embed = nn.Embedding(num_embeddings=num_residue_pos, embedding_dim=channels)

        self.mlp = nn.Sequential(nn.Linear(3 * channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, atom_type, residue_type, residue_pos):
        atom_emb = self.atom_embed(atom_type)
        residue_emb = self.residue_embed(residue_type)
        pos_embed = self.residue_pos_embed(residue_pos)

        x = torch.concat([atom_emb, residue_emb, pos_embed], dim=-1)
        return self.mlp(x)
