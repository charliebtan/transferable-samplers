#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    # This is when we run the script directly to test model
    from adaptive_blocks import AdaptiveAttnAndTransition
    from attention import Attention, AttentionBlock
    from embed import SinusoidalEmbedding, ResidueConditionalEmbedder, AtomConditionalEmbedder
else:
    from src.models.components.tarflow.adaptive_blocks import AdaptiveAttnAndTransition
    from src.models.components.tarflow.attention import Attention, AttentionBlock
    from src.models.components.tarflow.embed import SinusoidalEmbedding # TODO instantiate

MAX_SEQ_LEN = 512

class PermutationFromDict(torch.nn.Module):
    def __init__(self, permutation_key: str):
        super().__init__()
        self.permutation_key = permutation_key

    def forward(self, data: torch.Tensor, data_permutations_dict: dict[str, torch.Tensor], inverse: bool = False):
        data_permutations_dict = data_permutations_dict["permutations"] # TODO refactor out
        assert self.permutation_key in data_permutations_dict, f"Permutation key {self.permutation_key} not found in data_permutations"
        permutation = data_permutations_dict[self.permutation_key]
        if inverse:
            permutation = torch.argsort(permutation) # get inverse permutation
        permutation = permutation.unsqueeze(-1).expand(-1, -1, data.shape[-1])
        permuted_data = torch.gather(data, dim=1, index=permutation)
        return permuted_data

class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: PermutationFromDict,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        conditional: bool = False,
        use_adapt_ln: bool = False,
        use_attn_pair_bias: bool = False,
        pair_bias_hidden_dim: int = 16,
        use_transition: bool= False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        pos_embed_type: str = "learned",  # learned, sinusoidal
        debug: bool = False,
        lookahead_conditioning: bool = False,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.lookahead_conditioning = lookahead_conditioning

        if conditional:
            if not lookahead_conditioning:
                self.proj_cond = torch.nn.Linear(channels, channels)
            else:
                self.proj_cond = torch.nn.Sequential(
                    torch.nn.Linear(channels * 2, channels),
                    torch.nn.GELU(),
                    torch.nn.Linear(channels, channels),
                )

        if use_attn_pair_bias:
            num_heads = channels // head_dim
            # only a single projection for each block - we don't update cdists within the block
            # so i think this makes sense to just project once - you could learn a different projection for each layer
            # but i don't think this will be worthwhile
            self.pair_proj = torch.nn.Sequential(
                torch.nn.Linear(1, pair_bias_hidden_dim),
                torch.nn.GELU(),
                # needs projecting to num_heads as each head has its own attn_mask
                torch.nn.Linear(pair_bias_hidden_dim, num_heads, bias=False),  # softmax is invariant to bias
            )

            # Scale the weights of the MLP layers - to slow down "switching on of learned mask"
            with torch.no_grad():
                self.pair_proj[0].weight.mul_(1e-3)
                self.pair_proj[-1].weight.mul_(1e-9)

        self.use_attn_pair_bias = use_attn_pair_bias

        if pos_embed_type == "learned":
            if debug:
                # if debug use a larger value for the position embedding to make it easier to see borkage
                self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-1)
            else:
                self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        elif pos_embed_type == "sinusoidal":
            # if not constant you will fail the test checking they are the same - also gives room to increase later
            self.pos_embed = SinusoidalEmbedding(embed_size=channels, max_len=MAX_SEQ_LEN, div_value=0.0)(torch.arange(MAX_SEQ_LEN))
            self.pos_embed_scale = torch.nn.Parameter(torch.ones(1) * 1e-2)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}. Use 'learned' or 'sinusoidal'.")
        self.pos_embed_type = pos_embed_type

        attn_block = AdaptiveAttnAndTransition if use_adapt_ln else AttentionBlock
        self.attn_blocks = torch.nn.ModuleList(
            [
                attn_block(
                    channels=channels,
                    head_channels=head_dim,
                    expansion=expansion,
                    use_qkln=use_qkln,
                    use_attn_pair_bias=use_attn_pair_bias,
                    use_transition=use_transition,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        if debug:
            self.proj_out.weight.data = self.proj_out.weight.data * 1e-1
        else:
            self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer("attn_mask", torch.tril(torch.ones(num_patches, num_patches)))
        self.in_channels = in_channels

    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x_in = self.permutation(x, permutations)  # store permuted input for later

        x = self.proj_in(x)
        x = self.permutation(x, permutations)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid
        x = x + pos_embed

        if cond is not None:
            cond = self.permutation(cond, permutations)
            if self.lookahead_conditioning:
                lookahead_cond = torch.cat([cond[:, 1:], torch.zeros_like(cond[:, :1])], dim=1)  # shift back one token w/ zero pad
                cond = torch.cat([cond, lookahead_cond], dim=-1)  # concatenate the two
            cond_emb = self.proj_cond(cond)
        else:
            cond_emb = None

        pair_emb = None
        if self.use_attn_pair_bias:
            with torch.no_grad():  # don't want to backprop through this
                # pairwise distance matrix
                dist_matrix = torch.cdist(x_in, x_in)[..., None]
            pair_emb = self.pair_proj(dist_matrix)

        attn_mask = self.attn_mask
        if mask is not None:
            assert mask.shape[:1] == x.shape[:1], (
                f"First two dimensions of mask {mask.shape[:1]} and x {x.shape[:1]} do not match"
            )

            # WARNING there was a permutation of mask here but i can't see what it would do TODO
            attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask * mask[..., None]
            attn_mask = attn_mask.unsqueeze(1)

        attn_mask = attn_mask[..., : x.shape[1], : x.shape[1]]
        for block in self.attn_blocks:
            x = block(x, cond=cond_emb, pair=pair_emb, mask=mask, attn_mask=attn_mask)
            if mask is not None:
                assert x[torch.where(mask == 0)].sum() == 0, "Masked positions are nonzero"

        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # shift one token w/ zero pad
        x = x * mask[..., None] if mask is not None else x  # apply mask if provided

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tokenization_mask = (tokenization_map != -1).float().repeat_interleave(3, dim=-1) # TODO hardcode
            tokenization_mask = self.permutation(tokenization_mask, permutations)
            xb = xb * tokenization_mask
            xa = xa * tokenization_mask
        scale = (-xa.float()).exp().type(xa.dtype)
        x_out = self.permutation((x_in - xb) * scale, permutations, inverse=True)

        if tokenization_map is not None:
            data_dim = (tokenization_map != -1).int().sum(dim=[1, 2]) * 3 # this will inherently account for full padded residue tokens # TODO could be better
        elif mask is not None:
            data_dim = mask.sum(dim=-1) * 3 # TODO ugly and makes assumptions
        else:
            data_dim = x.shape[1] * 3  # assume all tokens are valid

        logdet = -xa.sum(dim=[1, 2]) / data_dim

        return x_out, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        attn_temp: float = 1.0, # TODO remove?
        which_cache: str = "cond", # TODO remove?
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x.clone()
        x = self.proj_in(x_in[:, i : i + 1]) + pos_embed[:, i : i + 1]

        if cond is not None:
            if self.lookahead_conditioning:
                lookahead_cond = torch.cat([cond[:, 1:], torch.zeros_like(cond[:, :1])], dim=1)  # shift back one token w/ zero pad
                cond = torch.cat([cond, lookahead_cond], dim=-1)  # concatenate the two
            cond_in = cond[:, i : i + 1]
            cond_emb = self.proj_cond(cond_in)
        else:
            cond_emb = None

        pair_emb = None
        if self.use_attn_pair_bias:
            # pairwise distance row
            with torch.no_grad():  # don't want to backprop through this
                dist_matrix = torch.cdist(x_in[:, : i + 1], x_in[:, : i + 1])[..., None]
                dist_row = dist_matrix[:, i : i + 1]
            pair_emb = self.pair_proj(dist_row)

        for block in self.attn_blocks:
            x = block(
                x, cond=cond_emb, pair=pair_emb, mask=None, attn_mask=None, attn_temp=attn_temp, which_cache=which_cache
            )  # here we use kv caching, so no attn_mask

        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {"cond": [], "uncond": []}
                m.v_cache = {"cond": [], "uncond": []}

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.permutation(x, permutations)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]][None, ...]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid

        if cond is not None:
            cond = self.permutation(cond, permutations)

        self.set_sample_mode(True)
        xs = [x[:, i] for i in range(x.size(1))]
        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tokenization_mask = (tokenization_map != -1).float().repeat_interleave(3, dim=-1) # TODO hardcode
            tokenization_mask = self.permutation(tokenization_mask, permutations)
            tokenization_masks = [tokenization_mask[:, i] for i in range(tokenization_mask.size(1))]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, cond, pos_embed, i, which_cache="cond")
            if tokenization_map is not None:
                zb = zb * tokenization_masks[i+1]
                za = za * tokenization_masks[i+1]
            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            xs[i + 1] = xs[i + 1] * scale + zb[:, 0]
            x = torch.stack(xs, dim=1)

        self.set_sample_mode(False)
        x = self.permutation(x, permutations, inverse=True)

        return x

def atoms_to_residue_tokens(atom_tokens: torch.Tensor, residue_tokenization: torch.Tensor):
    """
    Args:
        atom_tokens: [B, Seq_len, 3]
        residue_tokenization: [B, Num_AA, max_atoms_per_residue], with -1 for padding

    Returns:
        residue_tokens: [B, Num_AA, 3 * max_atoms_per_residue]
        residue_mask:   [B, Num_AA, max_atoms_per_residue], 1 for real atom, 0 for padding
    """
    B, N, D = atom_tokens.shape
    B_check, num_residues, max_atoms_per_residue = residue_tokenization.shape
    assert B == B_check, "Batch size mismatch"

    # Mask for valid (non-padding) atoms
    mask = (residue_tokenization != -1).float()  # [B, Num_AA, max_atoms_per_residue]

    # Replace -1 with 0 for safe gather
    safe_indices = residue_tokenization.clone()
    safe_indices[safe_indices == -1] = 0  # dummy placeholder, will be masked

    # Gather atom positions: [B, Num_AA, max_atoms_per_residue, 3]
    gathered = torch.gather(
        atom_tokens.unsqueeze(1).expand(-1, num_residues, -1, -1),  # [B, Num_AA, Seq_len, 3]
        dim=2,
        index=safe_indices.unsqueeze(-1).expand(-1, -1, -1, D)      # [B, Num_AA, max_atoms_per_residue, 3]
    )

    # Mask out padding tokens
    gathered = gathered * mask.unsqueeze(-1)

    # Flatten: [B, Num_AA, 3 * max_atoms_per_residue]
    residue_tokens = gathered.view(B, num_residues, D * max_atoms_per_residue)

    return residue_tokens

def residue_to_atom_tokens(
    residue_tokens: torch.Tensor,
    residue_tokenization: torch.Tensor,
    max_seq_len: int
) -> torch.Tensor:
    """
    Vectorized inversion of atoms_to_residue_tokens with support for padded MAX_SEQUENCE_LENGTH.

    Args:
        residue_tokens: [B, Num_AA, 3 * max_atoms_per_residue]
        residue_tokenization: [B, Num_AA, max_atoms_per_residue], with -1 for padding
        max_sequence_length: int, length of original padded atom sequence (e.g., MAX_SEQUENCE_LENGTH)

    Returns:
        atom_tokens: [B, MAX_SEQUENCE_LENGTH, 3], with padded zeros where original input had no atoms
    """
    B, Num_AA, flat_dim = residue_tokens.shape
    D = 3
    max_atoms_per_residue = flat_dim // D

    # Unflatten residue tokens
    residue_tokens = residue_tokens.view(B, Num_AA, max_atoms_per_residue, D)  # [B, R, A, 3]

    # Valid atom mask
    mask = (residue_tokenization != -1)  # [B, R, A]

    # Sanitize indices for scatter
    safe_indices = residue_tokenization.clone()
    safe_indices[~mask] = 0  # dummy for scatter (will be masked)

    # Prepare output tensor with zeros (padded)
    atom_tokens = torch.zeros(B, max_seq_len, D, device=residue_tokens.device)

    # Flatten for scatter
    scatter_idx = safe_indices.view(B, -1)              # [B, R*A]
    scatter_vals = residue_tokens.view(B, -1, D)        # [B, R*A, 3]
    scatter_mask = mask.view(B, -1)                     # [B, R*A]

    for b in range(B):
        valid_indices = scatter_idx[b][scatter_mask[b]]  # [#valid_atoms]
        valid_values = scatter_vals[b][scatter_mask[b]]  # [#valid_atoms, 3]
        atom_tokens[b].index_copy_(0, valid_indices, valid_values)

    return atom_tokens

def atom_to_residue_encoding(encoding: torch.Tensor, tokenization_map: torch.Tensor):
    c_index_per_residue = tokenization_map[:, :, 0] # get index of "C" atom in each residue
    c_index_per_residue_mask = c_index_per_residue != -1 # mask for valid residues
    c_index_per_residue[c_index_per_residue_mask == 0] = 0 # set invalid residues to 0
    B, R = c_index_per_residue.shape
    batch_indices = torch.arange(B, device=c_index_per_residue.device).unsqueeze(1).expand(B, R)
    return encoding[batch_indices, c_index_per_residue] * c_index_per_residue_mask

class TarFlow(torch.nn.Module): # rename to AtomTarFlow?
    def __init__(
        self,
        input_dimension: int,
        max_num_tokens: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        use_adapt_ln: bool = False,
        use_attn_pair_bias: bool = False,
        pair_bias_hidden_dim: int = 16,
        use_transition: bool = False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        permutation_keys: list[str] = ["n2c_residue-by-residue_standard_group-by-group", "n2c_residue-by-residue_standard_group-by-group_flip"], # defaults to SBG
        cond_embed: nn.Module | None = None, # TODO don't like name, could make a proper subclass 
        pos_embed_type: str = "learned",  # learned, sinusoidal
        nvp: bool = True,
        debug: bool = False,  # stops the weight initialization from being zero so tokens are not all the same
        lookahead_conditioning: bool = False,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        permutation_keys = list(permutation_keys) * (num_blocks // len(permutation_keys) + 1)  # repeat to match num_blocks
        self.conditional = False if cond_embed is None else True
        self.cond_embed = cond_embed
        self.debug = debug

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    input_dimension,
                    channels,
                    max_num_tokens,
                    PermutationFromDict(permutation_keys[i]),
                    layers_per_block,
                    head_dim=head_dim,
                    nvp=nvp,
                    use_adapt_ln=use_adapt_ln,
                    use_attn_pair_bias=use_attn_pair_bias,
                    pair_bias_hidden_dim=pair_bias_hidden_dim,
                    use_transition=use_transition,
                    use_qkln=use_qkln,
                    dropout=dropout,
                    conditional=self.conditional,
                    pos_embed_type=pos_embed_type,
                    debug=debug,
                    lookahead_conditioning=lookahead_conditioning,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)


    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        mask = permutations.get("mask", None)

        if mask is not None:
            assert mask.ndim == 2, "Mask should be 2D"
            assert torch.all(x.sum(dim=-1)[mask == 0] == 0), "x is not zero where mask is zero"
            mask = mask.view(x.shape[0], -1)  # needs to be this shape for embedder

        if self.conditional:
            assert encoding is not None, "Encoding must be provided for conditional model."
            if mask is not None:
                for key in encoding.keys():
                    if not key == "seq_len":  # seq_len is not a tensor, so we don't check it
                        assert torch.all(encoding[key][mask == 0] == 0), f"{key} is not zero where mask is zero"
            # (batch_size, seq_len, channels)
            cond = self.cond_embed(**encoding, mask=mask)
        else:
            cond = None

        logdets = torch.zeros((), device=x.device)

        for block in self.blocks:
            x, logdet = block(x, permutations, cond=cond, mask=mask)
            logdets = logdets + logdet

        return x, logdets

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        if self.conditional:
            assert encoding is not None, "Encoding must be provided for conditional model."
            assert x.shape[1] == encoding["aa_type"].shape[1], "x and encoding do not match"

            for key in ["atom_type", "aa_type", "aa_pos"]:
                if key in encoding:
                    if not key == "seq_len":  # seq_len is single value for each batch item, so we don't check it
                        assert not torch.any(encoding[key] == 0), f"{key} has padding zeros, padding not supported in reverse"
            cond = self.cond_embed(**encoding)
        else:
            cond = None

        for block in reversed(self.blocks):
            x = block.reverse(x, permutations, cond=cond)

        return x


class AtomTarFlow(TarFlow):
    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:

        batch_size = x.shape[0]
        permutations = permutations["atom"]

        # patichy to (batch_size, num_atoms, input_dimension)
        x = x.reshape(batch_size, -1, self.input_dimension)

        x, logdets = super().forward(x, permutations, encoding)

        # un-patchify
        x = x.reshape(batch_size, -1)

        return x, logdets

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        batch_size = x.shape[0]
        permutations = permutations["atom"]

        # patchify to (batch_size, num_atoms, input_dimension)
        x = x.reshape(batch_size, -1, self.input_dimension)

        x = super().reverse(x, permutations, encoding)

        # un-patchify
        x = x.reshape(batch_size, -1)

        return x

class ResidueTarFlow(TarFlow):

    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 3) # TODO remove
        atom_seq_len = x.shape[1]

        permutations = permutations["residue"]

        # patichy to (batch_size, max_residues, max_atoms_per_residue)
        x_start = x.clone()
        x = atoms_to_residue_tokens(x, permutations["tokenization_map"])
        # x_new = residue_to_atom_tokens(x, permutations["tokenization_map"], max_seq_len=atom_seq_len)
        # assert torch.allclose(x_new, x_start), "Reshape from residue to atom tokens did not preserve original x"

        residue_encoding = {
            "aa_type": atom_to_residue_encoding(encoding["aa_type"], permutations["tokenization_map"]),
            "seq_len": encoding["seq_len"],
        }

        x, logdets = super().forward(x, permutations, residue_encoding)

        # un-patchify
        x = residue_to_atom_tokens(
            x, permutations["tokenization_map"], max_seq_len=atom_seq_len
        )

        x = x.reshape(batch_size, -1) # TODO remove

        return x, logdets

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encoding: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 3) # TODO remove
        atom_seq_len = x.shape[1]

        permutations = permutations["residue"]

        # patichy to (batch_size, max_residues, max_atoms_per_residue)
        x_start = x.clone()
        x = atoms_to_residue_tokens(x, permutations["tokenization_map"])
        # x_new = residue_to_atom_tokens(x, permutations["tokenization_map"], max_seq_len=atom_seq_len)
        # assert torch.allclose(x_new, x_start), "Reshape from residue to atom tokens did not preserve original x"

        residue_encoding = {
            "aa_type": atom_to_residue_encoding(encoding["aa_type"], permutations["tokenization_map"]),
            "seq_len": encoding["seq_len"],
        }

        x = super().reverse(x, permutations, encoding=residue_encoding)

        # un-patchify
        x = residue_to_atom_tokens(
            x, permutations["tokenization_map"], max_seq_len=atom_seq_len
        )

        x = x.reshape(batch_size, -1) # TODO remove

        return x 

class MultiTarFlow(torch.nn.Module):
    """
    """

    def __init__(self, atom_model=None, residue_model=None):
        super().__init__()

        models = []
        if atom_model is not None:
            models.append(atom_model)
        if residue_model is not None:
            models.append(residue_model)

        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor, permutations: dict[str, torch.Tensor], encoding: dict[str, torch.Tensor] | None = None):
        logdets = torch.zeros((x.shape[0]), device=x.device)
        for model in self.models:
            x, logdet = model(x, permutations, encoding)
            logdets += logdet

        atom_mask = permutations["atom"].get("mask", None) # TODO remove
        if atom_mask is not None: 
            logdets = logdets / (atom_mask.sum(dim=-1) * 3)

        return x, logdets

    def reverse(self, x: torch.Tensor, permutations: dict[str, torch.Tensor], encoding: dict[str, torch.Tensor] | None = None):
        for model in reversed(self.models):
            x = model.reverse(x, permutations, encoding)
        return x

########################################################
""" Below are helper functions for testing the model """
########################################################


def load_padded_model_weights(model_pad, model):
    state_dict = model_pad.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model.state_dict():
            target_shape = model.state_dict()[k].shape
            if v.shape != target_shape:
                print(f"Loading {k} with shape {v.shape} into model with shape {target_shape}")
                # Slice to match target shape
                slices = tuple(slice(0, s) for s in target_shape)
                new_state_dict[k] = v[slices]
            else:
                new_state_dict[k] = v

    # Load sliced weights into smaller model
    model.load_state_dict(new_state_dict, strict=True)

    return model

@torch.no_grad()
def test_invertibility(model, x, permutations, encoding, mask=None, num_pad_tokens=2, num_dimensions=3):
    x_pred, _ = model(x, permutations, encoding=encoding)

    # print("x_pred", x_pred[0])

    if mask is not None:
        x = x[:, : -num_pad_tokens * num_dimensions]
        x_pred = x_pred[:, : -num_pad_tokens * num_dimensions]

        encoding = {
            "atom_type": encoding["atom_type"][:, :-num_pad_tokens],
            "aa_type": encoding["aa_type"][:, :-num_pad_tokens],
            "aa_pos": encoding["aa_pos"][:, :-num_pad_tokens],
            "seq_len": encoding["seq_len"],
        }

        permutations = {
            k: v[:, : -num_pad_tokens]
            for k, v in permutations.items()
        }

    x_recon = model.reverse(x_pred, permutations, encoding=encoding)

    x = x[0:8]
    x_recon = x_recon[0:8]
    x_pred = x_pred[0:8]

    print(torch.abs(x - x_recon).mean(dim=0).reshape(8, -1, 3))
    print("mae:", torch.abs(x - x_recon).mean())

    # print(torch.abs(x - x_pred).mean(dim=0).reshape(8, -1, 3))
    # print("mae:", torch.abs(x - x_pred).mean())
    # x = x[0]
    # x_recon = x_recon[0]
    # print(torch.abs(x - x_recon).reshape(-1, 3))


    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=0))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=1))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=2))
    # breakpoint()

    # Helpful prints for debugging
    # I often found it's clear that source of error is a few token positions

    # print(x[0, 0:8])
    # print(x_recon[0, 0:8])
    # print("mae:", torch.abs(x - x_recon).mean())
    # print("mse:", torch.mean((x - x_recon) ** 2))
    # print("max abs:", torch.max(abs(x - x_recon)))
    # print("position wise MAE", torch.abs(x - x_recon).mean(dim=0))

    assert torch.allclose(x, x_recon, atol=1e-4), "Invertibility test failed"
    print("Invertibility test passed")


def test_mask_model(model, x, permutations, encoding, model_pad, x_pad, permutations_pad, encoding_pad, mask):
    x_fwd, _ = model(x, permutations, encoding=encoding)
    x_fwd_pad, _ = model_pad(x_pad, permutations_pad, encoding=encoding_pad, mask=mask)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_pad[:, : x_fwd.shape[1]])))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_pad[:, : x_fwd.shape[1]])))

    assert torch.allclose(x_fwd, x_fwd_pad[:, : x_fwd.shape[1]], atol=1e-6), "Models do not generate the same x_fwd"

    print("Masked model fwd test passed")


def test_mask_model_no_pad(model, x, permutations, encoding, model_pad):
    x_fwd, _ = model(x, permutations, encoding=encoding)
    x_fwd_no_pad, _ = model_pad(x, permutations, encoding=encoding)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_no_pad)))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_no_pad)))

    assert torch.allclose(x_fwd, x_fwd_no_pad, atol=1e-4), "Models do not generate the same x_fwd"
    print("No pad model fwd test passed")


@torch.no_grad()
def test_logdet(model, x_i, permutations_i, enc_i):
    x_pred = model.reverse(x_i, permutations_i, encoding=enc_i)
    _, fwd_logdets = model(x_pred, permutations_i, encoding=enc_i)
    fwd_logdets = fwd_logdets * x_i.shape[1]  # rescale from mean to sum

    reverse_func = lambda x: model.reverse(x=x, permutations=permutations_i, encoding=enc_i)
    rev_jac_true = torch.autograd.functional.jacobian(reverse_func, x_i, vectorize=True)
    rev_logdets_true = torch.logdet(rev_jac_true[0].squeeze())

    logdets_diff = torch.mean(abs(-fwd_logdets - rev_logdets_true))
    assert torch.allclose(-fwd_logdets, rev_logdets_true, atol=1e-4), f"Log Dets Diff: {logdets_diff}"
    print("Log det test passed")


@torch.no_grad()
def test_logdet_mask(model, model_pad, x_i, permutations_i, permutations_pad_i, enc_i, enc_i_pad, mask_i, num_pad_tokens=2, num_dimensions=3):
    # TODO i don't think you need the non-pad model in this function
    x_pred = model.reverse(x_i, permutations_i, encoding=enc_i)
    x_pred_pad = model_pad.reverse(x_i, permutations_i, encoding=enc_i)

    print("x_pred max error:", torch.max(abs(x_pred - x_pred_pad)))
    print("x_pred mae:", torch.mean(abs(x_pred - x_pred_pad)))
    assert torch.allclose(x_pred, x_pred_pad, atol=1e-6), "Models do not generate the same x_pred"

    _, fwd_logdets = model_pad(x_pred, permutations_i, encoding=enc_i)
    fwd_logdets = fwd_logdets * x_i.shape[1]  # rescale from mean to sum

    # pad the output for the forward call
    x_pred_pad = torch.cat([x_pred_pad, torch.zeros_like(x_pred_pad[:, : num_pad_tokens * num_dimensions])], dim=1)
    _, fwd_logdets_pad = model_pad(x_pred_pad, permutations_pad_i, encoding=enc_i_pad, mask=mask_i)
    fwd_logdets_pad = fwd_logdets_pad * mask_i.sum(dim=-1) * num_dimensions  # rescale from mean to sum

    logdets_diff = torch.mean(abs(fwd_logdets - fwd_logdets_pad))
    print(f"fwd_logdets: {fwd_logdets.item()}")
    print(f"Log Dets Diff: {logdets_diff}")
    assert torch.allclose(fwd_logdets, fwd_logdets_pad, atol=1e-4), f"Log Dets Diff: {logdets_diff}"
    print("Masked log det test passed")


if __name__ == "__main__":
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.set_printoptions(sci_mode=True, precision=2)
    torch.manual_seed(1)

    x = torch.load("debug_prior_samples.pt")
    permutations = torch.load("debug_permutations.pt")
    encoding = torch.load("debug_encoding.pt")  



    model = ResidueTarFlow(
        input_dimension=246,  # 82 * 3 atom types
        channels=768,
        max_num_tokens=8,
        num_blocks=4,
        layers_per_block=4,
        permutation_keys=["n2c", "c2n"],
        cond_embed=ResidueConditionalEmbedder(
            hidden_dim=384,
            output_dim=768,
        ),
        use_adapt_ln=True,
        use_transition=True,
        use_qkln=True,
        pos_embed_type="sinusoidal",
        debug=True,
        lookahead_conditioning=True, 
    )

    model = model.cuda()

    print("\nstandard")
    test_invertibility(
        model, x, permutations, encoding,
    )

    for i in range(16):
        print("\nbatch item", i)

        x_i = x[i : i + 1]

        permutations_i = {
            "atom": {
                "permutations": {k: v[i : i + 1] for k, v in permutations["atom"]["permutations"].items()}
            },
            "residue": {
                "permutations": {k: v[i : i + 1] for k, v in permutations["residue"]["permutations"].items()},
                "tokenization_map": permutations["residue"]["tokenization_map"][i : i + 1],
            },
        }

        enc_i = {k: v[i : i + 1] for k, v in encoding.items()}

        test_logdet(model, x_i, permutations_i, enc_i)  # test logdet of the original model



    #                print("\npad + mask")
    #                test_mask_model(
    #                    model, x, permutations, encoding, model_pad, x_pad, permutations_pad, encoding_pad, mask
    #                )  # test forward of the padded model
    #                test_invertibility(
    #                    model_pad, x_pad, permutations_pad, encoding_pad, mask, num_pad_tokens=pad_tokens
    #                )  # test invertibility of the padded model

    #                print("\npad model with non-pad data")
    #                test_mask_model_no_pad(model, x, permutations, encoding, model_pad)  # test forward of the padded model
    #                test_invertibility(
    #                    model_pad, x, permutations, encoding, num_pad_tokens=pad_tokens
    #                )  # test invertibility of the padded model with non-padded data

    #                    print("\npad + mask")
    #                    test_logdet_mask(
    #                        model, model_pad, x_i, permutations_i, permutations_pad_i, enc_i, enc_pad_i, mask_i, num_pad_tokens=pad_tokens
    #                    )  # test logdet of the padded model

    #                    print("\npad model with non-pad data")
    #                    test_logdet(model_pad, x_i, permutations_i, enc_i)  # test logdet of the padded model with non-padded data
