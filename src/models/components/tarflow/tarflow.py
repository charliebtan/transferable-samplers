#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch

if __name__ == "__main__":
    # This is when we run the script directly to test model
    from adaptive_blocks import AdaptiveAttnAndTransition
    from attention import Attention, AttentionBlock
    from embed import ConditionalEmbedder, SinusoidalEmbedding
    from permutation import (
        Permutation,
        PermutationBackBone,
        PermutationBackBoneFlip,
        PermutationFlip,
        PermutationIdentity,
        PermutationRandom,
        shift_pos,
    )


else:
    from src.models.components.tarflow.adaptive_blocks import AdaptiveAttnAndTransition
    from src.models.components.tarflow.attention import Attention, AttentionBlock
    from src.models.components.tarflow.embed import ConditionalEmbedder, SinusoidalEmbedding
    from src.models.components.tarflow.permutation import (
        Permutation,
        PermutationBackBone,
        PermutationBackBoneFlip,
        PermutationFlip,
        PermutationIdentity,
        PermutationRandom,
        shift_pos,
    )

PERM_SET = (PermutationIdentity, PermutationRandom, PermutationBackBone)
PERM_FLIP_SET = (PermutationFlip, PermutationBackBoneFlip)

MAX_SEQ_LEN = 512


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
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
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)

        if conditional:
            self.proj_cond = torch.nn.Linear(channels, channels)

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
            self.pos_embed = SinusoidalEmbedding(embed_size=channels, max_len=MAX_SEQ_LEN)(torch.arange(MAX_SEQ_LEN))
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
        cond: torch.Tensor,
        atom_type: torch.Tensor,
        aa_type: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = self.permutation(x, atom_type=atom_type, aa_type=aa_type, mask=mask)  # store permuted input for later

        x = self.proj_in(x)
        x = self.permutation(x, atom_type=atom_type, aa_type=aa_type)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid

        # if it is a flip permutation the padding tokens are at the start of the sequence
        # for perfect invertiblity we need to shift the position embeddings such that
        # pos_emb[0] is on the first real token
        if mask is not None and type(self.permutation) in PERM_FLIP_SET:
            pos_embed = pos_embed[None, ...].repeat(x.shape[0], 1, 1)
            pos_embed = shift_pos(pos_embed, mask)

        x = x + pos_embed

        if cond is not None:
            cond = self.permutation(cond, atom_type=atom_type, aa_type=aa_type)
            cond_emb = self.proj_cond(cond)

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

            mask = self.permutation(mask, atom_type=atom_type, aa_type=aa_type)
            attn_mask = attn_mask.unsqueeze(0)
            if type(self.permutation) in PERM_SET:
                # mask out final rows
                attn_mask = attn_mask * mask[..., None]
            else:
                # mask out first columns
                attn_mask = attn_mask * mask[..., None].permute(0, 2, 1)

            attn_mask = attn_mask.unsqueeze(1)

        attn_mask = attn_mask[..., : x.shape[1], : x.shape[1]]
        for block in self.attn_blocks:
            x = block(x, cond=cond_emb, pair=pair_emb, mask=mask, attn_mask=attn_mask)
            if mask is not None:
                assert x[torch.where(mask == 0)].sum() == 0, "Masked positions are nonzero"

        x = self.proj_out(x)

        # hit with mask for the flip perms (no need for if statements)
        x = x * mask[..., None] if mask is not None else x
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # shift one token w/ zero pad
        x = x * mask[..., None] if mask is not None else x  # hit with mask for the non-flip perms

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        x_out = self.permutation((x_in - xb) * scale, atom_type=atom_type, aa_type=aa_type, inverse=True)

        if mask is None:
            logdet = -xa.mean(dim=[1, 2])
        else:
            logdet = -xa.sum(dim=[1, 2]) / (mask.sum(dim=-1) * self.in_channels)

        return x_out, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x.clone()
        x = self.proj_in(x_in[:, i : i + 1]) + pos_embed[:, i : i + 1]

        if cond is not None:
            cond_in = cond[:, i : i + 1]
            cond_emb = self.proj_cond(cond_in)

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
        cond: torch.Tensor | None = None,
        atom_type: torch.Tensor | None = None,
        aa_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.permutation(x, atom_type=atom_type, aa_type=aa_type)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]][None, ...]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid

        if cond is not None:
            cond = self.permutation(cond, atom_type=atom_type, aa_type=aa_type)

        self.set_sample_mode(True)
        xs = [x[:, i] for i in range(x.size(1))]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, cond, pos_embed, i, which_cache="cond")
            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            xs[i + 1] = xs[i + 1] * scale + zb[:, 0]
            x = torch.stack(xs, dim=1)

        self.set_sample_mode(False)
        x = self.permutation(x, atom_type=atom_type, aa_type=aa_type, inverse=True)

        return x


class TarFlow(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
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
        perm_type: str = "standard",  # standard, globloc, random
        cond_embed: ConditionalEmbedder | None = None,
        pos_embed_type: str = "learned",  # learned, sinusoidal
        nvp: bool = True,
        debug: bool = False,  # stops the weight initialization from being zero so tokens are not all the same
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size // in_channels

        if perm_type == "standard":
            permutations = [PermutationIdentity(), PermutationFlip()]
        elif perm_type == "globloc":  # this way the side chains go forwards and backwards alternating
            permutations = [PermutationBackBone(), PermutationFlip(), PermutationBackBoneFlip(), PermutationIdentity()]
        elif perm_type == "random":
            permutations = [PermutationIdentity(), PermutationFlip()]
            self.rand_permutation = PermutationRandom()

        assert num_blocks >= len(permutations), "num_blocks must be greater than number of permutations"
        assert num_blocks % len(permutations) == 0, "num_blocks must be divisible by number of permutations"
        self.perm_type = perm_type

        self.conditional = False if cond_embed is None else True
        self.cond_embed = cond_embed

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    # in_channels * patch_size**2,
                    in_channels * patch_size,
                    channels,
                    self.num_patches,
                    permutations[i % len(permutations)],
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
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer("var", torch.ones(self.num_patches, in_channels * patch_size**2))
        self.in_channels = in_channels
        self.channels = channels
        self.img_size = img_size
        self.conditional = self.conditional
        if self.in_channels != 1:
            if self.img_size % self.in_channels != 0:
                raise ValueError(
                    f"img_size ({self.img_size}) must be divisible by in_channels ({self.in_channels}). "
                    "Ensure that the input dimensions are compatible."
                )
        self.debug = debug

    def forward(
        self,
        x: torch.Tensor,
        encoding: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        batch_size = x.shape[0]
        # patchify
        x = x.reshape(batch_size, -1, self.in_channels)
        if mask is not None:
            assert mask.ndim == 2, "Mask should be 2D"
            assert torch.all(x.sum(dim=-1)[mask == 0] == 0), "x is not zero where mask is zero"
            assert torch.all(encoding["atom_type"][mask == 0] == 0), "atom_type is not zero where mask is zero"
            assert torch.all(encoding["aa_type"][mask == 0] == 0), "aa_type is not zero where mask is zero"
            assert torch.all(encoding["aa_pos"][mask == 0] == 0), "aa_pos is not zero where mask is zero"

            mask = mask.view(x.shape[0], -1)  # needs to be this shape for embedder

        cond = None
        if encoding is not None:
            assert self.conditional, (
                f"Passed in encoding for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )

            # (batch_size, seq_len, channels)
            cond = self.cond_embed(
                atom_type=encoding["atom_type"],
                aa_type=encoding["aa_type"],
                aa_pos=encoding["aa_pos"],
                seq_len=encoding["seq_len"],
                mask=mask,
            )

        logdets = torch.zeros((), device=x.device)
        perm = None

        if self.debug and self.perm_type == "random":
            np.random.seed(0)
            perms = []
            for i in range(len(self.blocks)):
                perm_len = x.shape[1]
                if mask is not None:
                    # Need to ensure the pad tokens do not get mixed in with the perm
                    pad_length = int((~mask.bool())[0].sum())
                    perm = torch.tensor(np.random.permutation(perm_len - pad_length))
                    pad = torch.arange(perm_len - pad_length, perm_len)
                    perm = torch.concat([perm, pad])
                else:
                    perm = torch.tensor(np.random.permutation(perm_len))
                perm = perm[None, ...].repeat(x.shape[0], 1).long()
                perms.append(perm)

        for i, block in enumerate(self.blocks):
            if self.debug and self.perm_type == "random":
                perm = perms[i]

            # random perm -- pass in perm = None
            if (not i % 2) and self.perm_type == "random":
                x, perm = self.rand_permutation(x, mask=mask, perm=perm if self.debug else None)

            # If perm_type == random, Id and IdFlip blocks will do RandomPermutation and RandomFlip
            x, logdet = block(x, cond=cond, mask=mask, atom_type=encoding["atom_type"], aa_type=encoding["aa_type"])
            logdets = logdets + logdet

            # inverse the rand perm -- pass in perm
            if (i % 2) and self.perm_type == "random":
                x, perm = self.rand_permutation(x, mask=mask, perm=perm, inverse=True)

        # un-patch
        x_pred = x.reshape(batch_size, -1)

        return x_pred, logdets

    def reverse(
        self,
        x: torch.Tensor,
        encoding: dict[str, torch.Tensor] | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        batch_size = x.shape[0]

        assert x.shape[1] == encoding["atom_type"].shape[1] * self.in_channels, "x and encoding do not match"
        assert not torch.any(encoding["atom_type"] == 0), "atom_type has padding zeros, padding not supports in reverse"
        assert not torch.any(encoding["aa_type"] == 0), "aa_type has padding zeros, padding not supports in reverse"
        assert not torch.any(encoding["aa_pos"] == 0), "aa_pos has padding zeros, padding not supports in reverse"

        # patchify
        x = x.reshape(batch_size, -1, self.in_channels)

        cond = None
        if encoding is not None:
            assert self.conditional, (
                f"Passed in encoding for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )
            cond = self.cond_embed(
                atom_type=encoding["atom_type"],
                aa_type=encoding["aa_type"],
                aa_pos=encoding["aa_pos"],
                seq_len=encoding["seq_len"],
            )

        seq = [x.reshape(batch_size, -1)]
        x = x * self.var.sqrt()[: x.shape[1]]
        perm = None

        if self.debug and self.perm_type == "random":
            np.random.seed(0)
            perms = []
            for i in range(len(self.blocks)):
                perm = torch.tensor(np.random.permutation(x.shape[1]))
                perm = perm[None, ...].repeat(x.shape[0], 1).long()
                perms.append(perm)

            perms = list(reversed(perms))

        for i, block in enumerate(reversed(self.blocks)):
            if self.debug and self.perm_type == "random":
                perm = perms[i]

            # random perm -- pass in perm = None
            if (not i % 2) and self.perm_type == "random":
                x, perm = self.rand_permutation(x, mask=None, perm=perm if self.debug else None)

            # If perm_type == random, Id and IdFlip blocks will do RandomPermutation and RandomFlip
            x = block.reverse(x, cond=cond, atom_type=encoding["atom_type"], aa_type=encoding["aa_type"])

            # inverse the rand perm -- pass in perm
            if (i % 2) and self.perm_type == "random":
                x, perm = self.rand_permutation(x, mask=None, perm=perm, inverse=True)

            seq.append(x.reshape(batch_size, -1))

        # un-patch
        x = x.reshape(batch_size, -1)

        if not return_sequence:
            return x
        else:
            return seq


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
def test_invertibility(model, x, encoding, mask=None, num_pad_tokens=2, num_dimensions=3):
    x_pred, _ = model(x, encoding=encoding, mask=mask)

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

    x_recon = model.reverse(x_pred, encoding=encoding)

    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=0))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=1))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=2))
    # breakpoint()

    # Helpful prints for debugging
    # I often found it's clear that source of error is a few token positions

    # print()
    # print(x[0, 0:8])
    # print(x_recon[0, 0:8])
    # print("mae:", torch.abs(x - x_recon).mean())
    # print("mse:", torch.mean((x - x_recon) ** 2))
    # print("max abs:", torch.max(abs(x - x_recon)))
    # print("position wise MAE", torch.abs(x - x_recon).mean(dim=0))

    assert torch.allclose(x, x_recon, atol=1e-4), "Invertibility test failed"
    print("Invertibility test passed")


def test_mask_model(model, x, encoding, model_pad, x_pad, encoding_pad, mask):
    x_fwd, _ = model(x, encoding=encoding)
    x_fwd_pad, _ = model_pad(x_pad, encoding=encoding_pad, mask=mask)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_pad[:, : x_fwd.shape[1]])))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_pad[:, : x_fwd.shape[1]])))

    assert torch.allclose(x_fwd, x_fwd_pad[0, : x_fwd.shape[1]], atol=1e-6), "Models do not generate the same x_fwd"

    print("Masked model fwd test passed")


def test_mask_model_no_pad(model, x, encoding, model_pad):
    x_fwd, _ = model(x, encoding=encoding)
    x_fwd_no_pad, _ = model_pad(x, encoding=encoding)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_no_pad)))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_no_pad)))

    assert torch.allclose(x_fwd, x_fwd_no_pad, atol=1e-4), "Models do not generate the same x_fwd"
    print("No pad model fwd test passed")


@torch.no_grad()
def test_logdet(model, x_i, enc_i):
    x_pred = model.reverse(x_i, enc_i)
    _, fwd_logdets = model(x_pred, enc_i)
    fwd_logdets = fwd_logdets * x_i.shape[1]  # rescale from mean to sum

    reverse_func = lambda x: model.reverse(x=x, encoding=enc_i)
    rev_jac_true = torch.autograd.functional.jacobian(reverse_func, x_i, vectorize=True)
    rev_logdets_true = torch.logdet(rev_jac_true[0].squeeze())

    logdets_diff = torch.mean(abs(-fwd_logdets - rev_logdets_true))
    assert torch.allclose(-fwd_logdets, rev_logdets_true, atol=1e-4), f"Log Dets Diff: {logdets_diff}"
    print("Log det test passed")


@torch.no_grad()
def test_logdet_mask(model, model_pad, x_i, enc_i, enc_i_pad, mask_i, num_pad_tokens=2, num_dimensions=3):
    x_pred = model.reverse(x_i, enc_i)
    _, fwd_logdets = model_pad(x_pred, enc_i)
    fwd_logdets = fwd_logdets * x_i.shape[1]  # rescale from mean to sum

    x_pred_pad = model_pad.reverse(x_i, enc_i)

    print("x_pred max error:", torch.max(abs(x_pred - x_pred_pad)))
    print("x_pred mae:", torch.mean(abs(x_pred - x_pred_pad)))
    assert torch.allclose(x_pred, x_pred_pad, atol=1e-6), "Models do not generate the same x_pred"

    # pad the output for the forward call
    x_pred_pad = torch.cat([x_pred_pad, torch.zeros_like(x_pred_pad[:, : num_pad_tokens * num_dimensions])], dim=1)
    _, fwd_logdets_pad = model_pad(x_pred_pad, enc_i_pad, mask_i)
    fwd_logdets_pad = fwd_logdets_pad * mask_i.sum(dim=-1) * num_dimensions  # rescale from mean to sum

    logdets_diff = torch.mean(abs(fwd_logdets - fwd_logdets_pad))
    print(f"fwd_logdets: {fwd_logdets.item()}")
    print(f"Log Dets Diff: {logdets_diff}")
    assert torch.allclose(fwd_logdets, fwd_logdets_pad, atol=1e-4), f"Log Dets Diff: {logdets_diff}"
    print("Masked log det test passed")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(sci_mode=True, precision=2)
    torch.manual_seed(1)

    ### Dummy data

    # alanine atoms
    alanine_atom_types = torch.tensor(
        [
            37,
            18,
            18,
            18,
            2,
            19,
            3,
            20,
            20,
            20,
            1,
            46,
            52,
        ]
    )

    # lysine atoms
    lysine_atom_types = torch.tensor(
        [
            37,
            18,
            18,
            18,
            2,
            19,
            3,
            20,
            20,
            11,
            30,
            28,
            4,
            23,
            21,
            7,
            26,
            27,
            45,
            34,
            35,
            36,
            1,
            46,
            52,
        ]
    )

    atom_type = torch.concat([alanine_atom_types, lysine_atom_types], dim=0)
    atom_type = atom_type.reshape(1, -1)
    aa_type = torch.tensor([1 for _ in range(len(alanine_atom_types))] + [12 for _ in range(len(lysine_atom_types))])
    aa_type = aa_type.reshape(1, -1)
    aa_pos = torch.arange(aa_type.shape[-1])[None, ...] + 1
    x = torch.randn((1, aa_type.shape[-1] * 3))

    # build encoding
    encoding = {
        "atom_type": atom_type,
        "aa_type": aa_type,
        "aa_pos": aa_pos,
        "seq_len": torch.ones(size=(x.shape[0], 1), dtype=torch.long) * 2,
    }

    batch_size = x.shape[0]
    img_size = x.shape[-1]
    in_channels = 3
    patch_size = 1
    channels = 64
    num_blocks = 4
    layers_per_block = 1

    pad_tokens = 2
    pad_dim = pad_tokens * in_channels

    x_pad = torch.cat([x, torch.zeros([batch_size, pad_dim])], dim=1)
    encoding_pad = {
        "atom_type": torch.cat(
            [encoding["atom_type"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
        "aa_type": torch.cat(
            [encoding["aa_type"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
        "aa_pos": torch.cat(
            [encoding["aa_pos"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
        "seq_len": encoding["seq_len"].clone(),
    }
    mask = torch.cat(
        [torch.ones([batch_size, img_size // in_channels], dtype=torch.float32), torch.zeros([batch_size, pad_tokens])],
        dim=1,
    )

    cond_embed = ConditionalEmbedder(channels=channels)

    for use_adapt_ln in [False, True]:
        for use_attn_pair_bias in [False, True]:
            for perm_type in ["standard", "random", "globloc"]:
                for pos_embed_type in ["sinusoidal", "learned"]:
                    print(
                        f"\nTesting with use_adapt_ln={use_adapt_ln} and use_attn_pair_bias={use_attn_pair_bias} "
                        f"and perm_type={perm_type} and pos_embed_type={pos_embed_type} \n"
                    )

                    model_pad = TarFlow(
                        in_channels,
                        img_size + pad_dim,
                        patch_size,
                        channels,
                        num_blocks,
                        layers_per_block,
                        cond_embed=cond_embed,
                        use_adapt_ln=use_adapt_ln,
                        use_attn_pair_bias=use_attn_pair_bias,
                        perm_type=perm_type,
                        pos_embed_type=pos_embed_type,
                        debug=True,
                    )
                    model = TarFlow(
                        in_channels,
                        img_size,
                        patch_size,
                        channels,
                        num_blocks,
                        layers_per_block,
                        cond_embed=cond_embed,
                        use_adapt_ln=use_adapt_ln,
                        use_attn_pair_bias=use_attn_pair_bias,
                        perm_type=perm_type,
                        pos_embed_type=pos_embed_type,
                        debug=True,
                    )
                    model = load_padded_model_weights(model_pad, model)

                    print("\nstandard")
                    test_invertibility(
                        model, x, encoding, num_pad_tokens=pad_tokens
                    )  # test invertibility of the original model

                    print("\npad + mask")
                    test_mask_model(
                        model, x, encoding, model_pad, x_pad, encoding_pad, mask
                    )  # test forward of the padded model

                    test_invertibility(
                        model_pad, x_pad, encoding_pad, mask, num_pad_tokens=pad_tokens
                    )  # test invertibility of the padded model

                    print("\npad model with non-pad data")
                    test_mask_model_no_pad(model, x, encoding, model_pad)  # test forward of the padded model
                    test_invertibility(
                        model_pad, x, encoding, num_pad_tokens=pad_tokens
                    )  # test invertibility of the padded model with non-padded data

                    for i in range(batch_size):
                        print("\nbatch item", i)

                        x_i = x[i : i + 1]
                        enc_i = {k: v[i : i + 1] for k, v in encoding.items()}

                        x_pad_i = x_pad[i : i + 1]
                        enc_pad_i = {k: v[i : i + 1] for k, v in encoding_pad.items()}
                        mask_i = mask[i : i + 1]

                        print("\nstandard")
                        test_logdet(model, x_i, enc_i)  # test logdet of the original model

                        print("\npad + mask")
                        test_logdet_mask(
                            model, model_pad, x_i, enc_i, enc_pad_i, mask_i, num_pad_tokens=pad_tokens
                        )  # test logdet of the padded model

                        print("\npad model with non-pad data")
                        test_logdet(model_pad, x_i, enc_i)  # test logdet of the padded model with non-padded data
