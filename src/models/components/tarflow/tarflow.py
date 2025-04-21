#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch

if __name__ == "__main__":
    # This is when we run the script directly to test model
    from adaptive_blocks import AdaptiveAttnAndTransition
    from attention import Attention, AttentionBlock
    from embed import ConditionalEmbedder
else:
    from src.models.components.tarflow.adaptive_blocks import AdaptiveAttnAndTransition
    from src.models.components.tarflow.attention import Attention, AttentionBlock
    from src.models.components.tarflow.embed import ConditionalEmbedder


class Permutation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError("Overload me")


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        if isinstance(dim, int):
            dim = [dim]
        return x.flip(dims=dim)


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
        use_adaln: bool = False,
        use_pair_bias: bool = False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)

        if conditional:
            self.proj_cond = torch.nn.Linear(channels, channels)

        if use_pair_bias:
            self.pair_proj = torch.nn.Linear(1, channels)

        self.use_adaln = use_adaln
        self.use_pair_bias = use_pair_bias
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        attn_block = AdaptiveAttnAndTransition if self.use_adaln else AttentionBlock
        self.attn_blocks = torch.nn.ModuleList(
            [
                attn_block(
                    channels=channels, 
                    head_channels=head_dim, 
                    expansion=expansion,
                    use_qkln=use_qkln,
                    use_pair_bias=use_pair_bias,
                    dropout=dropout,
                )
                    for _ in range(num_layers)]
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
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = self.permutation(x)  # store permuted input for later

        # by permuting after projection + pos_embed sum, we can have the same
        # output with / without padding tokens for PermutationFlip
        # without this, the pos_embed is flipped but then the "first" token
        # pos_embed is applied to a pad token
        x = self.proj_in(x) + self.pos_embed[: x.shape[1]]
        x = self.permutation(x)

        if cond is not None:
            cond = self.permutation(cond)
            cond_emb = self.proj_cond(cond)

        pair_emb = None
        if self.use_pair_bias:
            # pairwise distance matrix
            dist_matrix = torch.cdist(x_in, x_in)[..., None]
            pair_emb = self.pair_proj(dist_matrix)

        attn_mask = self.attn_mask
        if mask is not None:
            assert mask.shape[:1] == x.shape[:1], (
                f"First two dimensions of mask {mask.shape[:1]} and x {x.shape[:1]} do not match"
            )
            mask = self.permutation(mask)

            attn_mask = attn_mask.unsqueeze(0)
            if isinstance(self.permutation, PermutationIdentity):
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

        if isinstance(self.permutation, PermutationFlip):
            x = x * mask[..., None] if mask is not None else x
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # shift one token w/ zero pad
        if isinstance(self.permutation, PermutationIdentity):
            x = x * mask[..., None] if mask is not None else x

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        x_out = self.permutation((x_in - xb) * scale, inverse=True)

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
        x = self.proj_in(x_in[:, i : i + 1]) + pos_embed[i : i + 1]

        if cond is not None:
            cond_in = cond[:, i : i + 1]
            cond_emb = self.proj_cond(cond_in)
        
                
        pair_emb = None
        if self.use_pair_bias:
            # pairwise distance row
            dist_matrix = torch.cdist(x_in[:, :i+1], x_in[:, :i+1])[..., None]
            dist_row = dist_matrix[:, i : i +1]
            pair_emb = self.pair_proj(dist_row)

        for block in self.attn_blocks:
            x = block(
                x, 
                cond=cond_emb, 
                pair=pair_emb, 
                mask=None, 
                attn_mask=None, 
                attn_temp=attn_temp, 
                which_cache=which_cache
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
    ) -> torch.Tensor:
        x = self.permutation(x)

        pos_embed = self.pos_embed[: x.shape[1]]  # slice pos_embed before permutation
        pos_embed = self.permutation(pos_embed, dim=0)

        if cond is not None:
            cond = self.permutation(cond)

        self.set_sample_mode(True)
        xs = [x[:, i] for i in range(x.size(1))]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, cond, pos_embed, i, which_cache="cond")
            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            xs[i + 1] = xs[i + 1] * scale + zb[:, 0]
            x = torch.stack(xs, dim=1)

        self.set_sample_mode(False)
        x = self.permutation(x, inverse=True)

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
        use_adaln: bool = False,
        use_pair_bias: bool = False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        cond_embed: ConditionalEmbedder | None = None,
        nvp: bool = True,
        debug: bool = False,  # stops the weight initialization from being zero so tokens are not all the same
    ):
        assert num_blocks >= 2, "num_blocks must be at least 2 to cover both permutations"
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size // patch_size // in_channels
        permutations = [
            PermutationIdentity(),
            PermutationFlip(),
        ]

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
                    permutations[i % 2],
                    layers_per_block,
                    head_dim=head_dim,
                    nvp=nvp,
                    use_adaln=use_adaln,
                    use_pair_bias=use_pair_bias,
                    use_qkln=use_qkln,
                    dropout=dropout,
                    conditional=self.conditional,
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

    def forward(
        self,
        x: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        batch_size = x.shape[0]

        # patchify
        x = x.reshape(batch_size, -1, self.in_channels)
        if mask is not None:
            assert mask.ndim == 2, "Mask should be 2D"
            assert torch.all(x.sum(dim=-1)[mask == 0] == 0), "x is not zero where mask is zero"
            assert torch.all(encodings["atom_type"][mask == 0] == 0), "atom_type is not zero where mask is zero"
            assert torch.all(encodings["aa_type"][mask == 0] == 0), "aa_type is not zero where mask is zero"
            assert torch.all(encodings["aa_pos"][mask == 0] == 0), "aa_pos is not zero where mask is zero"

            mask = mask.view(x.shape[0], -1)  # needs to be this shape for embedder

        cond = None
        if encodings is not None:
            assert self.conditional, (
                f"Passed in encodings for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )

            # (batch_size, seq_len, channels)
            cond = self.cond_embed(
                atom_type=encodings["atom_type"], aa_type=encodings["aa_type"], aa_pos=encodings["aa_pos"], mask=mask
            )

        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, cond=cond, mask=mask)
            logdets = logdets + logdet

        # un-patch
        x_pred = x.reshape(batch_size, -1)

        return x_pred, logdets

    def reverse(
        self,
        x: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        batch_size = x.shape[0]

        assert x.shape[1] == encodings["atom_type"].shape[1] * self.in_channels, "x and encodings do not match"
        assert not torch.any(encodings["atom_type"] == 0), (
            "atom_type has padding zeros, padding not supports in reverse"
        )
        assert not torch.any(encodings["aa_type"] == 0), "aa_type has padding zeros, padding not supports in reverse"
        assert not torch.any(encodings["aa_pos"] == 0), "aa_pos has padding zeros, padding not supports in reverse"

        # patchify
        x = x.reshape(batch_size, -1, self.in_channels)

        cond = None
        if encodings is not None:
            assert self.conditional, (
                f"Passed in encodings for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )
            cond = self.cond_embed(
                atom_type=encodings["atom_type"], aa_type=encodings["aa_type"], aa_pos=encodings["aa_pos"]
            )

        seq = [x.reshape(batch_size, -1)]
        x = x * self.var.sqrt()[: x.shape[1]]
        for block in reversed(self.blocks):
            x = block.reverse(x, cond=cond)
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
def test_invertibility(model, x, encodings, mask=None, num_pad_tokens=4, num_dimensions=3):
    x = x.reshape(batch_size, -1, in_channels)  # reshape to (batch, seq_len, channels)
    x = x.reshape(batch_size, -1)  # reshape back to original shape
    x_pred, _ = model(x, encodings=encodings, mask=mask)

    if mask is not None:
        x = x[:, : num_pad_tokens * num_dimensions]
        x_pred = x_pred[:, : num_pad_tokens * num_dimensions]

        encodings = {
            "atom_type": encodings["atom_type"][:, :num_pad_tokens],
            "aa_type": encodings["aa_type"][:, :num_pad_tokens],
            "aa_pos": encodings["aa_pos"][:, :num_pad_tokens],
        }

    x_recon = model.reverse(x_pred, encodings=encodings)
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=0))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=1))
    # print((x - x_recon).reshape(x.shape[0], -1, num_dimensions).mean(dim=2))
    # breakpoint()

    # Helpful prints for debugging
    # I often found it's clear that source of error is a few token positions

    print()
    print(x[0, 0:8])
    print(x_recon[0, 0:8])
    print("mae:", torch.abs(x - x_recon).mean())
    print("mse:", torch.mean((x - x_recon) ** 2))
    print("max abs:", torch.max(abs(x - x_recon)))
    print("position wise MAE", torch.abs(x - x_recon).mean(dim=0))

    assert torch.allclose(x, x_recon, atol=1e-6), "Invertibility test failed"
    print("Invertibility test passed")


def test_mask_model(model, x, encodings, model_pad, x_pad, encodings_pad, mask):

    x_fwd, _ = model(x, encodings=encodings)
    x_fwd_pad, _ = model_pad(x_pad, encodings=encodings_pad, mask=mask)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_pad[:, :12])))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_pad[:, :12])))

    assert torch.allclose(x_fwd, x_fwd_pad[:, :12], atol=1e-6), "Models do not generate the same x_fwd"
    print("Masked model fwd test passed")


def test_mask_model_no_pad(model, x, encodings, model_pad):
    x_fwd, _ = model(x, encodings=encodings)
    x_fwd_no_pad, _ = model_pad(x, encodings=encodings)

    # print("x_fwd max error:", torch.max(abs(x_fwd - x_fwd_no_pad)))
    # print("x_fwd mae:", torch.mean(abs(x_fwd - x_fwd_no_pad)))

    assert torch.allclose(x_fwd, x_fwd_no_pad, atol=1e-6), "Models do not generate the same x_fwd"
    print("No pad model fwd test passed")


@torch.no_grad()
def test_logdet(model, x_i, enc_i):
    x_pred = model.reverse(x_i, enc_i)
    _, fwd_logdets = model(x_pred, enc_i)
    fwd_logdets = fwd_logdets * x_i.shape[1]  # rescale from mean to sum

    reverse_func = lambda x: model.reverse(x=x, encodings=enc_i)
    rev_jac_true = torch.autograd.functional.jacobian(reverse_func, x_i, vectorize=True)
    rev_logdets_true = torch.logdet(rev_jac_true[0].squeeze())

    logdets_diff = torch.mean(abs(-fwd_logdets - rev_logdets_true))
    assert torch.allclose(-fwd_logdets, rev_logdets_true, atol=1e-6), f"Log Dets Diff: {logdets_diff}"
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
    assert torch.allclose(fwd_logdets, fwd_logdets_pad, atol=1e-7), f"Log Dets Diff: {logdets_diff}"
    print("Masked log det test passed")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(sci_mode=True, precision=2)
    torch.manual_seed(1)

    batch_size = 128
    img_size = 12
    in_channels = 3
    patch_size = 1
    channels = 64
    num_blocks = 2  # needs to be at least 2 to cover both permutations
    layers_per_block = 1

    ### Dummy data
    x = torch.randn([batch_size, img_size])

    encodings = {
        "atom_type": torch.randint(high=2, size=(batch_size, img_size // in_channels)) + 1,
        "aa_type": torch.randint(high=2, size=(batch_size, img_size // in_channels)) + 1,
        "aa_pos": torch.randint(high=2, size=(batch_size, img_size // in_channels)) + 1,
    }

    ### Padded data with mask

    pad_tokens = 2
    pad_dim = pad_tokens * in_channels

    x_pad = torch.cat([x, torch.zeros([batch_size, pad_dim])], dim=1)
    encodings_pad = {
        "atom_type": torch.cat(
            [encodings["atom_type"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
        "aa_type": torch.cat(
            [encodings["aa_type"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
        "aa_pos": torch.cat(
            [encodings["aa_pos"].clone(), torch.zeros([batch_size, pad_tokens], dtype=torch.long)], dim=1
        ),
    }
    mask = torch.cat(
        [torch.ones([batch_size, img_size // in_channels], dtype=torch.float32), torch.zeros([batch_size, pad_tokens])],
        dim=1,
    )

    cond_embed = ConditionalEmbedder(channels=channels)

    for use_adaln in [False, True]:
        for use_pair_bias in [False, True]:
            print(f"Testing with use_adaln={use_adaln} and use_pair_bias={use_pair_bias}")
            model_pad = TarFlow(
                in_channels,
                img_size + pad_dim,
                patch_size,
                channels,
                num_blocks,
                layers_per_block,
                cond_embed=cond_embed,
                use_adaln=use_adaln,
                use_pair_bias=use_pair_bias,
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
                use_adaln=use_adaln,
                use_pair_bias=use_pair_bias,
                debug=True,
            )
            model = load_padded_model_weights(model_pad, model)

            print("\nstandard")
            test_invertibility(model, x, encodings)  # test invertibility of the original model

            print("\npad + mask")
            test_mask_model(model, x, encodings, model_pad, x_pad, encodings_pad, mask)  # test forward of the padded model
            test_invertibility(model_pad, x_pad, encodings_pad, mask)  # test invertibility of the padded model

            print("\npad model with non-pad data")
            test_mask_model_no_pad(model, x, encodings, model_pad)  # test forward of the padded model
            test_invertibility(model_pad, x, encodings)  # test invertibility of the padded model with non-padded data

            for i in range(batch_size - 1):
                print("\nbatch item", i)

                x_i = x[i : i + 1]
                enc_i = {k: v[i : i + 1] for k, v in encodings.items()}

                x_pad_i = x_pad[i : i + 1]
                enc_pad_i = {k: v[i : i + 1] for k, v in encodings_pad.items()}
                mask_i = mask[i : i + 1]

                print("\nstandard")
                test_logdet(model, x_i, enc_i)  # test logdet of the original model

                print("\npad + mask")
                test_logdet_mask(model, model_pad, x_i, enc_i, enc_pad_i, mask_i)  # test logdet of the padded model

                print("\npad model with non-pad data")
                test_logdet(model_pad, x_i, enc_i)  # test logdet of the padded model with non-padded data
