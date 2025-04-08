#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch

from .embed import ConditionalEmbedder


class Permutation(torch.nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError("Overload me")


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


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
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


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
        num_classes: int = 0,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)

        if conditional:
            self.proj_cond = torch.nn.Linear(channels, channels)

        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer("attn_mask", torch.tril(torch.ones(num_patches, num_patches)))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed

        if cond is not None:
            cond = self.permutation(cond)
            cond_emb = self.proj_cond(cond)
            x = x + cond_emb

        attn_mask = self.attn_mask
        if mask is not None:
            assert mask.shape[:1] == x.shape[:1], (
                f"First two dimensions of mask {mask.shape[:1]} and x {x.shape[:1]} do not match"
            )
            mask = self.permutation(mask)
            attn_mask = attn_mask.unsqueeze(0) * mask
            attn_mask = attn_mask.unsqueeze(1)
            x = x * mask

        for block in self.attn_blocks:
            x = block(x, attn_mask)
            if mask is not None:
                x = x * mask
                assert x[torch.where(mask == 0)].sum() == 0, "Masked positions are nonzero"

        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        scale = (-xa.float()).exp().type(xa.dtype)
        return self.permutation((x_in - xb) * scale, inverse=True), -xa.mean(dim=[1, 2])

    def reverse_step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x[:, i : i + 1]  # get i-th patch but keep the sequence dimension
        x = self.proj_in(x_in) + pos_embed[i : i + 1]

        if cond is not None:
            cond_in = cond[:, i : i + 1]
            cond_emb = self.proj_cond(cond_in)
            x = x + cond_emb

        if mask is not None:
            mask = mask[:, i : i + 1]

        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)  # here we use kv caching, so no attn_mask
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
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        if cond is not None:
            cond = self.permutation(cond)
        if mask is not None:
            mask = self.permutation(mask)

        self.set_sample_mode(True)
        # 512 x 8 x 1
        xs = [x[:, i] for i in range(x.size(1))]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, cond, pos_embed, i, mask, which_cache="cond")
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
        conditional: bool = False,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # self.num_patches = (img_size // patch_size) ** 2
        self.num_patches = img_size // patch_size // in_channels
        permutations = [
            PermutationIdentity(self.num_patches),
            PermutationFlip(self.num_patches),
        ]

        if conditional:
            self.cond_embed = ConditionalEmbedder(channels=channels)

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
                    nvp=nvp,
                    conditional=conditional,
                    num_classes=num_classes,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        # prior for nvp mode should be all ones, but needs to be learnd for the vp mode
        self.register_buffer("var", torch.ones(self.num_patches, in_channels * patch_size**2))
        self.in_channels = in_channels
        self.channels = channels
        self.img_size = img_size
        self.conditional = conditional
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
        # patchify
        x = x.reshape(-1, self.img_size // self.in_channels, self.in_channels)

        cond = None
        if encodings is not None:
            assert self.conditional, (
                f"Passed in encodings for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )

            # (batch_size, seq_len, channels)
            cond = self.cond_embed(
                atom_type=encodings["atom_type"], aa_type=encodings["aa_type"], aa_pos=encodings["aa_pos"]
            )

        logdets = torch.zeros((), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, cond, mask)
            logdets = logdets + logdet

        # un-patch
        x_pred = x.reshape(-1, self.img_size)
        return x_pred, logdets

    def reverse(
        self,
        x: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        # patchify
        x = x.reshape(-1, self.img_size // self.in_channels, self.in_channels)

        cond = None
        if encodings is not None:
            assert self.conditional, (
                f"Passed in encodings for transferrability, but conditional={self.conditional}."
                + " Set conditional attribute to True"
            )
            cond = self.cond_embed(
                atom_type=encodings["atom_type"], aa_type=encodings["aa_type"], aa_pos=encodings["aa_pos"]
            )

        seq = [x.reshape(-1, self.img_size)]
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, cond, mask)
            seq.append(x.reshape(-1, self.img_size))

        # un-patch
        x = x.reshape(-1, self.img_size)
        if not return_sequence:
            return x
        else:
            return seq


if __name__ == "__main__":
    torch.manual_seed(1)

    img_size = 66
    in_channels = 3
    cond_in_channels = None
    patch_size = 1
    channels = 64
    num_blocks = 3
    layers_per_block = 2
    batch_size = 32

    model = TarFlow(in_channels, img_size, patch_size, channels, num_blocks, layers_per_block, cond_in_channels)

    x = torch.randn([batch_size, img_size])
    cond = None
    x_pred, _ = model.forward(x, cond)
    x_recon = model.reverse(x_pred, cond)

    print(torch.abs(x - x_recon).mean())
    print(torch.mean((x - x_recon) ** 2))
    print(torch.max(abs(x - x_recon)))

    assert torch.allclose(x, x_recon, atol=1e-7), "Invertibility test failed"
    print("Invertibility test passed")

    for i in range(16):
        x_i = x[i : i + 1]
        with torch.no_grad():
            x_pred = model.reverse(x_i)
            x_recon, fwd_logdets = model(x_pred)
            fwd_logdets = fwd_logdets * img_size  # rescale from mean to sum

        rev_jac_true = torch.autograd.functional.jacobian(model.reverse, x_i, vectorize=True)
        rev_logdets_true = torch.logdet(rev_jac_true.squeeze())

        assert torch.allclose(-fwd_logdets, rev_logdets_true)
    print("logdet test passed")

    print("\nTest for Conditional TarFlow")

    encodings = {
        "atom_type": torch.ones(batch_size, img_size // in_channels, dtype=torch.long),
        "aa_type": torch.ones(batch_size, img_size // in_channels, dtype=torch.long),
        "aa_pos": torch.arange(0, img_size // in_channels, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1),
    }

    model = TarFlow(in_channels, img_size, patch_size, channels, num_blocks, layers_per_block, conditional=True)

    x = torch.randn([batch_size, img_size])
    x_pred, _ = model.forward(x, encodings=encodings)
    x_recon = model.reverse(x_pred, encodings=encodings)

    print(torch.abs(x - x_recon).mean())
    print(torch.mean((x - x_recon) ** 2))
    print(torch.max(abs(x - x_recon)))

    assert torch.allclose(x, x_recon, atol=1e-7), "Invertibility test failed"
    print("Invertibility test passed")

    for i in range(16):
        x_i = x[i : i + 1]
        enc_i = {k: v[i : i + 1] for k, v in encodings.items()}
        with torch.no_grad():
            x_pred = model.reverse(x_i, enc_i)
            x_recon, fwd_logdets = model(x_pred, enc_i)
            fwd_logdets = fwd_logdets * img_size  # rescale from mean to sum

        reverse_func = lambda x: model.reverse(x=x, encodings=enc_i)
        rev_jac_true = torch.autograd.functional.jacobian(reverse_func, x_i, vectorize=True)
        rev_logdets_true = torch.logdet(rev_jac_true[0].squeeze())

        logdets_diff = torch.mean(abs(-fwd_logdets - rev_logdets_true))
        assert torch.allclose(-fwd_logdets, rev_logdets_true, atol=1e-7), f"Log Dets Diff: {logdets_diff}"
    print("logdet test passed")

    print("\nTest for Conditional TarFlow w/ Mask")
    from torch.nn.functional import pad

    img_size1 = 33
    img_size2 = 66

    img_size = max(img_size1, img_size2)
    in_channels = 3
    cond_in_channels = channels
    patch_size = 1
    channels = 64
    num_blocks = 3
    layers_per_block = 2
    batch_size = 16

    x1 = torch.randn([batch_size // 2, img_size1])
    x2 = torch.randn([batch_size // 2, img_size2])

    x1 = pad(x1, (0, img_size - img_size1), "constant", 0)
    x2 = pad(x2, (0, img_size - img_size2), "constant", 0)

    x = torch.concat([x1, x2], axis=0)
    mask = (x != 0).reshape(-1, img_size // in_channels, in_channels).any(dim=-1, keepdim=True).type(torch.float32)

    encodings = {
        "atom_type": torch.ones(batch_size, img_size // in_channels, dtype=torch.long),
        "aa_type": torch.ones(batch_size, img_size // in_channels, dtype=torch.long),
        "aa_pos": torch.arange(0, img_size // in_channels, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1),
    }

    model = TarFlow(in_channels, img_size, patch_size, channels, num_blocks, layers_per_block, conditional=True)

    x_pred, _ = model.forward(x, encodings, mask)
    x_recon = model.reverse(x_pred, encodings, mask)

    print(torch.abs(x - x_recon).mean())
    print(torch.mean((x - x_recon) ** 2))
    print(torch.max(abs(x - x_recon)))

    assert torch.allclose(x, x_recon, atol=1e-7), "Invertibility test failed"
    print("Invertibility test passed")

    for i in range(16):
        x_i = x[i : i + 1]
        enc_i = {k: v[i : i + 1] for k, v in encodings.items()}
        mask_i = mask[i : i + 1]
        with torch.no_grad():
            x_pred = model.reverse(x_i, enc_i, mask_i)
            x_recon, fwd_logdets = model(x_pred, enc_i, mask_i)
            fwd_logdets = fwd_logdets * img_size  # rescale from mean to sum

        reverse_func = lambda x: model.reverse(x=x, encodings=enc_i, mask=mask_i)
        rev_jac_true = torch.autograd.functional.jacobian(reverse_func, x_i, vectorize=True)
        rev_logdets_true = torch.logdet(rev_jac_true[0].squeeze())

        logdets_diff = torch.mean(abs(-fwd_logdets - rev_logdets_true))
        assert torch.allclose(-fwd_logdets, rev_logdets_true, atol=1e-7), print(f"Log Dets Diff: {logdets_diff}")
    print("logdet test passed")
