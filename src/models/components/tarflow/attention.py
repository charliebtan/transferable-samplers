import torch
from torch import einsum
from einops import rearrange

def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int, use_qkln: bool = True, dropout: float = 0.0):
        assert in_channels % head_channels == 0
        super().__init__()
        self.num_heads = in_channels // head_channels
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.bias_proj = torch.nn.Linear(in_channels, self.num_heads, bias=False)
        # self.gate_proj = torch.nn.Linear(in_channels, in_channels)
        self.pair_norm = torch.nn.LayerNorm(in_channels)
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False
        self.dropout = dropout

        self.q_layer_norm = torch.nn.LayerNorm(in_channels) if use_qkln else torch.nn.Identity()
        self.k_layer_norm = torch.nn.LayerNorm(in_channels) if use_qkln else torch.nn.Identity()

        self.k_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}


    def forward(
        self,
        x,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond"
    ):
        assert exists(self.bias_proj) or not exists(pair)
        x = self.norm(x.float()).type(x.dtype)
        pair = self.pair_norm(pair) if exists(pair) else None
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        # g = self.gate_proj(x)

        bias = (
            rearrange(self.bias_proj(pair), "b ... h -> b h ...")
            if exists(pair)
            else 0.
        )
        
        # q, k, v, g = map(
        #     lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=self.num_heads), (q, k, v, g)
        # )

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=self.num_heads), (q, k, v)
        )

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        x = self._attn(q, k, v, bias, mask, temp)

        # I don't know why there is a sigmoid here in original proteina.
        # It does not show in their figure nor talked about in the paper.
        # This causes the output of the forward to NOT match the outputs of the other
        # forward attention functions (base and spda)
        # x = torch.sigmoid(g) * x
        print(x.shape)
        print(self.num_heads)
        x = rearrange(
            x, "b h n d -> b n (h d)", h=self.num_heads
        )     
        x = self.proj(x)
        return x 

    def _attn(self, q, k, v, bias, mask: torch.Tensor | None = None, temp: float= 1.0):
        """Perform attention update"""
        scale = self.sqrt_scale**2 / temp
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if exists(mask):
            mask = mask.type(torch.bool)
            L, S = q.size(-2), k.size(-2)
            attn_mask = torch.zeros(L, S, device=q.device, dtype=q.dtype)
            attn_mask.masked_fill(~mask, float("-inf"))
            sim += attn_mask

        attn = torch.softmax(sim + bias, dim=-1)
        attn = torch.dropout(attn, self.dropout, train=True)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


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
    def __init__(
        self,
        channels: int,
        head_channels: int,
        expansion: int = 4,
        use_qkln: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(channels, head_channels, use_qkln=use_qkln, dropout=dropout)
        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        if cond is not None:
            x = x + cond

        x = x * mask[..., None]
        x = x + self.attention(x, pair=pair, mask=attn_mask, temp=attn_temp, which_cache=which_cache)
        x = x + self.mlp(x)
        return x * mask[..., None]


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(sci_mode=True, precision=2)
    torch.manual_seed(1)

    attn = Attention(128, 64, use_qkln=False)

    x = torch.randn((128, 10, 128))
    mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool))
    y = attn(x, mask)

    attn.USE_SPDA = False
    z = attn(x, mask)

    error = (abs(y - z)).mean()
    print(f"Error between SPDA and base: {error}")
    assert torch.allclose(y, z, atol=1e-6), f"Error: {error}"


    w = attn(x, mask=mask, use_proteina=True)
    assert not torch.isnan(w).any()
    assert not torch.isinf(w).any()

    error_base_pro = (abs(z - w).mean())
    error_spda_pro = (abs(y - w)).mean()
    print(f"Error between proteina and base: {error_base_pro}")
    print(f"Error between proteina and SPDA: {error_spda_pro}")
    # should pass if x is not multiplied with g
    assert torch.allclose(y, w, atol=1e-7)

    pair = torch.randn((128, 10, 10, 128))
    w = attn(x, pair=pair, mask=mask, use_proteina=True)
    assert not torch.allclose(w, z)