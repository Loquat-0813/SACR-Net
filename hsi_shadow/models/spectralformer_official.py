from __future__ import annotations

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim_head * heads, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, n, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class SpectralFormerTransformer(nn.Module):
    """Official SpectralFormer transformer core with CAF skip fusion."""

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float, num_channel: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.skipcat = nn.ModuleList(
            [nn.Conv2d(num_channel + 1, num_channel + 1, kernel_size=(1, 2), stride=1) for _ in range(max(0, depth - 2))]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "ViT":
            for attn, ff in self.layers:
                x = ff(attn(x, mask=mask))
            return x
        if self.mode != "CAF":
            raise ValueError(f"Unsupported SpectralFormer mode: {self.mode}")
        last_output: list[torch.Tensor] = []
        for layer_idx, (attn, ff) in enumerate(self.layers):
            last_output.append(x)
            if layer_idx > 1:
                x = self.skipcat[layer_idx - 2](torch.cat([x.unsqueeze(3), last_output[layer_idx - 2].unsqueeze(3)], dim=3)).squeeze(3)
            x = ff(attn(x, mask=mask))
        return x


class SpectralFormerBaseline(nn.Module):
    """Shared-protocol adapter of the official GPL-3.0 SpectralFormer backbone.

    The original ``danfenghong/IEEE_TGRS_SpectralFormer`` repository is retained
    in ``external_sources``.  This file ports the official GSE + CAF backbone
    and only replaces the dataset/split driver with the common HanChuan loader.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        dim: int = 64,
        depth: int = 5,
        heads: int = 4,
        mlp_dim: int = 128,
        pool: str = "cls",
        dim_head: int = 16,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        mode: str = "CAF",
        band_patches: int = 3,
    ) -> None:
        super().__init__()
        if band_patches < 1 or band_patches % 2 == 0:
            raise ValueError("band_patches must be a positive odd integer.")
        self.bands, self.patch_size, self.band_patches, self.pool = bands, patch_size, band_patches, pool
        patch_dim = patch_size * patch_size * band_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, bands + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = SpectralFormerTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, bands, mode)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def _groupwise_spectral_embedding(self, x: torch.Tensor) -> torch.Tensor:
        band_indices = torch.arange(x.shape[1], device=x.device)
        half = self.band_patches // 2
        return torch.cat([x.index_select(1, (band_indices + offset) % x.shape[1]).flatten(2) for offset in range(-half, half + 1)], dim=-1)

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"SpectralFormerBaseline expects [N, {self.bands}, H, W], got {tuple(x.shape)}")
        tokens = self.patch_to_embedding(self._groupwise_spectral_embedding(x))
        tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        encoded = self.transformer(self.dropout(tokens + self.pos_embedding[:, : tokens.shape[1]]))
        features = encoded.mean(dim=1) if self.pool == "mean" else encoded[:, 0]
        return {"logits": self.mlp_head(features), "features": features}
