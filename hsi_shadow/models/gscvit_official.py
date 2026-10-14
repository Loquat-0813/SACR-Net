from __future__ import annotations

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.weight + self.bias


class PreNorm(nn.Module):
    def __init__(self, channels: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm(channels)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class SpectralCalibration(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupwiseSeparableConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, padding: int = 1, groups: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1, groups=groups),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, groups=groups, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupwiseSeparableSelfAttention(nn.Module):
    """PyTorch-only port of GSC-ViT's GSSA block.

    The official source uses einops for tensor rearrangement.  This adapter keeps
    the same local group-token attention and inter-group aggregation idea while
    avoiding non-core runtime dependencies in the cloud package.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        dim_head: int = 16,
        dropout: float = 0.1,
        group_spatial_size: int = 1,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = dim_head**-0.5
        self.group_spatial_size = int(group_spatial_size)
        self.group_token = nn.Parameter(torch.randn(channels))
        self.to_qkv = nn.Conv1d(channels, inner_dim * 3, kernel_size=1, bias=False)
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.group_qk = nn.Sequential(nn.LayerNorm(dim_head), nn.GELU(), nn.Linear(dim_head, dim_head * 2))
        self.group_attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))
        self.to_out = nn.Sequential(nn.Conv2d(inner_dim, channels, kernel_size=1), nn.Dropout(dropout))

    def _groups_to_map(self, grouped: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
        g = self.group_spatial_size
        gh, gw = height // g, width // g
        grouped = grouped.reshape(batch, gh, gw, self.heads, g, g, self.dim_head)
        grouped = grouped.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        return grouped.reshape(batch, self.heads * self.dim_head, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        g = self.group_spatial_size
        if height % g != 0 or width % g != 0:
            raise ValueError(f"GSC-ViT group size {g} does not divide patch shape {(height, width)}.")
        gh, gw = height // g, width // g
        num_groups = gh * gw

        grouped = x.reshape(batch, channels, gh, g, gw, g)
        grouped = grouped.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(batch * num_groups, channels, g * g)
        token = self.group_token.view(1, channels, 1).expand(grouped.shape[0], -1, -1)
        grouped_with_token = torch.cat([token, grouped], dim=-1)

        q, k, v = self.to_qkv(grouped_with_token).chunk(3, dim=1)
        q = q.reshape(grouped.shape[0], self.heads, self.dim_head, -1).transpose(2, 3)
        k = k.reshape(grouped.shape[0], self.heads, self.dim_head, -1).transpose(2, 3)
        v = v.reshape(grouped.shape[0], self.heads, self.dim_head, -1).transpose(2, 3)

        attn = self.attend((q * self.scale) @ k.transpose(-2, -1))
        out = attn @ v
        group_tokens, grouped_pixels = out[:, :, 0], out[:, :, 1:]

        if num_groups == 1:
            return self.to_out(self._groups_to_map(grouped_pixels, batch, height, width))

        group_tokens = group_tokens.reshape(batch, gh, gw, self.heads, self.dim_head)
        group_tokens = group_tokens.permute(0, 3, 1, 2, 4).contiguous().reshape(batch, self.heads, num_groups, self.dim_head)
        grouped_pixels = grouped_pixels.reshape(batch, gh, gw, self.heads, g * g, self.dim_head)
        grouped_pixels = grouped_pixels.permute(0, 3, 1, 2, 4, 5).contiguous().reshape(
            batch, self.heads, num_groups, g * g, self.dim_head
        )

        w_q, w_k = self.group_qk(group_tokens).chunk(2, dim=-1)
        group_attn = self.group_attend((w_q * self.scale) @ w_k.transpose(-2, -1))
        aggregated = torch.einsum("bhij,bhjnd->bhind", group_attn, grouped_pixels)
        aggregated = aggregated.reshape(batch, self.heads, gh, gw, g, g, self.dim_head)
        aggregated = aggregated.permute(0, 2, 3, 1, 4, 5, 6).contiguous().reshape(
            batch * num_groups, self.heads, g * g, self.dim_head
        )
        return self.to_out(self._groups_to_map(aggregated, batch, height, width))


class GSCTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        heads: int = 4,
        dim_head: int = 16,
        dropout: float = 0.1,
        group_spatial_size: int = 1,
        norm_output: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PreNorm(
                    channels,
                    GroupwiseSeparableSelfAttention(
                        channels=channels,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        group_spatial_size=group_spatial_size,
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.norm = ChannelLayerNorm(channels) if norm_output else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GSCViTBaseline(nn.Module):
    """Shared-protocol adapter for GSC-ViT.

    Source: flyzzie/TGRS-GSC-VIT, ``models/gscvit.py``.  The original training
    code uses random-ratio splits and a default 8x8 patch.  This adapter keeps
    the spectral calibration, groupwise separable convolution, GSSA blocks, and
    classifier while using the common HanChuan official 25/50-label protocol.
    The group size defaults to 1 so that the model can run on the paper's fixed
    11x11 patches without changing the dataset protocol.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        dims: tuple[int, ...] = (256, 128, 64),
        depth: tuple[int, ...] = (1, 1),
        heads: tuple[int, ...] = (4, 4),
        group_spatial_size: tuple[int, ...] = (1, 1),
        num_groups: tuple[int, ...] = (16, 16),
        padding: tuple[int, ...] = (1, 1),
        dropout: float = 0.1,
        dim_head: int = 16,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("GSCViTBaseline requires at least two stage dimensions.")
        stage_count = len(dims) - 1
        if not all(len(v) == stage_count for v in (depth, heads, group_spatial_size, num_groups, padding)):
            raise ValueError("GSCViT stage hyperparameters must match len(dims)-1.")
        self.bands = int(bands)
        self.patch_size = int(patch_size)

        self.calibration = nn.Sequential(
            SpectralCalibration(bands, dims[0]),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        for idx, (channels_in, channels_out) in enumerate(zip(dims[:-1], dims[1:])):
            self.stages.append(
                nn.ModuleDict(
                    {
                        "gsc": GroupwiseSeparableConv(channels_in, channels_out, padding=padding[idx], groups=num_groups[idx]),
                        "transformer": GSCTransformer(
                            channels=channels_out,
                            depth=depth[idx],
                            heads=heads[idx],
                            dim_head=dim_head,
                            dropout=dropout,
                            group_spatial_size=group_spatial_size[idx],
                            norm_output=idx != stage_count - 1,
                        ),
                        "pointwise": nn.Conv2d(channels_out, channels_out, kernel_size=1),
                        "bn": nn.BatchNorm2d(channels_out),
                        "relu": nn.ReLU(inplace=True),
                    }
                )
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"GSCViTBaseline expects [N, {self.bands}, H, W], got {tuple(x.shape)}")
        x = self.calibration(x)
        for stage in self.stages:
            x = stage["gsc"](x)
            residual = x
            x = stage["transformer"](x)
            x = stage["pointwise"](x) + residual
            x = stage["relu"](stage["bn"](x))
        features = self.pool(x).flatten(1)
        features = self.norm(features)
        return {"logits": self.head(features), "features": features}
