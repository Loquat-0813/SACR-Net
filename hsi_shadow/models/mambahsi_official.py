from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


try:
    from mamba_ssm import Mamba as _MambaSSM
except Exception:  # Optional: cloud-only paper baseline dependency.
    _MambaSSM = None


class CompatMamba(nn.Module):
    """Shape-compatible local smoke backend; never use it for paper numbers."""

    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2, **_: object) -> None:
        super().__init__()
        inner = d_model * expand
        self.in_proj = nn.Linear(d_model, inner * 2)
        self.conv = nn.Conv1d(inner, inner, kernel_size=d_conv, padding=d_conv // 2, groups=inner)
        self.out_proj = nn.Linear(inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = self.conv(mixed.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return self.out_proj(F.silu(mixed) * torch.sigmoid(gate))


def make_mamba(d_model: int, backend: str) -> nn.Module:
    if backend == "mamba_ssm":
        if _MambaSSM is None:
            raise ImportError("MambaHSI requires mamba-ssm. Install mamba-ssm==1.2.0 and triton==2.2.0 on the cloud.")
        return _MambaSSM(d_model=d_model, d_state=16, d_conv=4, expand=2)
    if backend == "compat":
        return CompatMamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
    raise ValueError(f"Unsupported Mamba backend: {backend}")


class SpeMamba(nn.Module):
    def __init__(self, channels: int, token_num: int, group_num: int, backend: str) -> None:
        super().__init__()
        self.token_num = token_num
        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = token_num * self.group_channel_num
        self.mamba = make_mamba(self.group_channel_num, backend)
        self.proj = nn.Sequential(nn.GroupNorm(group_num, self.channel_num), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if x.shape[1] < self.channel_num:
            x = torch.cat([x, x.new_zeros((b, self.channel_num - x.shape[1], h, w))], dim=1)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, self.token_num, self.group_channel_num)
        recon = self.mamba(x_flat).view(b, h, w, self.channel_num).permute(0, 3, 1, 2).contiguous()
        return x[:, : self.channel_num] + self.proj(recon)


class SpaMamba(nn.Module):
    def __init__(self, channels: int, group_num: int, backend: str) -> None:
        super().__init__()
        self.mamba = make_mamba(channels, backend)
        self.proj = nn.Sequential(nn.GroupNorm(group_num, channels), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flattened = x.permute(0, 2, 3, 1).contiguous().view(1, b * h * w, c)
        recon = self.mamba(flattened).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x + self.proj(recon)


class BothMamba(nn.Module):
    def __init__(self, channels: int, token_num: int, group_num: int, backend: str) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.spa_mamba = SpaMamba(channels, group_num, backend)
        self.spe_mamba = SpeMamba(channels, token_num, group_num, backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.weights.softmax(dim=0)
        return x + self.spa_mamba(x) * weights[0] + self.spe_mamba(x) * weights[1]


class MambaHSIBaseline(nn.Module):
    """Shared-protocol adapter of li-yapeng/MambaHSI (TGRS 2024) backbone."""

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        hidden_dim: int = 64,
        token_num: int = 4,
        group_num: int = 4,
        ssm_backend: str = "mamba_ssm",
    ) -> None:
        super().__init__()
        del patch_size
        if hidden_dim % group_num:
            raise ValueError("hidden_dim must be divisible by group_num.")
        self.bands = bands
        self.ssm_backend = ssm_backend
        self.patch_embedding = nn.Sequential(nn.Conv2d(bands, hidden_dim, 1), nn.GroupNorm(group_num, hidden_dim), nn.SiLU())
        block = lambda: BothMamba(hidden_dim, token_num, group_num, ssm_backend)
        self.mamba = nn.Sequential(block(), nn.AvgPool2d(2), block(), nn.AvgPool2d(2), block())
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 1), nn.GroupNorm(group_num, 128), nn.SiLU(), nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"MambaHSIBaseline expects [N, {self.bands}, H, W], got {tuple(x.shape)}")
        features_map = self.mamba(self.patch_embedding(x))
        logits_map = self.cls_head(features_map)
        return {"logits": logits_map.mean(dim=(-1, -2)), "features": features_map.mean(dim=(-1, -2)), "logits_map": logits_map}
