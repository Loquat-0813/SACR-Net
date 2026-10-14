from __future__ import annotations

import math

import torch
from torch import nn

from .mambahsi_official import make_mamba


def spectral_shape_transform(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.mean(dim=1, keepdim=True).clamp_min(eps)


class AxisStateBlock(nn.Module):
    """Patch-local spatial and spectral state propagation without cross-sample scans."""

    def __init__(self, channels: int, spectral_tokens: int, group_num: int, backend: str) -> None:
        super().__init__()
        if channels % group_num:
            raise ValueError("channels must be divisible by group_num")
        self.channels = channels
        self.spectral_tokens = spectral_tokens
        self.spectral_width = math.ceil(channels / spectral_tokens)
        self.padded_channels = self.spectral_tokens * self.spectral_width

        self.row_mamba = make_mamba(channels, backend)
        self.col_mamba = make_mamba(channels, backend)
        self.spectral_forward = make_mamba(self.spectral_width, backend)
        self.spectral_backward = make_mamba(self.spectral_width, backend)
        self.spatial_norm = nn.GroupNorm(group_num, channels)
        self.spectral_norm = nn.GroupNorm(group_num, channels)
        self.mix_logits = nn.Parameter(torch.zeros(2))
        self.out = nn.Sequential(
            nn.GroupNorm(group_num, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def _spatial_state(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        row_state = self.row_mamba(rows).reshape(batch, height, width, channels)

        cols = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        col_state = self.col_mamba(cols).reshape(batch, width, height, channels).permute(0, 2, 1, 3)
        spatial = 0.5 * (row_state + col_state)
        return self.spatial_norm(spatial.permute(0, 3, 1, 2).contiguous())

    def _spectral_state(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if channels < self.padded_channels:
            padding = x.new_zeros(batch, self.padded_channels - channels, height, width)
            x = torch.cat([x, padding], dim=1)
        tokens = x.permute(0, 2, 3, 1).reshape(batch * height * width, self.spectral_tokens, self.spectral_width)
        forward = self.spectral_forward(tokens)
        backward = torch.flip(self.spectral_backward(torch.flip(tokens, dims=[1])), dims=[1])
        spectral = 0.5 * (forward + backward)
        spectral = spectral.reshape(batch, height, width, self.padded_channels)[..., :channels]
        return self.spectral_norm(spectral.permute(0, 3, 1, 2).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self._spatial_state(x)
        spectral = self._spectral_state(x)
        weights = self.mix_logits.softmax(dim=0)
        return x + self.out(weights[0] * spatial + weights[1] * spectral)


class AxisMambaClassifier(nn.Module):
    """Clean patch classifier inspired by spatial-spectral state propagation."""

    def __init__(
        self,
        bands: int,
        num_classes: int,
        hidden_dim: int = 64,
        spectral_tokens: int = 8,
        group_num: int = 4,
        ssm_backend: str = "mamba_ssm",
    ) -> None:
        super().__init__()
        if hidden_dim % group_num:
            raise ValueError("hidden_dim must be divisible by group_num")
        self.bands = bands
        self.patch_embed = nn.Sequential(
            nn.Conv2d(bands, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [AxisStateBlock(hidden_dim, spectral_tokens, group_num, ssm_backend) for _ in range(3)]
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(x)
        for index, block in enumerate(self.blocks):
            features = block(features)
            if index < len(self.blocks) - 1:
                features = self.pool(features)
        return features

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"AxisMambaClassifier expects [N, {self.bands}, H, W], got {tuple(x.shape)}")
        features_map = self.encode(x)
        features = features_map.mean(dim=(-1, -2))
        return {"logits": self.classifier(features), "features": features}


class ShapeConditionedAxisMamba(AxisMambaClassifier):
    """AxisMamba with a lightweight spectral-shape conditioner, not a second state-space backbone."""

    def __init__(
        self,
        bands: int,
        num_classes: int,
        hidden_dim: int = 64,
        spectral_tokens: int = 8,
        group_num: int = 4,
        ssm_backend: str = "mamba_ssm",
        condition_strength: float = 0.25,
    ) -> None:
        super().__init__(bands, num_classes, hidden_dim, spectral_tokens, group_num, ssm_backend)
        self.condition_strength = float(condition_strength)
        self.shape_embed = nn.Sequential(
            nn.Conv2d(bands, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        self.shape_refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                    nn.GroupNorm(group_num, hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
                )
                for _ in self.blocks
            ]
        )
        self.conditioners = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=True) for _ in self.blocks]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(x)
        shape_features = self.shape_embed(spectral_shape_transform(x))
        for index, block in enumerate(self.blocks):
            shape_features = shape_features + self.shape_refine[index](shape_features)
            gamma, beta = self.conditioners[index](shape_features).chunk(2, dim=1)
            features = block(features)
            features = features * (1.0 + self.condition_strength * torch.tanh(gamma))
            features = features + self.condition_strength * torch.tanh(beta)
            if index < len(self.blocks) - 1:
                features = self.pool(features)
                shape_features = self.pool(shape_features)
        return features
