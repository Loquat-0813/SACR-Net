from __future__ import annotations

import torch
from torch import nn

from .backbones import Conv3DFeatureExtractor, HybridSNFeatureExtractor


def spectral_shape_transform(x: torch.Tensor, mode: str = "mean", eps: float = 1e-6) -> torch.Tensor:
    if mode == "mean":
        denom = x.mean(dim=1, keepdim=True).clamp_min(eps)
        return x / denom
    if mode == "l2":
        denom = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True).clamp_min(eps)
        return x / denom
    if mode == "derivative":
        diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        first = diff[:, :1, :, :]
        return torch.cat([first, diff], dim=1)
    raise ValueError(f"Unknown shape normalization mode: {mode}")


class BrightnessGate(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, illum: torch.Tensor) -> torch.Tensor:
        if illum is None:
            raise ValueError("IlluminationRobustNet requires illum features.")
        return self.net(illum)


class IlluminationRobustNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 96,
        dropout: float = 0.2,
        shape_norm: str = "mean",
        gate_hidden: int = 32,
        use_shape_branch: bool = True,
        use_brightness_gate: bool = True,
        backbone: str = "light_3dcnn",
        input_bands: int | None = None,
        patch_size: int = 11,
        hybridsn_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.shape_norm = shape_norm
        self.use_shape_branch = use_shape_branch
        self.use_brightness_gate = use_brightness_gate

        def make_branch() -> nn.Module:
            if backbone == "light_3dcnn":
                return Conv3DFeatureExtractor(feature_dim=feature_dim, dropout=dropout)
            if backbone == "hybridsn":
                if input_bands is None:
                    raise ValueError("HybridSN backbone requires input_bands.")
                return HybridSNFeatureExtractor(
                    input_bands=int(input_bands),
                    patch_size=int(patch_size),
                    feature_dim=feature_dim,
                    hidden_dim=hybridsn_hidden_dim,
                    dropout=dropout,
                )
            raise ValueError(f"Unknown illum_robust backbone: {backbone}")

        self.raw_branch = make_branch()
        self.shape_branch = (
            make_branch()
            if use_shape_branch
            else None
        )
        self.gate = BrightnessGate(hidden=gate_hidden) if use_brightness_gate and use_shape_branch else None
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None):
        raw_feat = self.raw_branch(x)
        outputs = {"raw_features": raw_feat}
        if not self.use_shape_branch:
            logits = self.classifier(raw_feat)
            outputs.update({"logits": logits, "features": raw_feat})
            return outputs

        shape_x = spectral_shape_transform(x, mode=self.shape_norm)
        shape_feat = self.shape_branch(shape_x)
        outputs["shape_features"] = shape_feat
        if self.gate is None:
            feat = 0.5 * raw_feat + 0.5 * shape_feat
        else:
            gate = self.gate(illum)
            feat = gate * raw_feat + (1.0 - gate) * shape_feat
            outputs["gate"] = gate.squeeze(1)
        logits = self.classifier(feat)
        outputs.update({"logits": logits, "features": feat})
        return outputs
