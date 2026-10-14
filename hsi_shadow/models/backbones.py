from __future__ import annotations

import torch
from torch import nn


class Conv3DFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input BxCxHxW, got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        return self.proj(self.net(x))


class Light3DCNN(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int = 96, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = Conv3DFeatureExtractor(feature_dim=feature_dim, dropout=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None):
        feat = self.features(x)
        return {"logits": self.classifier(feat), "features": feat}


class HybridSNFeatureExtractor(nn.Module):
    """Feature extractor variant of HybridSN for spectral-shape branch models."""

    def __init__(
        self,
        input_bands: int,
        patch_size: int,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if input_bands <= 12:
            raise ValueError(f"HybridSN requires input_bands > 12, got {input_bands}.")
        if patch_size <= 8:
            raise ValueError(f"HybridSN requires patch_size > 8, got {patch_size}.")

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        conv2d_in_channels = 32 * (input_bands - 12)
        conv2d_out_size = patch_size - 8
        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        flattened_dim = 64 * conv2d_out_size * conv2d_out_size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input BxCxHxW, got {tuple(x.shape)}")
        feat3d = self.conv3d(x.unsqueeze(1))
        b, channels, depth, height, width = feat3d.shape
        feat2d = feat3d.reshape(b, channels * depth, height, width)
        spatial = self.conv2d(feat2d)
        return self.head(spatial)


class HybridSN(nn.Module):
    """Hybrid spectral-spatial CNN baseline for hyperspectral image classification.

    The architecture follows the common HybridSN pattern: several valid 3D
    convolutions over spectral-spatial patches, a spectral-to-channel reshape,
    then a 2D spatial convolution and fully connected classifier.
    """

    def __init__(
        self,
        num_classes: int,
        input_bands: int,
        patch_size: int,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if input_bands <= 12:
            raise ValueError(f"HybridSN requires input_bands > 12, got {input_bands}.")
        if patch_size <= 8:
            raise ValueError(f"HybridSN requires patch_size > 8, got {patch_size}.")

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        conv2d_in_channels = 32 * (input_bands - 12)
        conv2d_out_size = patch_size - 8
        self.conv2d = nn.Sequential(
            nn.Conv2d(conv2d_in_channels, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        flattened_dim = 64 * conv2d_out_size * conv2d_out_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None):
        if x.dim() != 4:
            raise ValueError(f"Expected input BxCxHxW, got {tuple(x.shape)}")
        feat3d = self.conv3d(x.unsqueeze(1))
        b, channels, depth, height, width = feat3d.shape
        feat2d = feat3d.reshape(b, channels * depth, height, width)
        spatial = self.conv2d(feat2d)
        logits = self.classifier(spatial)
        return {"logits": logits, "features": spatial.flatten(1)}
