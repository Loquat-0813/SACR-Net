from __future__ import annotations

import math

import torch
from torch import nn


class _Residual3D(nn.Module):
    """Two 1×1×7 spectral convolutions used by the legacy SSRN backbone."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class _Residual2D(nn.Module):
    """Two 3×3 spatial convolutions used after spectral compression."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SSRNBaseline(nn.Module):
    """PyTorch adapter of the official SSRN-8 spectral–spatial residual backbone.

    Source architecture: Zilong Zhong's MIT-licensed ``SSRN`` repository,
    ``Utils/ssrn_SS_IN.py``.  The original TensorFlow/Keras code only accepts
    Indian Pines/Pavia University scripts.  This adapter preserves its
    spectral-residual -> spectral-compression -> spatial-residual sequence and
    exposes the shared patch-classification interface used for the official
    WHU-Hi-HanChuan protocol.  It is therefore an adapted reproduction, not a
    direct execution of the legacy training script.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if bands < 7:
            raise ValueError(f"SSRN requires at least 7 bands, got {bands}.")
        if patch_size < 3:
            raise ValueError(f"SSRN requires patch_size >= 3, got {patch_size}.")
        self.bands = int(bands)
        self.num_classes = int(num_classes)
        self.patch_size = int(patch_size)

        # Official SSRN-8: 1x1x7 spectral stem (stride 2), one residual
        # spectral block, then a 1x1xD spectral collapse to 128 channels.
        self.spectral_stem = nn.Sequential(
            nn.Conv3d(1, spectral_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
        )
        self.spectral_residual = _Residual3D(spectral_channels)
        spectral_depth = (bands - 7) // 2 + 1
        self.spectral_collapse = nn.Sequential(
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                spectral_channels,
                compressed_channels,
                kernel_size=(spectral_depth, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(compressed_channels),
            nn.ReLU(inplace=True),
        )

        # The Keras implementation reshapes [H, W, 128, 1] and applies a
        # 3x3x128 convolution.  This is equivalent to a 2-D 3x3 convolution
        # with 128 input channels after spectral collapse.
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(compressed_channels, spectral_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(spectral_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_residual = _Residual2D(spectral_channels)
        self.spatial_tail = nn.Sequential(nn.BatchNorm2d(spectral_channels), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(spectral_channels, num_classes),
        )

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SSRNBaseline expects [N, bands, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.bands:
            raise ValueError(f"Expected {self.bands} spectral bands, got {x.shape[1]}")
        return self._compress_spectral_stem(self.spectral_stem(x.unsqueeze(1)))

    def _compress_spectral_stem(self, spectral_stem: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral_residual(spectral_stem)
        return self.spectral_collapse(spectral).squeeze(2)

    def _classify_compressed(self, compressed: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial = self.spatial_tail(self.spatial_residual(self.spatial_stem(compressed)))
        features = spatial.mean(dim=(-1, -2))
        logits = self.classifier[1:](self.classifier[0](spatial))
        return {"logits": logits, "features": features}

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        return self._classify_compressed(self._spectral_features(x))


class _SpectralShapeFiLM(nn.Module):
    """Local spectral-shape conditioner with zero-initialized feature modulation."""

    def __init__(self, bands: int, channels: int, shape_channels: int) -> None:
        super().__init__()
        self.bands = int(bands)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, shape_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(shape_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(shape_channels, channels * 2, kernel_size=1),
        )
        projection = self.encoder[-1]
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"Expected [N, {self.bands}, H, W], got {tuple(x.shape)}")
        shape = x / x.mean(dim=1, keepdim=True).clamp_min(1e-6)
        gamma, beta = self.encoder(shape.unsqueeze(1)).chunk(2, dim=1)
        return gamma, beta


class ShapeConditionedSSRN(SSRNBaseline):
    """SSRN with a lightweight illumination-invariant spectral-shape adapter.

    The SSRN spectral and spatial backbones are unchanged. The adapter uses
    local mean-normalized spectra to FiLM-modulate the SSRN spectral stem. Its
    final projection is initialized to zero, making initialization exactly
    equivalent to the matched SSRN baseline.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
        shape_channels: int = 8,
        condition_strength: float = 0.25,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        if shape_channels < 1:
            raise ValueError(f"shape_channels must be positive, got {shape_channels}.")
        if not 0.0 < condition_strength <= 1.0:
            raise ValueError(f"condition_strength must be in (0, 1], got {condition_strength}.")
        self.condition_strength = float(condition_strength)
        self.shape_film = _SpectralShapeFiLM(bands, spectral_channels, shape_channels)

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"Expected [N, {self.bands}, H, W], got {tuple(x.shape)}")
        spectral_stem = self.spectral_stem(x.unsqueeze(1))
        gamma, beta = self.shape_film(x)
        conditioned = spectral_stem * (1.0 + self.condition_strength * torch.tanh(gamma))
        conditioned = conditioned + self.condition_strength * torch.tanh(beta)
        compressed = self._compress_spectral_stem(conditioned)
        return self._classify_compressed(compressed)


class _MultiScaleSpectralSpatialAttention(nn.Module):
    """Residual multi-scale spatial filtering plus compressed-spectral attention.

    The block operates after SSRN spectral compression, where channels already
    encode spectral responses and the feature map still preserves the local
    patch layout.  Its final fusion projection is zero-initialized, so a model
    containing this block is exactly equivalent to SSRN at initialization.
    """

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, ...] = (3, 5, 7),
        reduction: int = 8,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction < 1:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        if not kernels:
            raise ValueError("MSAF requires at least one spatial kernel.")
        for kernel in kernels:
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(f"MSAF kernels must be positive odd integers, got {kernels}.")
        self.kernels = tuple(int(kernel) for kernel in kernels)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in self.kernels
            ]
        )
        self.fusion = nn.Conv2d(channels * len(self.kernels), channels, kernel_size=1)
        hidden = max(1, channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        residual = self.fusion(multi_scale)
        attention = self.channel_attention(x)
        return residual * attention


class MultiScaleAttentionSSRN(SSRNBaseline):
    """SSRN with a residual multi-scale spectral-spatial attention fusion block.

    MSAF-SSRN deliberately preserves the raw SSRN radiometric path instead of
    enforcing quotient normalization.  The added branch enhances local
    crop-structure evidence through 3/5/7 depthwise spatial filters and
    compressed-spectral channel attention.  Because the residual fusion layer is
    zero-initialized, the model starts as the matched SSRN baseline and only
    deviates when the extra module learns useful evidence.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
        msaf_kernels: tuple[int, ...] = (3, 5, 7),
        msaf_reduction: int = 8,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        self.msaf = _MultiScaleSpectralSpatialAttention(
            channels=compressed_channels,
            kernels=tuple(int(kernel) for kernel in msaf_kernels),
            reduction=int(msaf_reduction),
        )

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        compressed = self._spectral_features(x)
        enhanced = compressed + self.msaf(compressed)
        return self._classify_compressed(enhanced)


class _ResidualSpectralChannelAttention(nn.Module):
    """Zero-initialized residual channel reweighting after SSRN compression.

    This is intentionally lighter than MSAF: no extra spatial filtering is
    applied.  A 1-D convolution over compressed channels learns local spectral
    channel interactions from a global patch descriptor.  The convolution is
    zero-initialized, so the residual branch is exactly zero at initialization
    while still receiving a nonzero gradient.
    """

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.channel_conv = nn.Conv1d(
            1,
            1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
        )
        nn.init.zeros_(self.channel_conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected [N, {self.channels}, H, W], got {tuple(x.shape)}")
        pooled = x.mean(dim=(-1, -2)).unsqueeze(1)
        channel_logits = self.channel_conv(pooled).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * torch.tanh(channel_logits)


class ResidualSpectralChannelAttentionSSRN(SSRNBaseline):
    """SSRN with a conservative residual spectral-channel attention module.

    RSCA-SSRN tests whether channel-only spectral recalibration is safer than
    the failed MSAF spatial multi-scale branch for the three-dataset support
    target.  It preserves the raw SSRN path, applies no new spatial filters,
    and starts exactly as SSRN because the residual attention branch is
    zero-initialized.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
        rsca_kernel: int = 5,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        self.rsca = _ResidualSpectralChannelAttention(channels=compressed_channels, kernel_size=int(rsca_kernel))

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        compressed = self._spectral_features(x)
        enhanced = compressed + self.rsca(compressed)
        return self._classify_compressed(enhanced)


class PrototypeMetricSSRN(SSRNBaseline):
    """SSRN trained with an auxiliary prototype-metric embedding.

    The classifier path is inherited unchanged from SSRN.  The metric
    projection is initialized to identity when the embedding dimension matches
    the SSRN spatial feature dimension, so the initial classifier logits are
    exactly identical to SSRN.  The projection is used only by the auxiliary
    prototype loss during training; inference still uses the ordinary SSRN
    classifier logits.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
        metric_dim: int = 24,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        if metric_dim < 1:
            raise ValueError(f"metric_dim must be positive, got {metric_dim}.")
        self.metric_projection = nn.Linear(spectral_channels, int(metric_dim), bias=False)
        if int(metric_dim) == int(spectral_channels):
            nn.init.eye_(self.metric_projection.weight)
        else:
            nn.init.xavier_uniform_(self.metric_projection.weight)

    def _classify_compressed(self, compressed: torch.Tensor) -> dict[str, torch.Tensor]:
        out = super()._classify_compressed(compressed)
        out["metric_features"] = self.metric_projection(out["features"])
        return out


class CPCLSSRN(PrototypeMetricSSRN):
    """SSRN with prototype calibration plus supervised contrastive training.

    CPCL keeps the classifier logits identical to SSRN at initialization and
    reuses the identity-initialized metric projection from the prototype-metric
    branch. The difference is entirely in the training objective: the companion
    screen adds both class-prototype calibration and supervised contrastive
    regularization on the shared metric features.
    """

    pass


class LogQuotientDualStreamSSRN(SSRNBaseline):
    """Raw SSRN with a separate, raw-radiance log-quotient spectral stream."""

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        spectral_depth = (bands - 7) // 2 + 1
        self.quotient_spectral_stem = nn.Sequential(
            nn.Conv3d(1, spectral_channels, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False),
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
        )
        self.quotient_spectral_residual = _Residual3D(spectral_channels)
        self.quotient_spectral_collapse = nn.Sequential(
            nn.BatchNorm3d(spectral_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                spectral_channels,
                compressed_channels,
                kernel_size=(spectral_depth, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(compressed_channels),
            nn.ReLU(inplace=True),
        )
        self.quotient_projection = nn.Conv2d(compressed_channels, compressed_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.quotient_projection.weight)

    def _quotient_features(self, quotient: torch.Tensor) -> torch.Tensor:
        if quotient.ndim != 4 or quotient.shape[1] != self.bands:
            raise ValueError(f"Expected quotient [N, {self.bands}, H, W], got {tuple(quotient.shape)}")
        stem = self.quotient_spectral_stem(quotient.unsqueeze(1))
        spectral = self.quotient_spectral_residual(stem)
        return self.quotient_spectral_collapse(spectral).squeeze(2)

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor | None = None,
        quotient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del illum
        if quotient is None:
            raise ValueError("LogQuotientDualStreamSSRN requires a raw-radiance log-quotient view.")
        raw_compressed = self._spectral_features(x)
        quotient_compressed = self._quotient_features(quotient)
        fused = raw_compressed + self.quotient_projection(quotient_compressed)
        return self._classify_compressed(fused)


class RawQuotientDualStreamSSRN(LogQuotientDualStreamSSRN):
    """Dual-stream SSRN driven by a raw spectral quotient that permits zeros."""

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor | None = None,
        quotient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if quotient is None:
            raise ValueError("RawQuotientDualStreamSSRN requires a raw-radiance quotient view.")
        return super().forward(x, illum=illum, quotient=quotient)


class _AdaptiveQuotientGate(nn.Module):
    """Sample-wise gate that lets the model suppress a harmful quotient stream.

    The final affine layer is initialized to a constant gate value. Combined
    with the zero-initialized quotient projection inherited from RQ-SSRN, the
    whole model is exactly equivalent to SSRN at initialization while retaining
    a learnable path for quotient evidence.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        initial_gate: float = 0.25,
        gate_max: float = 1.0,
        use_illum: bool = True,
    ) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"gate hidden dimension must be positive, got {hidden}.")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError(f"gate_max must be in (0, 1], got {gate_max}.")
        if not 0.0 < initial_gate < gate_max:
            raise ValueError(f"initial_gate must be in (0, gate_max), got {initial_gate}/{gate_max}.")
        self.gate_max = float(gate_max)
        self.use_illum = bool(use_illum)
        input_dim = int(channels) * 2 + (1 if self.use_illum else 0)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        initial_probability = float(initial_gate) / self.gate_max
        final.bias.data.fill_(math.log(initial_probability / (1.0 - initial_probability)))

    def forward(
        self,
        raw_compressed: torch.Tensor,
        quotient_compressed: torch.Tensor,
        illum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw_pool = raw_compressed.mean(dim=(-1, -2))
        quotient_pool = quotient_compressed.mean(dim=(-1, -2))
        features = [raw_pool, quotient_pool]
        if self.use_illum:
            if illum is None:
                illum_z = raw_pool.new_zeros(raw_pool.shape[0], 1)
            else:
                illum_z = illum[:, :1].to(dtype=raw_pool.dtype, device=raw_pool.device)
            features.append(illum_z)
        gate = self.gate_max * torch.sigmoid(self.mlp(torch.cat(features, dim=1)))
        return gate.view(-1, 1, 1, 1)


class AdaptiveQuotientSSRN(LogQuotientDualStreamSSRN):
    """RQ-SSRN with an adaptive residual gate for cross-scene robustness.

    AQ-SSRN keeps the SSRN raw-radiance stream as the identity path and treats
    the raw quotient stream as a gated residual. The gate is predicted from
    paired compressed raw/quotient features plus the standardized brightness
    statistic. This allows the model to reduce quotient influence when
    amplitude is discriminative for a dataset.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        spectral_channels: int = 24,
        compressed_channels: int = 128,
        dropout: float = 0.5,
        gate_hidden: int = 64,
        gate_initial: float = 0.25,
        gate_max: float = 1.0,
        gate_uses_illum: bool = True,
    ) -> None:
        super().__init__(
            bands=bands,
            num_classes=num_classes,
            patch_size=patch_size,
            spectral_channels=spectral_channels,
            compressed_channels=compressed_channels,
            dropout=dropout,
        )
        self.adaptive_quotient_gate = _AdaptiveQuotientGate(
            channels=compressed_channels,
            hidden=gate_hidden,
            initial_gate=gate_initial,
            gate_max=gate_max,
            use_illum=gate_uses_illum,
        )

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor | None = None,
        quotient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if quotient is None:
            raise ValueError("AdaptiveQuotientSSRN requires a raw-radiance quotient view.")
        raw_compressed = self._spectral_features(x)
        quotient_compressed = self._quotient_features(quotient)
        gate = self.adaptive_quotient_gate(raw_compressed, quotient_compressed, illum=illum)
        fused = raw_compressed + gate * self.quotient_projection(quotient_compressed)
        out = self._classify_compressed(fused)
        out["gate"] = gate.flatten(1)
        return out


class RawDualStreamSSRN(LogQuotientDualStreamSSRN):
    """Parameter-matched RQ control with a standardized raw auxiliary stream.

    The module layout intentionally matches RawQuotientDualStreamSSRN. The
    caller supplies a train-only standardized raw-radiance auxiliary view instead
    of a quotient view, isolating the quotient transformation from added capacity.
    """

    def forward(
        self,
        x: torch.Tensor,
        illum: torch.Tensor | None = None,
        quotient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if quotient is None:
            raise ValueError("RawDualStreamSSRN requires a standardized raw auxiliary view.")
        return super().forward(x, illum=illum, quotient=quotient)
