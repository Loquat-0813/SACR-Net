from __future__ import annotations

from .backbones import HybridSN, Light3DCNN
from .axis_mamba import AxisMambaClassifier, ShapeConditionedAxisMamba
from .diffformer_official import DiffFormerBaseline
from .gscvit_official import GSCViTBaseline
from .illum_robust import IlluminationRobustNet
from .mambahsi_official import MambaHSIBaseline
from .spectralformer_official import SpectralFormerBaseline
from .ssrn_official import (
    AdaptiveQuotientSSRN,
    CPCLSSRN,
    LogQuotientDualStreamSSRN,
    MultiScaleAttentionSSRN,
    PrototypeMetricSSRN,
    ResidualSpectralChannelAttentionSSRN,
    RawDualStreamSSRN,
    RawQuotientDualStreamSSRN,
    SSRNBaseline,
    ShapeConditionedSSRN,
)

__all__ = [
    "HybridSN",
    "Light3DCNN",
    "AxisMambaClassifier",
    "ShapeConditionedAxisMamba",
    "DiffFormerBaseline",
    "GSCViTBaseline",
    "IlluminationRobustNet",
    "MambaHSIBaseline",
    "SpectralFormerBaseline",
    "SSRNBaseline",
    "ShapeConditionedSSRN",
    "MultiScaleAttentionSSRN",
    "PrototypeMetricSSRN",
    "CPCLSSRN",
    "ResidualSpectralChannelAttentionSSRN",
    "LogQuotientDualStreamSSRN",
    "RawDualStreamSSRN",
    "RawQuotientDualStreamSSRN",
    "AdaptiveQuotientSSRN",
    "build_model",
]


def build_model(cfg: dict, num_classes: int, input_bands: int | None = None, patch_size: int | None = None):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "light_3dcnn")
    if name == "light_3dcnn":
        return Light3DCNN(
            num_classes=num_classes,
            feature_dim=int(model_cfg.get("feature_dim", 96)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    if name == "hybridsn":
        if input_bands is None:
            raise ValueError("HybridSN requires input_bands from dataset metadata.")
        if patch_size is None:
            patch_size = int(cfg.get("patch_size", 11))
        return HybridSN(
            num_classes=num_classes,
            input_bands=int(input_bands),
            patch_size=int(patch_size),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.4)),
        )
    if name == "illum_robust":
        return IlluminationRobustNet(
            num_classes=num_classes,
            feature_dim=int(model_cfg.get("feature_dim", 96)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            shape_norm=model_cfg.get("shape_norm", "mean"),
            gate_hidden=int(model_cfg.get("gate_hidden", 32)),
            use_shape_branch=bool(model_cfg.get("use_shape_branch", True)),
            use_brightness_gate=bool(model_cfg.get("use_brightness_gate", True)),
            backbone=model_cfg.get("backbone", "light_3dcnn"),
            input_bands=input_bands,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            hybridsn_hidden_dim=int(model_cfg.get("hybridsn_hidden_dim", 256)),
        )
    if name in {"ssrn", "ssrn_std", "q_ssrn", "snv_ssrn", "l2_ssrn"}:
        if input_bands is None:
            raise ValueError("SSRN-family models require input_bands from dataset metadata.")
        return SSRNBaseline(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    if name == "shape_ssrn":
        if input_bands is None:
            raise ValueError("Shape-conditioned SSRN requires input_bands from dataset metadata.")
        return ShapeConditionedSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            shape_channels=int(model_cfg.get("shape_channels", 8)),
            condition_strength=float(model_cfg.get("condition_strength", 0.25)),
        )
    if name == "msaf_ssrn":
        if input_bands is None:
            raise ValueError("MSAF-SSRN requires input_bands from dataset metadata.")
        return MultiScaleAttentionSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            msaf_kernels=tuple(int(v) for v in model_cfg.get("msaf_kernels", [3, 5, 7])),
            msaf_reduction=int(model_cfg.get("msaf_reduction", 8)),
        )
    if name == "rsca_ssrn":
        if input_bands is None:
            raise ValueError("RSCA-SSRN requires input_bands from dataset metadata.")
        return ResidualSpectralChannelAttentionSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            rsca_kernel=int(model_cfg.get("rsca_kernel", 5)),
        )
    if name == "pml_ssrn":
        if input_bands is None:
            raise ValueError("PML-SSRN requires input_bands from dataset metadata.")
        return PrototypeMetricSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            metric_dim=int(model_cfg.get("metric_dim", model_cfg.get("spectral_channels", 24))),
        )
    if name == "cpcl_ssrn":
        if input_bands is None:
            raise ValueError("CPCL-SSRN requires input_bands from dataset metadata.")
        return CPCLSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            metric_dim=int(model_cfg.get("metric_dim", model_cfg.get("spectral_channels", 24))),
        )
    if name == "lq_ssrn":
        if input_bands is None:
            raise ValueError("Log-quotient dual-stream SSRN requires input_bands from dataset metadata.")
        return LogQuotientDualStreamSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    if name == "rq_ssrn":
        if input_bands is None:
            raise ValueError("Raw-quotient dual-stream SSRN requires input_bands from dataset metadata.")
        return RawQuotientDualStreamSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    if name == "aq_ssrn":
        if input_bands is None:
            raise ValueError("Adaptive quotient SSRN requires input_bands from dataset metadata.")
        return AdaptiveQuotientSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            gate_hidden=int(model_cfg.get("gate_hidden", 64)),
            gate_initial=float(model_cfg.get("gate_initial", 0.25)),
            gate_max=float(model_cfg.get("gate_max", 1.0)),
            gate_uses_illum=bool(model_cfg.get("gate_uses_illum", True)),
        )
    if name == "raw_dual_ssrn":
        if input_bands is None:
            raise ValueError("Raw dual-stream SSRN requires input_bands from dataset metadata.")
        return RawDualStreamSSRN(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            spectral_channels=int(model_cfg.get("spectral_channels", 24)),
            compressed_channels=int(model_cfg.get("compressed_channels", 128)),
            dropout=float(model_cfg.get("dropout", 0.5)),
        )
    if name == "spectralformer":
        if input_bands is None:
            raise ValueError("SpectralFormer requires input_bands from dataset metadata.")
        return SpectralFormerBaseline(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            dim=int(model_cfg.get("dim", 64)),
            depth=int(model_cfg.get("depth", 5)),
            heads=int(model_cfg.get("heads", 4)),
            mlp_dim=int(model_cfg.get("mlp_dim", 128)),
            dim_head=int(model_cfg.get("dim_head", 16)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            emb_dropout=float(model_cfg.get("emb_dropout", 0.1)),
            mode=str(model_cfg.get("mode", "CAF")),
            band_patches=int(model_cfg.get("band_patches", 3)),
        )
    if name == "mambahsi":
        if input_bands is None:
            raise ValueError("MambaHSI requires input_bands from dataset metadata.")
        return MambaHSIBaseline(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            token_num=int(model_cfg.get("token_num", 4)),
            group_num=int(model_cfg.get("group_num", 4)),
            ssm_backend=str(model_cfg.get("ssm_backend", "mamba_ssm")),
        )
    if name == "axis_mamba":
        if input_bands is None:
            raise ValueError("AxisMamba requires input_bands from dataset metadata.")
        return AxisMambaClassifier(
            bands=int(input_bands),
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            spectral_tokens=int(model_cfg.get("spectral_tokens", 8)),
            group_num=int(model_cfg.get("group_num", 4)),
            ssm_backend=str(model_cfg.get("ssm_backend", "mamba_ssm")),
        )
    if name == "shape_axis_mamba":
        if input_bands is None:
            raise ValueError("Shape-conditioned AxisMamba requires input_bands from dataset metadata.")
        return ShapeConditionedAxisMamba(
            bands=int(input_bands),
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            spectral_tokens=int(model_cfg.get("spectral_tokens", 8)),
            group_num=int(model_cfg.get("group_num", 4)),
            ssm_backend=str(model_cfg.get("ssm_backend", "mamba_ssm")),
            condition_strength=float(model_cfg.get("condition_strength", 0.25)),
        )
    if name == "gscvit":
        if input_bands is None:
            raise ValueError("GSC-ViT requires input_bands from dataset metadata.")
        return GSCViTBaseline(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            dims=tuple(int(v) for v in model_cfg.get("dims", [256, 128, 64])),
            depth=tuple(int(v) for v in model_cfg.get("depth", [1, 1])),
            heads=tuple(int(v) for v in model_cfg.get("heads", [4, 4])),
            group_spatial_size=tuple(int(v) for v in model_cfg.get("group_spatial_size", [1, 1])),
            num_groups=tuple(int(v) for v in model_cfg.get("num_groups", [16, 16])),
            padding=tuple(int(v) for v in model_cfg.get("padding", [1, 1])),
            dropout=float(model_cfg.get("dropout", 0.1)),
            dim_head=int(model_cfg.get("dim_head", 16)),
        )
    if name == "diffformer":
        if input_bands is None:
            raise ValueError("DiffFormer requires input_bands from dataset metadata.")
        return DiffFormerBaseline(
            bands=int(input_bands),
            num_classes=num_classes,
            patch_size=int(patch_size if patch_size is not None else cfg.get("patch_size", 11)),
            dim=int(model_cfg.get("dim", 64)),
            transformer_layers=int(model_cfg.get("transformer_layers", 2)),
            block_depth=int(model_cfg.get("block_depth", 2)),
            heads=int(model_cfg.get("heads", 4)),
            mlp_dim=int(model_cfg.get("mlp_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            layernorm_eps=float(model_cfg.get("layernorm_eps", 1e-3)),
            hidden=int(model_cfg.get("hidden", 32)),
        )
    raise ValueError(f"Unknown model name: {name}")
