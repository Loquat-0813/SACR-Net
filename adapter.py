"""Frozen 2 x 3 x 2 Raw+SNV independent-dual SSRN ablation."""

import torch
from torch import nn

from hsi_shadow.models.ssrn_official import RawQuotientDualStreamSSRN
from rq_encoders import ChannelRelations


class Model(nn.Module):
    def __init__(self, cfg, classes, bands, patch):
        super().__init__()
        if cfg.get("auxiliary_view") != "snv" or cfg.get("input_view") != "raw":
            raise ValueError("This ablation requires Raw+SNV, not RQ")
        if cfg.get("fusion_initialization") != "zero":
            raise ValueError("The original projection must remain zero initialized")
        self.use_dense = bool(cfg["ablation"]["dense"])
        self.c_kernel = int(cfg["ablation"]["c_kernel"])
        self.use_npaf = bool(cfg["ablation"]["npaf"])
        if self.c_kernel not in (0, 1, 3):
            raise ValueError(self.c_kernel)
        if (self.c_kernel == 0) != (cfg.get("amplitude_variant") == "none"):
            raise ValueError("C data preparation and model setting disagree")

        # Construct the original independent dual stream before any added
        # parameters, preserving every shared initial tensor for this seed.
        self.core = RawQuotientDualStreamSSRN(bands, classes, patch_size=patch)
        channels = self.core.spectral_collapse[3].num_features
        spatial_channels = self.core.spatial_stem[1].num_features
        if self.use_dense:
            self.dense = ChannelRelations(sparse=False, channels=channels, heads=4)
        if self.c_kernel:
            self.c = nn.Conv2d(2, channels, self.c_kernel,
                               padding=self.c_kernel // 2, bias=True)
            nn.init.zeros_(self.c.weight)
            nn.init.zeros_(self.c.bias)
        if self.use_npaf:
            from npaf_author import NPAF
            if not isinstance(self.core.spatial_stem[2], nn.ReLU):
                raise TypeError("Unexpected spatial stem activation")
            # The same NPAF seed across all cells, independent of dense/C
            # constructor draws. Common base tensors were constructed first.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(cfg['seed']) + 300000)
                self.core.spatial_stem[2] = NPAF(spatial_channels)
        self.supervision = "none"
        self.use_ogm = False

    def forward(self, x, illum=None, quotient=None, amplitude=None):
        del illum
        if quotient is None:
            raise ValueError("SNV auxiliary view is missing")
        f_raw = self.core._spectral_features(x)
        f_s = self.core._quotient_features(quotient)
        if self.use_dense:
            f_s = self.dense(f_s)
        if self.c_kernel:
            if amplitude is None or amplitude.shape != (x.shape[0], 2, x.shape[2], x.shape[3]):
                raise ValueError("Aligned [N,2,H,W] spectral-moment map required")
            f_s = f_s + self.c(amplitude)
        fused = f_raw + self.core.quotient_projection(f_s)
        return self.core._classify_compressed(fused)


def build_model(cfg, num_classes, input_bands, patch_size):
    return Model(cfg, num_classes, input_bands, patch_size)


def fit_scale(cfg, ds):
    return None
