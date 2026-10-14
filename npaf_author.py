"""NPAF adapted verbatim in computation from AchunLee/CLOLN_TGRS.

Source: CLOLN_demo.ipynb, class NPAF, repository master.
The author's notebook defines ReLU -> depthwise conv2d -> BatchNorm2d.
The existing timm 0.4.12 environment exposes the author's initializer via
timm.models.layers instead of timm.layers; computation is unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import weight_init


class NPAF(nn.ReLU):
    def __init__(self, dim: int, act_num: int = 1):
        super().__init__()
        self.act_num = act_num
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(dim, 1, 2 * act_num + 1, 2 * act_num + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        weight_init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        return self.bn(F.conv2d(super().forward(x), self.weight,
                                padding=self.act_num, groups=self.dim))
