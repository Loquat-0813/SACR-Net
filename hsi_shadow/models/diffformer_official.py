from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * torch.sigmoid(gate) + value


class DifferentialMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("DiffFormer attention dim must be divisible by heads.")
        self.heads = heads
        self.dim_head = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.heads, c // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._split(self.to_q(x)), self._split(self.to_k(x)), self._split(self.to_v(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        diff = scores[:, 1:] - scores[:, :-1]
        diff = torch.cat([torch.zeros_like(scores[:, :1]), diff], dim=1)
        attn = torch.softmax(scores + diff, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2])
        return self.to_out(out)


class SpatialSpectralTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        heads: int = 4,
        depth: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": DifferentialMultiHeadSelfAttention(dim=dim, heads=heads, dropout=dropout),
                        "drop1": nn.Dropout(dropout),
                        "norm1": nn.LayerNorm(dim, eps=layernorm_eps),
                        "fc1": nn.Linear(dim, mlp_dim),
                        "act": SwiGLU(),
                        "fc2": nn.Linear(mlp_dim // 2, dim),
                        "drop2": nn.Dropout(dropout),
                        "norm2": nn.LayerNorm(dim, eps=layernorm_eps),
                    }
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            attn = layer["attn"](x)
            x = layer["norm1"](x + layer["drop1"](attn))
            ff = layer["fc2"](layer["act"](layer["fc1"](x)))
            x = layer["norm2"](x + layer["drop2"](ff))
        return x


class DiffFormerBaseline(nn.Module):
    """PyTorch shared-protocol reproduction of DiffFormer's DSST network.

    Source: mahmad000/DiffFormer, ``DiffFormer.ipynb``.  The official release is
    a TensorFlow/Keras notebook using PCA-reduced image cubes and random splits.
    This adapter ports the DSST patch embedding, differential multi-head
    attention, SWiGLU feed-forward block, class token, sinusoidal positional
    encoding, and two-layer classifier to the common PyTorch HanChuan protocol.
    """

    def __init__(
        self,
        bands: int,
        num_classes: int,
        patch_size: int = 11,
        dim: int = 64,
        transformer_layers: int = 2,
        block_depth: int = 2,
        heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-3,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.bands = int(bands)
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv3d(
            in_channels=bands,
            out_channels=dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos_encoding", sinusoidal_position_encoding(2, dim), persistent=False)
        self.blocks = nn.ModuleList(
            [
                SpatialSpectralTransformerBlock(
                    dim=dim,
                    heads=heads,
                    depth=block_depth,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    layernorm_eps=layernorm_eps,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor, illum: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del illum
        if x.ndim != 4 or x.shape[1] != self.bands:
            raise ValueError(f"DiffFormerBaseline expects [N, {self.bands}, H, W], got {tuple(x.shape)}")
        tokens = self.patch_embed(x.unsqueeze(2)).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        if self.pos_encoding.shape[1] < tokens.shape[1]:
            raise ValueError(f"DiffFormer positional buffer is too short for {tokens.shape[1]} tokens.")
        tokens = tokens + self.pos_encoding[:, : tokens.shape[1]].to(tokens.device)
        for block in self.blocks:
            tokens = block(tokens)
        features = tokens[:, 0]
        return {"logits": self.head(features), "features": features}
