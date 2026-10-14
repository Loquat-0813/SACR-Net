from __future__ import annotations

import torch
import torch.nn.functional as F


def make_relighting_view(
    x: torch.Tensor,
    min_scale: float = 0.45,
    max_scale: float = 1.25,
) -> torch.Tensor:
    b = x.shape[0]
    scale = torch.empty((b, 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(min_scale, max_scale)
    return (x * scale).clamp(0.0, 1.0)


def consistency_kl(logits: torch.Tensor, aug_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    p = F.softmax(logits.detach() / temperature, dim=1)
    log_q = F.log_softmax(aug_logits / temperature, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean") * (temperature**2)

