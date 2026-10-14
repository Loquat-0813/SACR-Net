from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .losses import consistency_kl, make_relighting_view
from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward_model(model, batch: dict, x: torch.Tensor | None = None) -> dict:
    kwargs = {"illum": batch["illum"]}
    if "quotient" in batch:
        kwargs["quotient"] = batch["quotient"]
    if "amplitude" in batch:
        kwargs["amplitude"] = batch["amplitude"]
    return model(batch["x"] if x is None else x, **kwargs)


def batch_prototype_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"batch_prototype_loss expects [N, D] features, got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("batch_prototype_loss labels must be a [N] tensor aligned with features.")
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    unique, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    if unique.numel() < 2:
        return features.new_tensor(0.0)
    features = F.normalize(features.float(), dim=1)
    prototypes = []
    for class_index in range(unique.numel()):
        prototypes.append(features[inverse == class_index].mean(dim=0))
    prototypes_tensor = F.normalize(torch.stack(prototypes, dim=0), dim=1)
    logits = features @ prototypes_tensor.T
    logits = logits / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, inverse)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"supervised_contrastive_loss expects [N, D] features, got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("supervised_contrastive_loss labels must be a [N] tensor aligned with features.")
    if features.shape[0] < 2:
        return features.new_tensor(0.0)

    labels = labels.view(-1)
    features = F.normalize(features.float(), dim=1)
    logits = features @ features.T
    logits = logits / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
    valid_anchors = positive_mask.sum(dim=1) > 0
    if not bool(valid_anchors.any()):
        return features.new_tensor(0.0)

    logits_mask = ~eye
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_mask_f = positive_mask.float()
    mean_log_prob_pos = (positive_mask_f * log_prob).sum(dim=1) / positive_mask_f.sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos[valid_anchors].mean()


def train_one_epoch(model, loader: DataLoader, optimizer, device, cfg: dict, epoch: int | None = None) -> dict:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_cons = 0.0
    total_gate_l1 = 0.0
    total_proto = 0.0
    total_contrastive = 0.0
    total = 0
    cons_weight = float(cfg.get("consistency_weight", 0.0))
    gate_l1_weight = float(cfg.get("gate_l1_weight", cfg.get("adaptive_gate_l1_weight", 0.0)))
    proto_weight = float(cfg.get("prototype_loss_weight", 0.0))
    contrastive_weight = float(cfg.get("supcon_loss_weight", cfg.get("contrastive_loss_weight", 0.0)))
    proto_warmup = int(cfg.get("prototype_warmup_epochs", 0))
    proto_temperature = float(cfg.get("prototype_temperature", 0.2))
    contrastive_temperature = float(cfg.get("supcon_temperature", cfg.get("contrastive_temperature", 0.1)))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    relight_cfg = cfg.get("relight", {})
    relight_enabled = bool(relight_cfg.get("enabled", False)) and cons_weight > 0
    proto_enabled = proto_weight > 0.0 and (epoch is None or int(epoch) > proto_warmup)
    contrastive_enabled = contrastive_weight > 0.0 and (epoch is None or int(epoch) > proto_warmup)

    max_batches = cfg.get("max_train_batches")
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = batch_to_device(batch, device)
        x = batch["x"]
        y = batch["y"]
        illum = batch["illum"]
        optimizer.zero_grad(set_to_none=True)
        out = forward_model(model, batch, x=x)
        logits = out["logits"]
        ce = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        cons = torch.tensor(0.0, device=device)
        gate_l1 = torch.tensor(0.0, device=device)
        proto = torch.tensor(0.0, device=device)
        contrastive = torch.tensor(0.0, device=device)
        if relight_enabled:
            x_aug = make_relighting_view(
                x,
                min_scale=float(relight_cfg.get("min_scale", 0.45)),
                max_scale=float(relight_cfg.get("max_scale", 1.25)),
            )
            aug_out = forward_model(model, batch, x=x_aug)
            cons = consistency_kl(logits, aug_out["logits"], temperature=float(cfg.get("temperature", 1.0)))
        if gate_l1_weight > 0.0 and "gate" in out:
            gate_l1 = out["gate"].float().mean()
        if proto_enabled or contrastive_enabled:
            metric_features = out.get("metric_features", out.get("features"))
            if metric_features is None:
                raise ValueError("prototype/contrastive losses require model outputs to contain features.")
        if proto_enabled:
            proto = batch_prototype_loss(metric_features, y, temperature=proto_temperature)
        if contrastive_enabled:
            contrastive = supervised_contrastive_loss(metric_features, y, temperature=contrastive_temperature)
        loss = (
            ce
            + cons_weight * cons
            + gate_l1_weight * gate_l1
            + proto_weight * proto
            + contrastive_weight * contrastive
        )
        loss.backward()
        optimizer.step()

        n = int(y.numel())
        total += n
        total_loss += float(loss.item()) * n
        total_ce += float(ce.item()) * n
        total_cons += float(cons.item()) * n
        total_gate_l1 += float(gate_l1.item()) * n
        total_proto += float(proto.item()) * n
        total_contrastive += float(contrastive.item()) * n

    return {
        "loss": total_loss / max(total, 1),
        "ce": total_ce / max(total, 1),
        "consistency": total_cons / max(total, 1),
        "gate_l1": total_gate_l1 / max(total, 1),
        "prototype": total_proto / max(total, 1),
        "contrastive": total_contrastive / max(total, 1),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device, num_classes: int, max_batches: int | None = None, predictions_path=None) -> dict:
    model.eval()
    loss_sum = 0.0
    loss_count = 0
    y_true = []
    y_pred = []
    gates = []
    coords, brightness = [], []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = batch_to_device(batch, device)
        out = forward_model(model, batch)
        loss_sum += float(torch.nn.functional.cross_entropy(out["logits"], batch["y"], reduction="sum"))
        loss_count += batch["y"].numel()
        pred = out["logits"].argmax(dim=1)
        y_true.append(batch["y"].detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        if predictions_path is not None:
            coords.append(batch["coord"].detach().cpu().numpy())
            brightness.append(batch["illum"][:, 1].detach().cpu().numpy())
        if "gate" in out:
            gates.append(out["gate"].detach().cpu().numpy())

    if not y_true:
        return {"num_samples": 0, "oa": 0.0, "aa": 0.0, "kappa": 0.0, "macro_f1": 0.0}
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    metrics = classification_metrics(y_true_np, y_pred_np, num_classes=num_classes)
    metrics["loss"] = loss_sum / max(loss_count, 1)
    if predictions_path is not None:
        np.savez_compressed(predictions_path, y_true=y_true_np, y_pred=y_pred_np,
                            coords=np.concatenate(coords), brightness=np.concatenate(brightness))
    if gates:
        gate_np = np.concatenate(gates)
        metrics["gate_mean"] = float(gate_np.mean())
        metrics["gate_std"] = float(gate_np.std())
    return metrics


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int | None = None,
    sampler=None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
