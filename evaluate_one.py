"""One frozen checkpoint: replay validation, then test without parameter updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from adapter import build_model
from hsi_shadow.data.patch_dataset import AmplitudeStats, BandStats, build_datasets
from hsi_shadow.engine import evaluate, make_loader


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assign_saved_statistics(datasets, stat_path):
    with np.load(stat_path) as stored:
        saved = {k: stored[k].copy() for k in stored.files}
    for ds in datasets.values():
        checks = [
            (ds.band_stats.offset, saved["raw_offset"]),
            (ds.band_stats.scale, saved["raw_scale"]),
            (ds.raw_control_stats.offset, saved["raw_control_stats_offset"]),
            (ds.raw_control_stats.scale, saved["raw_control_stats_scale"]),
        ]
        if ds.amplitude_stats is not None:
            checks.extend([
                (ds.amplitude_stats.offset, saved["amplitude_offset"]),
                (ds.amplitude_stats.scale, saved["amplitude_scale"]),
            ])
        for actual, frozen in checks:
            if not np.allclose(actual, frozen, rtol=0, atol=1e-6):
                raise AssertionError("Frozen training-support statistics do not match reconstructed support")
        ds.band_stats = BandStats(saved["raw_offset"], saved["raw_scale"], ds.band_stats.mode)
        ds.raw_control_stats = BandStats(saved["raw_control_stats_offset"],
                                         saved["raw_control_stats_scale"], ds.raw_control_stats.mode)
        if ds.amplitude_stats is not None:
            ds.amplitude_stats = AmplitudeStats(saved["amplitude_offset"], saved["amplitude_scale"])
    return sorted(saved)


def run(item, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(item["checkpoint_remote"])
    meta_path = Path(item["config_remote"])
    stat_path = Path(item["preprocessing_remote"])
    split_path = Path(item["split_remote"])
    frozen_path = Path(item["frozen_split_remote"])
    for path, expected in (
        (checkpoint, item["checkpoint_sha256"]),
        (meta_path, item["config_sha256"]),
        (stat_path, item["preprocessing_sha256"]),
        (split_path, item["split_sha256_file"]),
        (frozen_path, item["frozen_split_sha256"]),
    ):
        if sha(path) != expected:
            raise AssertionError(f"Source file SHA256 mismatch: {path}")
    meta = read_json(meta_path)
    original_cfg = meta["config"]
    if item["dataset"] != original_cfg["dataset"] or int(original_cfg["seed"]) != item["seed"]:
        raise AssertionError("Config identity mismatch")
    cfg = json.loads(json.dumps(original_cfg))
    cfg["split"]["path"] = str(frozen_path)
    cfg["statistics_support"]["path"] = str(frozen_path)
    if item["model"] == "S":
        cfg["amplitude_variant"] = "none"
    datasets, built_meta = build_datasets(cfg)
    stat_keys = assign_saved_statistics(datasets, stat_path)
    with np.load(split_path) as saved_split:
        for name in ("train", "val", "test"):
            if not np.array_equal(datasets[name].label_mask, saved_split[name]):
                raise AssertionError(f"{name} mask changed")
    if len(datasets["test"]) != item["test_count"]:
        raise AssertionError("Test count changed")
    classes = int(built_meta["num_classes"])
    bands = int(built_meta["cube_shape"][2])
    model = build_model(cfg, classes, bands, int(cfg["patch_size"]))
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if loaded["config"] != original_cfg:
        raise AssertionError("Checkpoint embedded config differs from training meta")
    model.load_state_dict(loaded["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if model.training or any(m.training for m in model.modules()):
        raise AssertionError("Model is not fully in eval mode")
    before = state_sha(model)
    val_file = output_dir / "val_replay_predictions.npz"
    val_loader = make_loader(datasets["val"], batch_size=128, shuffle=False, num_workers=0)
    val = evaluate(model, val_loader, device, num_classes=classes, predictions_path=val_file)
    prior_val = read_json(checkpoint.parent / "metrics.json")["val"]
    for key in ("oa", "macro_f1", "loss"):
        if abs(val[key] - prior_val[key]) > 1e-6:
            raise AssertionError(f"Validation replay changed {key}: {val[key]} vs {prior_val[key]}")
    old_val = checkpoint.parent / "val_predictions.npz"
    if old_val.exists():
        with np.load(old_val) as expected, np.load(val_file) as actual:
            for key in ("coords", "y_true", "y_pred"):
                if not np.array_equal(actual[key], expected[key]):
                    raise AssertionError(f"Validation replay predictions changed: {key}")
    batch_size = 128
    while True:
        try:
            test_loader = make_loader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=0)
            started = time.perf_counter()
            test = evaluate(model, test_loader, device, num_classes=classes,
                            predictions_path=output_dir / "test_predictions.npz")
            seconds = time.perf_counter() - started
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 16:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
    after = state_sha(model)
    if before != after or any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Parameter or BatchNorm buffer changed during inference")
    with np.load(output_dir / "test_predictions.npz") as prediction:
        y = prediction["y_true"]
        pred = prediction["y_pred"]
        coords = prediction["coords"]
    if len(y) != item["test_count"] or not np.array_equal(y, datasets["test"].label_mask[coords[:, 0], coords[:, 1]] - 1):
        raise AssertionError("Test coordinates or label mapping mismatch")
    cm = np.asarray(test["confusion_matrix"], dtype=np.int64)
    if cm.shape != (classes, classes) or cm.sum() != len(y):
        raise AssertionError("Confusion matrix size mismatch")
    np.save(output_dir / "test_confusion.npy", cm)
    test["selected_epoch"] = item["selected_epoch"]
    test["scene"] = item["scene"]
    test["seed"] = item["seed"]
    test["model"] = item["model"]
    test["batch_size"] = batch_size
    test["inference_seconds"] = seconds
    test["state_sha256_before_after"] = before
    test["stat_keys"] = stat_keys
    test["checkpoint_sha256"] = item["checkpoint_sha256"]
    (output_dir / "test_metrics.tmp").write_text(json.dumps(test, indent=2), encoding="utf-8")
    os.replace(output_dir / "test_metrics.tmp", output_dir / "test_metrics.json")
    print(json.dumps({"scene": item["scene"], "seed": item["seed"], "model": item["model"],
                      "samples": len(y), "oa": test["oa"], "macro_f1": test["macro_f1"],
                      "seconds": seconds, "batch_size": batch_size}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    manifest = read_json(args.manifest)
    item = manifest[args.index]
    out = Path(args.output_root) / f"{item['scene']}_{item['model']}_{item['seed']}"
    run(item, out)


if __name__ == "__main__":
    main()
