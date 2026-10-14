"""Provenance helpers for the JARS revision; no outcome-based selection."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record():
    root = Path(__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*.py")) if not any(x in p.parts for x in ("runs", "vendor"))}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def parameter_hash(model, shared_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if shared_only and not name.startswith("core."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_provenance(out_dir, cfg, datasets, meta, model):
    from hsi_shadow.data.registry import get_dataset_spec
    out_dir = Path(out_dir)
    spec = get_dataset_spec(cfg["dataset"])
    data_files = {name: sha256_file(Path(cfg["data_root"]) / name)
                  for name in (spec.image_path, spec.gt_path)}
    train = datasets["train"]
    stats = {"raw_offset": train.band_stats.offset, "raw_scale": train.band_stats.scale}
    for name in ("raw_quotient_stats", "raw_control_stats"):
        value = getattr(train, name)
        if value is not None:
            stats[name + "_offset"] = value.offset
            stats[name + "_scale"] = value.scale
    if getattr(train, "amplitude_stats", None) is not None:
        stats["amplitude_offset"] = train.amplitude_stats.offset
        stats["amplitude_scale"] = train.amplitude_stats.scale
    np.savez_compressed(out_dir / "preprocessing.npz", **stats)
    np.savez_compressed(out_dir / "splits.npz", **{name: meta[name + "_mask"] for name in ("train", "val", "test")})
    record = {"python": platform.python_version(), "source": source_record(), "data_sha256": data_files,
              "initial_model_sha256": parameter_hash(model), "initial_shared_sha256": parameter_hash(model, True),
              "split_sha256": sha256_file(cfg["split"]["path"]) if cfg["split"]["type"] == "frozen" else None,
              "class_counts": {name: np.bincount(ds.label_mask[ds.label_mask > 0], minlength=meta["num_classes"]+1)[1:].tolist()
                               for name, ds in datasets.items()},
              "primary_metric": "all-development Macro-F1 over fixed full class list",
              "statistics_support": meta.get("statistics_support"),
              "diagnostic_only": bool(cfg.get("revision", {}).get("smoke", False))}
    (out_dir / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

