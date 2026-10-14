from __future__ import annotations

import argparse
import hashlib
import os
import time
import random
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from hsi_shadow.runtime import ensure_windows_torch_runtime

ensure_windows_torch_runtime()

import torch
from torch.utils.data import WeightedRandomSampler

from hsi_shadow.data.patch_dataset import build_datasets
from hsi_shadow.engine import evaluate, make_loader, save_json, set_seed, train_one_epoch
from adapter import build_model, fit_scale
from module_train import train_one_epoch


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_fingerprint(mask: np.ndarray) -> str:
    """Return a stable fingerprint for the exact labeled split mask."""
    values = np.ascontiguousarray(mask.astype(np.int32, copy=False))
    return hashlib.sha256(values.tobytes()).hexdigest()


def make_class_balanced_sampler(dataset, num_classes: int) -> WeightedRandomSampler:
    coords = getattr(dataset, "coords", None)
    label_mask = getattr(dataset, "label_mask", None)
    if coords is None or label_mask is None:
        raise ValueError("balanced_sampler requires an HsiPatchDataset-like object with coords and label_mask.")
    if len(coords) == 0:
        raise ValueError("balanced_sampler requires a nonempty training dataset.")
    labels = label_mask[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
    if int(labels.min()) < 0 or int(labels.max()) >= int(num_classes):
        raise ValueError("balanced_sampler observed labels outside [0, num_classes).")
    counts = np.bincount(labels, minlength=int(num_classes))
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override (use 0 for Windows/sandbox local preflight).",
    )
    args = parser.parse_args()


    cfg = load_config(args.config)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if args.data_root:
        cfg["data_root"] = args.data_root

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    datasets, meta = build_datasets(cfg)
    fit_scale(cfg, datasets)

    train_cfg = cfg.get("training", {})
    if args.fast_dev_run:
        train_cfg["epochs"] = 1
        train_cfg["max_train_batches"] = 2
        train_cfg["max_eval_batches"] = 2
        cfg["training"] = train_cfg
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative.")
        train_cfg["num_workers"] = args.num_workers
        cfg["training"] = train_cfg
    batch_size = int(train_cfg.get("batch_size", 64))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    num_workers = int(train_cfg.get("num_workers", 0))
    limited_eval = train_cfg.get("max_eval_batches") is not None
    shuffle_limited_eval = bool(train_cfg.get("shuffle_limited_eval", False)) and limited_eval
    train_sampler = None
    if bool(train_cfg.get("balanced_sampler", False)):
        train_sampler = make_class_balanced_sampler(datasets["train"], num_classes=int(meta["num_classes"]))
    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            seed=seed,
            sampler=train_sampler,
        ),
        "val": make_loader(datasets["val"], batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    }

    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    device_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(f"device={device_info}", flush=True)
    model = build_model(
        cfg,
        num_classes=meta["num_classes"],
        input_bands=meta["cube_shape"][2],
        patch_size=int(cfg.get("patch_size", 11)),
    ).to(device)
    if cfg.get("fusion_initialization", "zero") == "conventional":
        # Same architecture and all other initial parameters; only this weight changes.
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed + 900000)
            model.quotient_projection.reset_parameters()
    # Model construction consumes a different amount of RNG for one/two branches.
    # Reset here so the training RNG starts at the same point in paired runs.
    set_seed(seed + 100000)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"{cfg['dataset']}_{cfg['model']['name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE.json").exists():
        raise RuntimeError("Completed run must be skipped by campaign")
    from revision_support import save_provenance
    if not (out_dir / "last.pt").exists():
        save_provenance(out_dir, cfg, datasets, meta, model)
        from revision_support import sha256_file, source_record
        groups = {}
        for name, value in model.state_dict().items():
            group = ('npaf' if name.startswith('core.spatial_stem.2.') else
                     'core' if name.startswith('core.') else name.split('.')[0])
            groups.setdefault(group, {})[name] = hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        save_json(out_dir / 'INITIAL.json', {'groups': groups,
            'parameters': sum(p.numel() for p in model.parameters()),
            'config_sha256': sha256_file(args.config), 'source': source_record()})
    initial = json.loads((out_dir / 'INITIAL.json').read_text())
    scene = cfg['dataset'].removeprefix('whu_')
    for group, cell in [('core','M00'),('dense','M01'),('npaf','M04')]:
        ref = out_dir.parent / f'{scene}_{cell}_s{seed}' / 'INITIAL.json'
        if group in initial['groups'] and ref.exists():
            expected = json.loads(ref.read_text())['groups'][group]
            if initial['groups'][group] != expected:
                raise RuntimeError(f'Paired initial {group} changed')
    save_json(
        out_dir / "meta.json",
        {
            "config": cfg,
            "cube_shape": list(meta["cube_shape"]),
            "num_classes": meta["num_classes"],
            "low_light_threshold": meta["low_light_threshold"],
            "train_samples": len(datasets["train"]),
            "val_samples": len(datasets["val"]),
            "test_samples": len(datasets["test"]),
            "test_low_light_samples": len(datasets.get("test_low_light", [])),
            "split_fingerprints": {
                "train": mask_fingerprint(meta["train_mask"]),
                "val": mask_fingerprint(meta["val_mask"]),
                "test": mask_fingerprint(meta["test_mask"]),
            },
            "device_info": device_info,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "data_preprocess": meta.get("data_preprocess"),
        },
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    epochs = int(train_cfg.get("epochs", 20))
    checkpoint_selection = str(train_cfg.get("checkpoint_selection", "val_oa"))
    has_validation = len(datasets["val"]) > 0
    if checkpoint_selection not in {"val_oa", "final_epoch"}:
        raise ValueError("checkpoint_selection must be 'val_oa' or 'final_epoch'.")
    if checkpoint_selection == "val_oa" and not has_validation:
        raise ValueError("checkpoint_selection='val_oa' requires a nonempty validation split.")
    best_val = -1.0
    history = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    completed_epoch = 0
    elapsed_before = 0.0
    best_state = None
    if (out_dir / 'last.pt').exists():
        # Trusted checkpoint created by this campaign, including Python/NumPy RNG.
        resume = torch.load(out_dir / 'last.pt', map_location='cpu', weights_only=False)
        if resume['config'] != cfg:
            raise RuntimeError('Resume configuration differs')
        from revision_support import source_record
        if resume['source'] != source_record():
            raise RuntimeError('Resume source differs')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        history = resume['history']; best_val = resume['best_val']; best_state = resume['best_state']
        completed_epoch = resume['epoch']; elapsed_before = resume['elapsed_seconds']
        random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng'])
        torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng'])
        loaders['train'].generator.set_state(resume['loader_rng'])
        print('Resuming epoch', completed_epoch + 1, flush=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    from revision_support import source_record
    frozen_source = source_record()
    for epoch in range(completed_epoch + 1, epochs + 1):
        epoch_started = time.perf_counter()
        train_log = train_one_epoch(model, loaders["train"], optimizer, device, train_cfg, epoch=epoch)
        scheduler.step()
        val_log = None
        if has_validation:
            val_log = evaluate(
                model,
                loaders["val"],
                device,
                num_classes=meta["num_classes"],
                max_batches=train_cfg.get("max_eval_batches"),
            )
        row = {"epoch": epoch, "train": train_log, "val": val_log, "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        save_json(out_dir / "history.json", {"history": history})
        if val_log is None:
            print(f"epoch={epoch:03d} loss={train_log['loss']:.4f} selection=final_epoch", flush=True)
        else:
            print(
                f"epoch={epoch:03d} loss={train_log['loss']:.4f} "
                f"val_oa={val_log['oa']:.4f} val_f1={val_log['macro_f1']:.4f}",
                flush=True,
            )
        should_save = checkpoint_selection == "final_epoch" or val_log["oa"] >= best_val
        if should_save:
            if val_log is not None:
                best_val = val_log["oa"]
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        snapshot = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'config': cfg, 'source': frozen_source,
            'history': history, 'epoch': epoch, 'best_val': best_val, 'best_state': best_state,
            'elapsed_seconds': elapsed_before + time.perf_counter()-started,
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'loader_rng': loaders['train'].generator.get_state()}
        torch.save(snapshot, out_dir / 'last.tmp')
        os.replace(out_dir / 'last.tmp', out_dir / 'last.pt')

    torch.save({'model': best_state, 'config': cfg}, out_dir / 'best.tmp')
    os.replace(out_dir / 'best.tmp', out_dir / 'best.pt')

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    selected = max(history, key=lambda row: (row["val"]["oa"], row["epoch"]))
    final = {
        "val": (
            evaluate(
                model,
                loaders["val"],
                device,
                num_classes=meta["num_classes"],
                max_batches=train_cfg.get("max_eval_batches"),
                predictions_path=out_dir / "val_predictions.npz",
            )
            if has_validation
            else None
        ),
        "selected_epoch": selected["epoch"],
        "selected_train": selected["train"],
    }
    save_json(out_dir / "history.json", {"history": history})
    if has_validation:
        predictions = np.load(out_dir / "val_predictions.npz")
        counts = np.zeros((meta["num_classes"], meta["num_classes"]), dtype=np.int64)
        np.add.at(counts, (predictions["y_true"], predictions["y_pred"]), 1)
        np.save(out_dir / "val_confusion.npy", counts)
    save_json(out_dir / "metrics.json", final)
    save_json(out_dir / "timing.json", {"train_and_eval_seconds": elapsed_before + time.perf_counter()-started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "diagnostic_only": bool(args.fast_dev_run or train_cfg.get("max_eval_batches") is not None)})
    from revision_support import sha256_file
    save_json(out_dir / 'DONE.tmp', {'epochs': epochs, 'selected_epoch': selected['epoch'],
        'checkpoint_sha256': sha256_file(out_dir / 'best.pt'),
        'config_sha256': sha256_file(args.config), 'source': frozen_source})
    os.replace(out_dir / 'DONE.tmp', out_dir / 'DONE.json')
    print(f"saved: {out_dir}")
    print(final)


if __name__ == "__main__":
    main()

