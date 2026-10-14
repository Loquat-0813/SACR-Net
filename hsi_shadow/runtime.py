from __future__ import annotations

import os
from pathlib import Path


def ensure_windows_torch_runtime() -> None:
    """Add project-local DLL aliases needed by some Windows PyTorch builds."""
    if os.name != "nt":
        return
    project_root = Path(__file__).resolve().parents[1]
    dll_dirs = [
        project_root / "vendor" / "dll",
        Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "bin",
    ]
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

