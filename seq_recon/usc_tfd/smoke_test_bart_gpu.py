#!/usr/bin/env python3
"""Tiny synthetic BART Python and GPU-NUFFT smoke test; never reads T13 data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def run_smoke(bart_toolbox_path: Path) -> dict[str, object]:
    bart_root = bart_toolbox_path.resolve()
    python_dir = bart_root / "python"
    if not (bart_root / "bart").is_file() or not python_dir.is_dir():
        raise RuntimeError("BART toolbox must contain bart and python/")
    os.environ["BART_TOOLBOX_PATH"] = str(bart_root)
    os.environ["TOOLBOX_PATH"] = str(bart_root)  # supported legacy fallback in BART v0.9.00
    sys.path.insert(0, str(python_dir))
    module = importlib.import_module("bart")
    if not callable(getattr(module, "bart", None)):
        raise RuntimeError("bart.bart is unavailable")

    source = np.ones((4, 4), dtype=np.complex64)
    fft = module.bart(1, "fft -u 3", source)
    if not np.all(np.isfinite(fft)):
        raise RuntimeError("BART Python NumPy round-trip produced non-finite FFT output")

    # BART v0.9.00 accepts the same -x dimensions syntax as USC's formal
    # scale-estimation path. This validates CUDA execution on synthetic data.
    trajectory = np.zeros((3, 8, 1), dtype=np.complex64)
    kspace = np.ones((1, 8, 1), dtype=np.complex64)
    image = module.bart(1, "nufft -g -x 8:8:1 -a", trajectory, kspace)
    if image.shape[0:2] != (8, 8) or not np.all(np.isfinite(image)):
        raise RuntimeError(f"GPU NUFFT output invalid: shape={image.shape}")
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bart_toolbox_path": str(bart_root),
        "bart_python_module": getattr(module, "__file__", "imported"),
        "numpy_python_roundtrip": "PASS",
        "gpu_nufft_command": "nufft -g -x 8:8:1 -a",
        "gpu_nufft": "PASS",
        "input_is_synthetic_only": True,
        "overall": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart-toolbox-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_smoke(args.bart_toolbox_path)
    except Exception as error:
        report = {"overall": "FAIL", "error": repr(error), "input_is_synthetic_only": True}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
