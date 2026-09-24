#!/usr/bin/env python3
"""Read-only prerequisite report for a future BART GPU execution."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _result(ok: bool, detail: str) -> dict[str, object]:
    return {"status": "PASS" if ok else "FAIL", "detail": detail}


def _command_output(command: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return False, repr(error)
    return completed.returncode == 0, (completed.stdout or completed.stderr).strip()


def inspect_environment(bart_toolbox_path: Path | None) -> dict[str, object]:
    configured = bart_toolbox_path or (Path(os.environ["BART_TOOLBOX_PATH"]) if os.environ.get("BART_TOOLBOX_PATH") else None)
    report: dict[str, object] = {"BART_TOOLBOX_PATH": _result(configured is not None and configured.is_dir(), str(configured) if configured else "unset")}
    executable = shutil.which("bart")
    if executable is None and configured is not None:
        candidate = configured / "bart"
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    report["bart_executable"] = _result(executable is not None, executable or "not found")
    if executable:
        version_ok, version = _command_output([executable, "version"])
        report["bart_version"] = _result(version_ok, version or "version unavailable")
    else:
        report["bart_version"] = _result(False, "bart executable unavailable")

    python_dir = configured / "python" if configured is not None else None
    if python_dir is not None and python_dir.is_dir():
        sys.path.insert(0, str(python_dir))
    try:
        module = importlib.import_module("bart")
        report["bart_python_module"] = _result(True, getattr(module, "__file__", "imported"))
    except Exception as error:
        report["bart_python_module"] = _result(False, repr(error))

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    report["CUDA_VISIBLE_DEVICES"] = _result(cuda_visible != "", cuda_visible)
    gpu_ok, gpu_detail = _command_output(["nvidia-smi", "-L"])
    report["gpu_visibility"] = _result(gpu_ok, gpu_detail or "nvidia-smi unavailable")
    packages = {}
    for name in ("numpy", "scipy", "ismrmrd", "tomli"):
        try:
            module = importlib.import_module(name)
            packages[name] = _result(True, getattr(module, "__version__", "imported"))
        except Exception as error:
            packages[name] = _result(False, repr(error))
    report["python_packages"] = packages
    report["overall"] = "PASS" if all(
        value.get("status") == "PASS" for key, value in report.items() if key in {"BART_TOOLBOX_PATH", "bart_executable", "bart_python_module", "gpu_visibility"}
    ) and all(value["status"] == "PASS" for value in packages.values()) else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart-toolbox-path", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = inspect_environment(args.bart_toolbox_path)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
