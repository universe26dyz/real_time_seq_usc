#!/usr/bin/env python3
"""Read-only BART v0.9.00 and GPU readiness report for Step 2B-1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


EXPECTED_BART_VERSION = "v0.9.00"
REQUIRED_OPTIONS = {
    "nufft": ("-g", "-x", "-a"),
    "nlinv": ("-a", "-b", "-S", "-d", "-i", "-x", "-t"),
    "pics": ("-g", "-m", "-w", "-e", "-S", "-R", "-d", "-i", "-t"),
}


def _result(ok: bool, detail: str) -> dict[str, str]:
    return {"status": "PASS" if ok else "FAIL", "detail": detail}


def _run(command: list[str]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return None, repr(error)
    return completed.returncode, (completed.stdout or completed.stderr).strip()


def _bart_version(executable: str | None) -> dict[str, str]:
    if executable is None:
        return _result(False, "bart executable unavailable")
    for args in ([executable, "version", "-V"], [executable, "version"]):
        code, detail = _run(args)
        if code == 0:
            return _result(True, detail or "version command succeeded")
    return _result(False, detail or "bart version command failed")


def _solver_commands(executable: str | None) -> dict[str, dict[str, str]]:
    names = ("nlinv", "nufft", "resize", "normalize", "pics")
    if executable is None:
        return {name: _result(False, "bart executable unavailable") for name in names}
    result: dict[str, dict[str, str]] = {}
    for name in names:
        code, detail = _run([executable, name, "-h"])
        # Some BART releases return nonzero after displaying command help.
        recognized = code is not None and bool(detail) and "unknown command" not in detail.lower()
        result[name] = _result(recognized, detail or "no command help output")
    return result


def _option_capabilities(commands: dict[str, dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        command: {
            option: _result(option in commands[command]["detail"], f"{command} {option}")
            for option in options
        }
        for command, options in REQUIRED_OPTIONS.items()
    }


def _smoke_result(path: Path | None) -> dict[str, str]:
    if path is None:
        return _result(False, "not verified; run smoke_test_bart_gpu.py on the GPU server")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _result(False, f"could not read smoke report: {error}")
    return _result(raw.get("overall") == "PASS", str(path.resolve()))


def inspect_environment(
    bart_toolbox_path: Path | None,
    expected_conda_env: str = "Pulseq_gpu",
    gpu_smoke_report: Path | None = None,
) -> dict[str, Any]:
    configured = bart_toolbox_path or (Path(os.environ["BART_TOOLBOX_PATH"]) if os.environ.get("BART_TOOLBOX_PATH") else None)
    report: dict[str, Any] = {
        "expected_bart_version": EXPECTED_BART_VERSION,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "conda_environment": _result(os.environ.get("CONDA_DEFAULT_ENV") == expected_conda_env, os.environ.get("CONDA_DEFAULT_ENV", "unset")),
        "BART_TOOLBOX_PATH": _result(configured is not None and configured.is_dir(), str(configured) if configured else "unset"),
        "TOOLBOX_PATH_legacy": _result(True, os.environ.get("TOOLBOX_PATH", "unset (optional legacy fallback)")),
        "CUDA_VISIBLE_DEVICES": _result(True, os.environ.get("CUDA_VISIBLE_DEVICES", "unset (CUDA default visibility)")),
    }
    executable = shutil.which("bart")
    if executable is None and configured is not None:
        candidate = configured / "bart"
        executable = str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    report["bart_executable"] = _result(executable is not None, executable or "not found")
    report["bart_version"] = _bart_version(executable)
    report["bart_version_identity"] = _result(
        report["bart_version"]["status"] == "PASS" and EXPECTED_BART_VERSION in report["bart_version"]["detail"],
        f"expected {EXPECTED_BART_VERSION}; observed {report['bart_version']['detail']}",
    )

    python_dir = configured / "python" if configured is not None else None
    if python_dir is not None and python_dir.is_dir():
        sys.path.insert(0, str(python_dir))
    try:
        module = importlib.import_module("bart")
        report["bart_python_module"] = _result(True, getattr(module, "__file__", "imported"))
        api = callable(getattr(module, "bart", None))
        report["bart_python_api"] = _result(api, "bart.bart callable" if api else "bart.bart missing or not callable")
    except Exception as error:
        report["bart_python_module"] = _result(False, repr(error))
        report["bart_python_api"] = _result(False, "bart module import failed")

    gpu_code, gpu_detail = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    report["gpu_visibility"] = _result(gpu_code == 0, gpu_detail or "nvidia-smi unavailable")
    cuda_code, cuda_detail = _run(["nvcc", "--version"])
    report["cuda_compiler"] = _result(cuda_code == 0, cuda_detail or "nvcc unavailable (not required for an already-built BART)")
    packages: dict[str, dict[str, str]] = {}
    for name in ("numpy", "scipy", "ismrmrd", "tomli"):
        try:
            module = importlib.import_module(name)
            packages[name] = _result(True, getattr(module, "__version__", "imported"))
        except Exception as error:
            packages[name] = _result(False, repr(error))
    report["python_packages"] = packages
    commands = _solver_commands(executable)
    report["solver_commands"] = commands
    capabilities = _option_capabilities(commands)
    report["required_cli_capabilities"] = capabilities
    report["bart_gpu_smoke_test"] = _smoke_result(gpu_smoke_report)

    required = ("conda_environment", "BART_TOOLBOX_PATH", "bart_executable", "bart_version", "bart_version_identity", "bart_python_module", "bart_python_api", "gpu_visibility", "bart_gpu_smoke_test")
    ready = all(report[key]["status"] == "PASS" for key in required)
    ready = ready and all(item["status"] == "PASS" for item in packages.values()) and all(item["status"] == "PASS" for item in commands.values())
    ready = ready and all(item["status"] == "PASS" for command in capabilities.values() for item in command.values())
    report["overall"] = "PASS" if ready else "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bart-toolbox-path", type=Path)
    parser.add_argument("--expected-conda-env", default="Pulseq_gpu")
    parser.add_argument("--gpu-smoke-report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = inspect_environment(args.bart_toolbox_path, args.expected_conda_env, args.gpu_smoke_report)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
