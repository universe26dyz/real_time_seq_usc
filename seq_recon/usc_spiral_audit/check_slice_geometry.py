#!/usr/bin/env python3
"""Report actual slice geometry from scanned ISMRMRD acquisition headers.

The H5 acquisition header is preferred.  Optional DICOM lookup is only used
when H5 does not contain physical positions; nothing is estimated from design
parameters or scanner GUI settings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_H5_ROOT = Path("/media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5")


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((item for item in element if item.tag.rsplit("}", 1)[-1] == name), None)


def protocol_name(handle: h5py.File) -> str | None:
    if "dataset/xml" not in handle:
        return None
    try:
        raw = handle["dataset/xml"][0]
        root = ET.fromstring(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        measurement = _child(root, "measurementInformation")
        value = _child(measurement, "protocolName")
        return value.text.strip() if value is not None and value.text else None
    except (ET.ParseError, UnicodeDecodeError):
        return None


def header_geometry(dataset: h5py.Dataset) -> dict[str, Any]:
    headers = dataset["head"]
    if "idx" not in (headers.dtype.names or ()):
        return {"status": "not_available", "reason": "ISMRMRD acquisition header idx is absent"}
    slices = np.asarray(headers["idx"]["slice"], dtype=np.int64)
    positions = np.asarray(headers["position"], dtype=float)
    read_dir = np.asarray(headers["read_dir"], dtype=float)
    phase_dir = np.asarray(headers["phase_dir"], dtype=float)
    slice_dir = np.asarray(headers["slice_dir"], dtype=float)
    if not slices.size or not positions.size or not np.any(np.isfinite(positions)):
        return {"status": "not_available", "reason": "ISMRMRD acquisition headers do not contain usable slice positions"}

    unique_slices = np.unique(slices)
    position_by_slice: list[np.ndarray] = []
    orientation_by_slice: list[np.ndarray] = []
    consistency: dict[str, float] = {}
    for slice_id in unique_slices:
        mask = slices == slice_id
        slice_positions = positions[mask]
        representative = np.median(slice_positions, axis=0)
        position_by_slice.append(representative)
        consistency[str(int(slice_id))] = float(np.max(np.linalg.norm(slice_positions - representative, axis=1)))
        orientation_by_slice.append(np.median(slice_dir[mask], axis=0))
    position_array = np.asarray(position_by_slice)
    normal = np.median(np.asarray(orientation_by_slice), axis=0)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm > 0:
        normal /= normal_norm
    deltas = np.diff(position_array, axis=0)
    spacing = np.linalg.norm(deltas, axis=1)
    projected = np.abs(deltas @ normal) if normal_norm > 0 else None
    all_consistent = max(consistency.values(), default=float("inf")) < 1e-3
    spacing_consistent = bool(spacing.size and np.ptp(spacing) < 1e-3)
    estimated = float(np.median(spacing)) if spacing.size else None
    option = "approximately_6_mm" if estimated is not None and abs(estimated - 6.0) < 0.05 else ("approximately_2_mm" if estimated is not None and abs(estimated - 2.0) < 0.05 else "neither_2_nor_6_mm")
    return {
        "status": "available",
        "source": "ISMRMRD acquisition header",
        "confidence": "high" if all_consistent and spacing_consistent else "medium",
        "slice_count": int(unique_slices.size),
        "slice_indices": unique_slices.astype(int).tolist(),
        "positions_mm": position_array.tolist(),
        "position_unit": "mm (ISMRMRD AcquisitionHeader.position)",
        "spacing_mm": spacing.tolist(),
        "through_plane_spacing_mm": projected.tolist() if projected is not None else None,
        "estimated_spacing_mm": estimated,
        "spacing_consistent": spacing_consistent,
        "per_slice_position_variation_mm": consistency,
        "orientation": {"read_dir": np.median(read_dir, axis=0).tolist(), "phase_dir": np.median(phase_dir, axis=0).tolist(), "slice_dir": normal.tolist() if normal_norm > 0 else None},
        "gui_spacing_question": {"measured_result": option, "conclusion": ("acquisition-header positions confirm approximately 6 mm through-plane spacing" if option == "approximately_6_mm" else "acquisition-header positions do not support a 2 mm or 6 mm conclusion")},
    }


def write_plot(destination: Path, geometry: dict[str, Any]) -> str | None:
    if geometry.get("status") != "available":
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(geometry["positions_mm"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for component, label in enumerate(("x", "y", "z")):
        axes[0].plot(geometry["slice_indices"], positions[:, component], marker="o", label=label)
    axes[0].set(title="ISMRMRD header positions", xlabel="slice index", ylabel="position (mm)")
    axes[0].legend()
    axes[1].plot(np.arange(1, len(geometry["spacing_mm"]) + 1), geometry["spacing_mm"], marker="o")
    axes[1].set(title="Consecutive slice spacing", xlabel="slice transition", ylabel="distance (mm)")
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)
    return str(destination)


def audit_one(h5_path: Path) -> Path:
    with h5py.File(h5_path, "r") as handle:
        dataset = handle.get("dataset/data")
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{h5_path}: dataset/data is absent")
        name = protocol_name(handle) or h5_path.parent.name
        geometry = header_geometry(dataset)
    output = {"scan_name": name, "h5_path": str(h5_path.resolve()), "generated_time": datetime.now(timezone.utc).isoformat(), **geometry}
    output["qc_slice_positions_png"] = write_plot(h5_path.parent / "qc_usc_acquisition" / "slice_positions.png", geometry)
    destination = h5_path.parent / "slice_geometry_report.json"
    destination.write_text(json.dumps(_json(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, default=DEFAULT_H5_ROOT)
    args = parser.parse_args()
    files = sorted(args.h5_root.rglob("*.h5"))
    if not files:
        parser.error(f"no H5 files found below {args.h5_root}")
    for path in files:
        print(audit_one(path))


if __name__ == "__main__":
    main()
