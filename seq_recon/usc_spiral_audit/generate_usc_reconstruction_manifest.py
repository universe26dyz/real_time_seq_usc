#!/usr/bin/env python3
"""Audit scanned ISMRMRD H5 spiral acquisitions for USC SpiralTFD preparation.

The script reads only H5 headers/XML and separately discovered trajectory MAT
files.  It deliberately leaves values unknown when they are not encoded in the
scanned metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET

import h5py
import matplotlib
import numpy as np
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_H5_ROOT = Path("/media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5")
DEFAULT_SEQUENCE_ROOT = Path("/home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_sequences")


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _scalar(value.item())
    return value


def _json(value: Any) -> Any:
    value = _scalar(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    return next((item for item in element if item.tag.rsplit("}", 1)[-1] == name), None)


def _text(element: ET.Element | None, name: str) -> str | None:
    item = _child(element, name)
    return item.text.strip() if item is not None and item.text else None


def _number(text: str | None) -> int | float | str | None:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _space(element: ET.Element | None, name: str) -> dict[str, Any] | None:
    space = _child(element, name)
    if space is None:
        return None
    matrix = _child(space, "matrixSize")
    fov = _child(space, "fieldOfView_mm")
    return {
        "matrix": {axis: _number(_text(matrix, axis)) for axis in ("x", "y", "z")},
        "field_of_view_mm": {axis: _number(_text(fov, axis)) for axis in ("x", "y", "z")},
    }


def _encoding_limits(element: ET.Element | None) -> dict[str, Any] | None:
    limits = _child(element, "encodingLimits")
    if limits is None:
        return None
    result: dict[str, Any] = {}
    for limit in limits:
        name = limit.tag.rsplit("}", 1)[-1]
        result[name] = {field: _number(_text(limit, field)) for field in ("minimum", "maximum", "center")}
    return result


def parse_ismrmrd_xml(raw_xml: bytes | str | None) -> dict[str, Any]:
    if raw_xml is None:
        return {"status": "not_available", "reason": "dataset/xml is absent"}
    try:
        root = ET.fromstring(raw_xml.decode("utf-8") if isinstance(raw_xml, bytes) else raw_xml)
    except (ET.ParseError, UnicodeDecodeError) as error:
        return {"status": "not_available", "reason": f"XML parse failed: {error}"}

    measurement = _child(root, "measurementInformation")
    system = _child(root, "acquisitionSystemInformation")
    encoding = _child(root, "encoding")
    parameters = _child(root, "sequenceParameters")
    return {
        "status": "available",
        "measurement_information": {
            "measurement_id": _text(measurement, "measurementID"),
            "protocol_name": _text(measurement, "protocolName"),
            "patient_position": _text(measurement, "patientPosition"),
        },
        "acquisition_system_information": (
            {item.tag.rsplit("}", 1)[-1]: _number(item.text.strip()) if item.text else None for item in system}
            if system is not None else None
        ),
        "encoding": {
            "encoded_space": _space(encoding, "encodedSpace"),
            "recon_space": _space(encoding, "reconSpace"),
            "limits": _encoding_limits(encoding),
            "trajectory": _text(encoding, "trajectory"),
        } if encoding is not None else None,
        "sequence_parameters": (
            {item.tag.rsplit("}", 1)[-1]: _number(item.text.strip()) if item.text else None for item in parameters}
            if parameters is not None else None
        ),
    }


def discover_trajectory_summaries(sequence_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sequence_root.rglob("DIAGNOSTIC_SUMMARY.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            trajectory_path = Path(summary["trajectory_path"])
            if trajectory_path.is_file():
                summaries.append({"summary_path": path, "summary": summary, "trajectory_path": trajectory_path})
        except (OSError, ValueError, KeyError):
            continue
    return summaries


def match_trajectory(protocol_name: str | None, summaries: list[dict[str, Any]]) -> tuple[Path | None, str | None]:
    if not protocol_name:
        return None, "ISMRMRD XML has no protocol name"
    token = re.sub(r"^pulseq_", "", protocol_name, flags=re.IGNORECASE).lower()
    matches = [item for item in summaries if str(item["summary"].get("test_id", "")).lower().startswith(token)]
    if len(matches) == 1:
        return matches[0]["trajectory_path"], None
    if not matches:
        return None, f"no diagnostic summary test_id starts with {token!r}"
    return None, f"multiple trajectory summaries match {token!r}"


def _mat_value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def inspect_trajectory(path: Path | None) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    if path is None:
        return {"status": "not_available", "reason": "trajectory could not be matched"}, None, None
    try:
        data = loadmat(path, squeeze_me=True, variable_names=[
            "base_k_played", "base_k", "global_acquisition_index", "global_arm_angle_deg",
            "slice_index_per_acq", "arm_index_in_slice",
        ])
    except (OSError, ValueError, NotImplementedError) as error:
        return {"status": "not_available", "path": str(path), "reason": f"MAT read failed: {error}"}, None, None

    coordinates = _mat_value(data, "base_k_played")
    coordinate_key = "base_k_played"
    if coordinates is None:
        coordinates, coordinate_key = _mat_value(data, "base_k"), "base_k"
    if coordinates is None or np.asarray(coordinates).ndim != 2 or np.asarray(coordinates).shape[1] < 2:
        return {"status": "not_available", "path": str(path), "reason": "no usable 2-D base trajectory array"}, None, None

    coordinates = np.asarray(coordinates)
    arm_indices = _mat_value(data, "global_acquisition_index")
    if arm_indices is None:
        arm_indices = _mat_value(data, "global_arm_index")
    n_arms = int(np.asarray(arm_indices).size) if arm_indices is not None else None
    return {
        "status": "available",
        "path": str(path.resolve()),
        "coordinate_field": coordinate_key,
        "shape": list(coordinates.shape),
        "dtype": str(coordinates.dtype),
        "coordinate_range": {
            "kx": [float(np.min(coordinates[:, 0])), float(np.max(coordinates[:, 0]))],
            "ky": [float(np.min(coordinates[:, 1])), float(np.max(coordinates[:, 1]))],
        },
        "samples_per_base_arm": int(coordinates.shape[0]),
        "number_of_acquisition_arms_from_mat": n_arms,
        "normalization": "unknown: trajectory coordinate units are not encoded in the MAT metadata",
    }, coordinates, _mat_value(data, "global_arm_angle_deg")


def summarize_headers(dataset: h5py.Dataset) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    headers = dataset["head"]
    names = headers.dtype.names or ()
    sample_counts = np.asarray(headers["number_of_samples"], dtype=np.int64)
    active_channels = np.asarray(headers["active_channels"], dtype=np.int64)
    flags = np.asarray(headers["flags"], dtype=np.uint64)
    first_data = np.asarray(dataset[0]["data"]) if len(dataset) else np.array([])
    index = headers["idx"] if "idx" in names else None
    flag_counts = Counter(int(value) for value in flags)
    return {
        "n_acquisitions": int(len(dataset)),
        "samples_per_acquisition": {"unique_values": np.unique(sample_counts).tolist(), "consistent": bool(len(np.unique(sample_counts)) == 1)},
        "channels": {"active_unique_values": np.unique(active_channels).tolist(), "consistent": bool(len(np.unique(active_channels)) == 1)},
        "data_dtype": str(first_data.dtype) if first_data.size else None,
        "data_elements_per_acquisition": int(first_data.size) if first_data.size else None,
        "h5_data_field_dtype": str(dataset.dtype.fields["data"][0]) if "data" in (dataset.dtype.names or ()) else None,
        "acquisition_flags_summary": {f"0x{key:016x}": value for key, value in sorted(flag_counts.items())},
        "discard_pre_unique": np.unique(headers["discard_pre"]).tolist(),
        "discard_post_unique": np.unique(headers["discard_post"]).tolist(),
        "sample_time_us_unique": np.unique(headers["sample_time_us"]).astype(float).tolist(),
        "trajectory_dimensions_unique": np.unique(headers["trajectory_dimensions"]).tolist(),
        "slice_indices": np.unique(index["slice"]).astype(int).tolist() if index is not None else None,
    }, {"headers": headers, "slice": np.asarray(index["slice"], dtype=np.int64) if index is not None else np.array([], dtype=np.int64)}


def analyze_timing(headers: np.ndarray, xml: dict[str, Any]) -> dict[str, Any]:
    timestamps = np.asarray(headers["acquisition_time_stamp"], dtype=np.int64)
    diffs = np.diff(timestamps)
    tr = (xml.get("sequence_parameters") or {}).get("TR") if xml.get("status") == "available" else None
    return {
        "sequence_TR_from_ismrmrd_xml_ms": {"value": tr, "source": "ISMRMRD XML sequenceParameters.TR" if tr is not None else "unknown"},
        "acquisition_timestamp": {
            "available": bool(timestamps.size),
            "unit": "raw scanner timestamp ticks; no tick-to-second conversion is asserted by this audit",
            "first": int(timestamps[0]) if timestamps.size else None,
            "last": int(timestamps[-1]) if timestamps.size else None,
            "monotonic_non_decreasing": bool(np.all(diffs >= 0)) if diffs.size else None,
            "delta_statistics": ({"mean": float(np.mean(diffs)), "std": float(np.std(diffs)), "min": int(np.min(diffs)), "max": int(np.max(diffs))} if diffs.size else None),
        },
    }


def analyze_ordering(slice_index: np.ndarray, headers: np.ndarray) -> dict[str, Any]:
    if not slice_index.size:
        return {"pattern": "unknown", "confidence": "low", "reason": "acquisition header has no slice index"}
    changes = np.r_[True, slice_index[1:] != slice_index[:-1]]
    run_slices = slice_index[changes]
    run_lengths = np.diff(np.r_[np.flatnonzero(changes), len(slice_index)])
    timestamps = np.asarray(headers["acquisition_time_stamp"], dtype=np.int64)
    evidence = {
        "slice_run_values": run_slices.astype(int).tolist(),
        "slice_run_lengths": run_lengths.astype(int).tolist(),
        "timestamps_monotonic_non_decreasing": bool(np.all(np.diff(timestamps) >= 0)) if timestamps.size > 1 else None,
    }
    if len(run_slices) == len(np.unique(slice_index)) and len(np.unique(run_slices)) == len(run_slices):
        return {"pattern": "slice-major", "confidence": "high", "reason": "each header slice index occupies one contiguous run in stored acquisition order", "evidence": evidence}
    if len(run_slices) > len(np.unique(slice_index)):
        return {"pattern": "interleaved", "confidence": "high", "reason": "header slice index returns after changing in stored acquisition order", "evidence": evidence}
    return {"pattern": "unknown", "confidence": "low", "reason": "slice-index runs are insufficient to classify ordering", "evidence": evidence}


def analyze_oversampling(acquisition: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    samples = acquisition["samples_per_acquisition"]["unique_values"]
    raw = samples[0] if len(samples) == 1 else None
    trajectory_samples = trajectory.get("samples_per_base_arm") if trajectory.get("status") == "available" else None
    if raw is not None and trajectory_samples and raw % trajectory_samples == 0:
        factor = raw // trajectory_samples
        return {"detected": factor > 1, "factor": int(factor), "raw_samples": int(raw), "trajectory_samples": int(trajectory_samples), "possible_averaged_sample_count": int(raw // factor), "interpretation": "integer raw/trajectory sample ratio; reconstruction must independently confirm averaging versus another vendor export transform"}
    return {"detected": False, "factor": None, "raw_samples": raw, "trajectory_samples": trajectory_samples, "possible_averaged_sample_count": None, "interpretation": "no integer raw/trajectory sample ratio could be established"}


def reconstruction_estimate(slice_index: np.ndarray, user_arms_per_frame: int | None) -> dict[str, Any]:
    if slice_index.size:
        counts = Counter(int(value) for value in slice_index)
        values = sorted(counts.values())
        arms = values[0] if len(set(values)) == 1 else None
        arms_entry = {"value": arms, "source": "ISMRMRD acquisition header idx.slice counts" if arms is not None else "unknown", "per_slice_counts": {str(key): value for key, value in sorted(counts.items())}}
    else:
        arms_entry = {"value": None, "source": "unknown", "per_slice_counts": None}
    frame_entry = {"value": user_arms_per_frame, "source": "user_defined" if user_arms_per_frame else "unknown", "note": "set --user-arms-per-frame to make a reconstruction grouping choice explicit"}
    frames = arms_entry["value"] // user_arms_per_frame if arms_entry["value"] and user_arms_per_frame and arms_entry["value"] % user_arms_per_frame == 0 else None
    return {"arms_per_slice": arms_entry, "arms_per_frame": frame_entry, "frames_per_slice": {"value": frames, "source": "derived from measured arms_per_slice and user_defined arms_per_frame" if frames is not None else "unknown"}}


def write_qc(output_dir: Path, coordinates: np.ndarray | None, angles: np.ndarray | None, slice_index: np.ndarray, headers: np.ndarray, user_arms_per_frame: int | None) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if coordinates is not None:
        fig, axis = plt.subplots(figsize=(6, 6))
        draw_angles = np.asarray(angles).ravel()[:12] if angles is not None else np.array([0.0])
        for angle in draw_angles:
            radians = np.deg2rad(float(angle))
            rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
            rotated = coordinates[:, :2] @ rotation.T
            axis.plot(rotated[:, 0], rotated[:, 1], linewidth=0.5)
        axis.set(title="Base trajectory and first available arm rotations", xlabel="kx", ylabel="ky", aspect="equal")
        path = output_dir / "trajectory.png"; fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(str(path))
    if slice_index.size:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.scatter(np.arange(slice_index.size), slice_index, s=1)
        axis.set(title="Acquisition ordering from ISMRMRD idx.slice", xlabel="stored acquisition index", ylabel="slice index")
        path = output_dir / "slice_ordering.png"; fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(str(path))
    timestamps = np.asarray(headers["acquisition_time_stamp"], dtype=np.int64)
    if timestamps.size:
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.plot(np.arange(timestamps.size), timestamps, linewidth=0.5)
        axis.set(title="Raw acquisition timestamps", xlabel="stored acquisition index", ylabel="timestamp ticks")
        path = output_dir / "acquisition_timeline.png"; fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(str(path))
    fig, axis = plt.subplots(figsize=(8, 2.5))
    if user_arms_per_frame and slice_index.size:
        count = int(np.sum(slice_index == slice_index[0]))
        arm = np.arange(count)
        axis.scatter(arm, arm // user_arms_per_frame, s=4)
        axis.set(title=f"User-defined frame grouping preview ({user_arms_per_frame} arms/frame)", xlabel="arm index in first slice", ylabel="frame index")
    else:
        axis.text(0.5, 0.5, "No frame grouping preview: --user-arms-per-frame was not supplied", ha="center", va="center")
        axis.set_axis_off()
    path = output_dir / "frame_grouping_preview.png"; fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(str(path))
    return paths


def audit_one(h5_path: Path, summaries: list[dict[str, Any]], user_arms_per_frame: int | None, make_qc: bool) -> Path:
    with h5py.File(h5_path, "r") as handle:
        dataset = handle.get("dataset/data")
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{h5_path}: dataset/data is absent")
        raw_xml = handle["dataset/xml"][0] if "dataset/xml" in handle else None
        xml = parse_ismrmrd_xml(raw_xml)
        acquisition, values = summarize_headers(dataset)
    protocol = (xml.get("measurement_information") or {}).get("protocol_name")
    trajectory_path, trajectory_reason = match_trajectory(protocol, summaries)
    trajectory, coordinates, angles = inspect_trajectory(trajectory_path)
    if trajectory_reason:
        trajectory["match_reason"] = trajectory_reason
    output = {
        "scan_name": protocol or h5_path.parent.name,
        "file_info": {"h5_path": str(h5_path.resolve()), "trajectory_path": str(trajectory_path.resolve()) if trajectory_path else None, "generated_time": datetime.now(timezone.utc).isoformat()},
        "ismrmrd_metadata": xml,
        "acquisition": acquisition,
        "oversampling": analyze_oversampling(acquisition, trajectory),
        "timing": analyze_timing(values["headers"], xml),
        "ordering": analyze_ordering(values["slice"], values["headers"]),
        "trajectory": trajectory,
        "reconstruction_estimate": reconstruction_estimate(values["slice"], user_arms_per_frame),
        "limitations": ["No value is taken from Pulseq design scripts.", "Timestamp ticks are retained without a seconds conversion because the H5 metadata does not state one.", "The XML trajectory field is reported verbatim and is not treated as a replacement for the separately supplied trajectory MAT."],
    }
    if make_qc:
        output["qc_files"] = write_qc(h5_path.parent / "qc_usc_acquisition", coordinates, angles, values["slice"], values["headers"], user_arms_per_frame)
    destination = h5_path.parent / "usc_reconstruction_manifest.json"
    destination.write_text(json.dumps(_json(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-root", type=Path, default=DEFAULT_H5_ROOT)
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--user-arms-per-frame", type=int, default=None, help="Optional reconstruction-only grouping; never inferred from scanner metadata.")
    parser.add_argument("--no-qc", action="store_true", help="Do not write PNG QC files.")
    args = parser.parse_args()
    if args.user_arms_per_frame is not None and args.user_arms_per_frame <= 0:
        parser.error("--user-arms-per-frame must be positive")
    h5_paths = sorted(args.h5_root.rglob("*.h5"))
    if not h5_paths:
        parser.error(f"no H5 files found below {args.h5_root}")
    summaries = discover_trajectory_summaries(args.sequence_root)
    for h5_path in h5_paths:
        print(audit_one(h5_path, summaries, args.user_arms_per_frame, not args.no_qc))


if __name__ == "__main__":
    main()
