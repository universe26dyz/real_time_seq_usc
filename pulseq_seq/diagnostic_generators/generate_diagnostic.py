"""Shared thin launcher for T13--T15 GA policy diagnostics."""

import copy
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_loader import load_config
from write_rtspiral_svr_uih import build_sequence


def run(test_id: str, ga_policy: str, num_slices: int, sequence_name: str) -> None:
    config = copy.deepcopy(load_config(str(PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json")))
    config["geometry"]["num_slices"] = num_slices
    output_root = PROJECT_ROOT / "diagnostic_sequences" / test_id
    result = build_sequence(config, output_root, ga_policy=ga_policy, sequence_filename=sequence_name)
    metadata = result.metadata
    fov_z_mm = (num_slices - 1) * config["geometry"]["slice_shift_mm"] + config["geometry"]["slice_thickness_mm"]
    summary = {
        "test_id": test_id,
        "ga_policy": ga_policy,
        "ga_angle_deg": config["spiral"]["ga_angle_deg"],
        "ga_lut_length": metadata["ga_lut_length"],
        "slice_reset": metadata["slice_reset"],
        "num_slices": num_slices,
        "arms_per_slice": metadata["played_arms_per_slice"],
        "default_reconstruction_arms_per_frame": metadata["default_reconstruction_arms_per_frame"],
        "slice_thickness_mm": config["geometry"]["slice_thickness_mm"],
        "slice_shift_mm": config["geometry"]["slice_shift_mm"],
        "fov_z_mm": fov_z_mm,
        "slice_positions_mm": [position * 1e3 for position in metadata["slice_positions_m"]],
        "label_semantics": {
            "SLC": "slice index; Pulseq SET once immediately before each slice first RF",
            "LIN": "trajectory/view index for T13/T14; local arm index for T15",
            "REP": "absent",
            "temporal_frame": "reconstruction-only, 7 consecutive arms by default",
        },
        "actual_te_ms": result.actual_te_s * 1e3,
        "actual_tr_ms": result.actual_tr_s * 1e3,
        "total_adc": metadata["total_played_arms"],
        "scan_time_s": metadata["actual_total_scan_time_s"],
        "seq_check_timing": result.timing_ok,
        "max_physical_axis_gradient_mT_per_m": metadata["maximum_played_axis_gradient_mT_per_m"],
        "sequence_signature": result.signature,
        "seq_path": str(result.sequence_path.resolve()),
        "seq_size_mb": result.sequence_path.stat().st_size / (1024 * 1024),
        "trajectory_path": str(result.trajectory_path.resolve()),
    }
    (output_root / "DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
