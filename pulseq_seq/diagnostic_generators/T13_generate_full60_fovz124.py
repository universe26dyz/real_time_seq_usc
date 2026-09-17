#!/usr/bin/env python3
"""
T13: Full 60-slice production RTSpiral sequence with ONLY FOV-z changed
from 6 mm to the physical stack support of 124 mm.

Original full production acquisition kept unchanged:
- 60 slices
- slice thickness 6 mm
- slice-centre shift 2 mm
- 50 frames/slice
- 7 arms/frame
- 350 TRs/slice
- 21,000 ADC acquisitions total
- trueFISP
- GA spiral, 222.4969 deg
- 100 deg flip angle
- 2.5 ms spiral readout
- 2 us ADC dwell
- M1-nulled gropt rewinder
- no trigger
- same TE/TR generation
- same SlicePositions (-59 ... +59 mm)
- same Matrix / Resolution / Center

ONLY intended change:
    FOV = [360 mm, 320 mm, 124 mm]
instead of:
    FOV = [360 mm, 320 mm, 6 mm]

Physical support:
    (60 - 1) * 2 + 6 = 124 mm

Recommended location:
    /home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_generators/

Run:
    cd /home/universe/SVR/real_time_seq_usc/pulseq_seq
    conda run --no-capture-output -n Pulseq \
      python diagnostic_generators/T13_generate_full60_fovz124.py
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

TEST_ID = "T13_full60_fovz124"

EXPECTED_NUM_SLICES = 60
EXPECTED_SLICE_SHIFT_MM = 2.0
EXPECTED_SLICE_THICKNESS_MM = 6.0
EXPECTED_ARMS_PER_FRAME = 7
EXPECTED_FRAMES_PER_SLICE = 50
EXPECTED_TRS_PER_SLICE = 350
EXPECTED_FOV_X_MM = 360.0
EXPECTED_FOV_Y_MM = 320.0
FOV_Z_MM = 124.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_loader import load_config
import write_rtspiral_svr_uih as generator


def verify_production_config(config: dict) -> None:
    """Abort if the base config differs from the expected full production setup."""
    g = config["geometry"]
    rt = config["realtime"]
    sp = config["spiral"]
    rf = config["rf"]
    tr = config["triggers"]

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    check(g["num_slices"] == 60, f"num_slices={g['num_slices']} != 60")
    check(np.isclose(g["slice_shift_mm"], 2.0), f"slice_shift_mm={g['slice_shift_mm']} != 2")
    check(np.isclose(g["slice_thickness_mm"], 6.0), f"slice_thickness_mm={g['slice_thickness_mm']} != 6")
    check(g["matrix"] == [240, 213], f"matrix={g['matrix']} != [240, 213]")
    check(np.allclose(g["fov_mm"], [360.0, 320.0]), f"fov_mm={g['fov_mm']} != [360,320]")

    check(rt["arms_per_frame"] == 7, f"arms_per_frame={rt['arms_per_frame']} != 7")
    check(rt["frames_per_slice"] == 50, f"frames_per_slice={rt['frames_per_slice']} != 50")
    check(rt["trs_per_slice"] == 350, f"trs_per_slice={rt['trs_per_slice']} != 350")

    check(sp["contrast"] == "trueFISP", f"contrast={sp['contrast']} != trueFISP")
    check(sp["arm_ordering"] == "ga", f"arm_ordering={sp['arm_ordering']} != ga")
    check(np.isclose(sp["ga_angle_deg"], 222.4969), f"GA={sp['ga_angle_deg']} != 222.4969")
    check(np.isclose(sp["readout_duration_s"], 2.5e-3), "readout duration changed")
    check(np.isclose(sp["adc_dwell_s"], 2e-6), "ADC dwell changed")
    check(sp["rewinder_method"] == "gropt", "rewinder method changed")
    check(sp["m1_nulling"] is True, "M1 nulling disabled")
    check(sp["rotate_grads"] is True, "rotate_grads disabled")

    check(np.isclose(rf["flip_angle_deg"], 100.0), "flip angle changed")
    check(tr["physio_trigger_enabled"] is False, "physio trigger enabled")
    check(tr["external_ttl_enabled"] is False, "external TTL enabled")

    if failures:
        raise RuntimeError(
            "T13 must differ from production ONLY in FOV-z. "
            "Base config mismatch:\n- " + "\n- ".join(failures)
        )


def t13_apply_uih_definitions(
    seq,
    config: dict,
    actual_te_s: float,
    actual_tr_s: float,
    positions_m: list[float],
) -> None:
    """Production UIH definitions, except FOV-z is changed to 124 mm."""
    g = config["geometry"]
    fov_mm = g["fov_mm"]
    matrix = g["matrix"]
    thickness_m = g["slice_thickness_mm"] * 1e-3

    definitions = {
        "Dimension": config["uih_definitions"]["dimension"],
        "FOV": [
            fov_mm[0] * 1e-3,
            fov_mm[1] * 1e-3,
            FOV_Z_MM * 1e-3,
        ],
        "SliceNumber": g["num_slices"],
        "SliceThickness": thickness_m,
        "Matrix": matrix,
        "Resolution": matrix,
        "Center": [matrix[0] / 2.0, matrix[1] / 2.0],
        "SlicePositions": positions_m,
        "Name": TEST_ID,
        "TE": actual_te_s,
        "TR": actual_tr_s,
        "FA": config["rf"]["flip_angle_deg"],
    }

    for key, value in definitions.items():
        seq.set_definition(key, value)


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json"
    config = copy.deepcopy(load_config(str(config_path)))

    verify_production_config(config)

    # Only rename outputs for this diagnostic test.
    config["project"]["name"] = TEST_ID

    root = f"diagnostic_sequences/{TEST_ID}"
    config["output"]["seq_dir"] = f"{root}/out_seq"
    config["output"]["trajectory_dir"] = f"{root}/out_trajectory"
    config["output"]["report_dir"] = f"{root}/reports"

    # ONLY monkey-patch the UIH definition writer in this process.
    generator.apply_uih_definitions = t13_apply_uih_definitions

    physical_support_mm = (
        (EXPECTED_NUM_SLICES - 1) * EXPECTED_SLICE_SHIFT_MM
        + EXPECTED_SLICE_THICKNESS_MM
    )

    if not math.isclose(physical_support_mm, FOV_Z_MM, abs_tol=1e-12):
        raise RuntimeError(
            f"physical support={physical_support_mm} mm != FOV_Z_MM={FOV_Z_MM} mm"
        )

    print("=" * 78)
    print("T13 full 60-slice FOV-z diagnostic")
    print("=" * 78)
    print(f"Base config            : {config_path}")
    print(f"Num slices             : {EXPECTED_NUM_SLICES}")
    print(f"Slice thickness        : {EXPECTED_SLICE_THICKNESS_MM:.3f} mm")
    print(f"Slice-centre shift     : {EXPECTED_SLICE_SHIFT_MM:.3f} mm")
    print(f"Physical stack support : {physical_support_mm:.3f} mm")
    print("Production FOV-z       : 6.000 mm")
    print(f"T13 FOV-z              : {FOV_Z_MM:.3f} mm")
    print(f"Arms/frame             : {EXPECTED_ARMS_PER_FRAME}")
    print(f"Frames/slice           : {EXPECTED_FRAMES_PER_SLICE}")
    print(f"TRs/slice              : {EXPECTED_TRS_PER_SLICE}")
    print(f"Total ADC              : {EXPECTED_NUM_SLICES * EXPECTED_TRS_PER_SLICE}")
    print()
    print("Only intended change: FOV-z 6 mm -> 124 mm.")
    print()

    result = generator.build_sequence(config, PROJECT_ROOT)

    total_adc = EXPECTED_NUM_SLICES * EXPECTED_TRS_PER_SLICE
    frame_time_ms = result.actual_tr_s * EXPECTED_ARMS_PER_FRAME * 1e3
    slice_time_s = result.actual_tr_s * EXPECTED_TRS_PER_SLICE
    total_scan_time_s = result.actual_tr_s * total_adc
    seq_size_mb = result.sequence_path.stat().st_size / (1024 ** 2)

    generated_fov = np.asarray(
        result.sequence.get_definition("FOV"),
        dtype=float,
    )
    expected_fov_m = np.array([0.360, 0.320, 0.124], dtype=float)

    if not np.allclose(generated_fov, expected_fov_m, atol=1e-12, rtol=0.0):
        raise RuntimeError(
            f"Generated FOV={generated_fov}, expected={expected_fov_m}"
        )

    if int(result.sequence.get_definition("SliceNumber")) != 60:
        raise RuntimeError("SliceNumber changed unexpectedly")

    if not math.isclose(
        float(result.sequence.get_definition("SliceThickness")),
        0.006,
        abs_tol=1e-12,
    ):
        raise RuntimeError("SliceThickness changed unexpectedly")

    report_dir = PROJECT_ROOT / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "T13_DIAGNOSTIC_SUMMARY.json"

    summary = {
        "test_id": TEST_ID,
        "purpose": (
            "Full production 60-slice sequence with only UIH FOV-z "
            "changed from 6 mm to physical support 124 mm."
        ),
        "only_intended_change": {
            "production_fov_z_mm": 6.0,
            "t13_fov_z_mm": 124.0,
        },
        "preserved_acquisition": {
            "num_slices": 60,
            "slice_thickness_mm": 6.0,
            "slice_shift_mm": 2.0,
            "physical_support_mm": 124.0,
            "arms_per_frame": 7,
            "frames_per_slice": 50,
            "trs_per_slice": 350,
            "total_adc_acquisitions": total_adc,
            "fov_xy_mm": [360.0, 320.0],
            "matrix": config["geometry"]["matrix"],
            "nominal_resolution_mm": config["geometry"]["inplane_resolution_mm"],
            "contrast": config["spiral"]["contrast"],
            "ga_angle_deg": config["spiral"]["ga_angle_deg"],
            "spiral_readout_ms": config["spiral"]["readout_duration_s"] * 1e3,
            "adc_dwell_us": config["spiral"]["adc_dwell_s"] * 1e6,
            "flip_angle_deg": config["rf"]["flip_angle_deg"],
        },
        "generated_definitions": {
            "FOV_m": generated_fov.tolist(),
            "SliceNumber": int(result.sequence.get_definition("SliceNumber")),
            "SliceThickness_m": float(result.sequence.get_definition("SliceThickness")),
            "Matrix": np.asarray(
                result.sequence.get_definition("Matrix")
            ).ravel().tolist(),
            "Resolution": np.asarray(
                result.sequence.get_definition("Resolution")
            ).ravel().tolist(),
            "Center": np.asarray(
                result.sequence.get_definition("Center")
            ).ravel().tolist(),
        },
        "resolved": {
            "actual_te_ms": result.actual_te_s * 1e3,
            "actual_tr_ms": result.actual_tr_s * 1e3,
            "frame_time_ms": frame_time_ms,
            "slice_time_s": slice_time_s,
            "total_scan_time_s": total_scan_time_s,
            "timing_ok": result.timing_ok,
            "sequence_signature": result.signature,
            "seq_path": str(result.sequence_path),
            "seq_size_mb": seq_size_mb,
            "trajectory_path": str(result.trajectory_path),
        },
        "scanner_test_question": (
            "Does changing only FOV-z from 6 mm to 124 mm remove "
            "the UIH warning: 'The stack is out of the linear gradient area'?"
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("T13 generated successfully")
    print("=" * 78)
    print(f"SEQ        : {result.sequence_path}")
    print(f"SEQ size   : {seq_size_mb:.3f} MB")
    print(f"Trajectory : {result.trajectory_path}")
    print(f"Signature  : {result.signature}")
    print("FOV        : [360, 320, 124] mm")
    print(
        f"TE / TR    : "
        f"{result.actual_te_s*1e3:.3f} / {result.actual_tr_s*1e3:.3f} ms"
    )
    print(f"Frame time : {frame_time_ms:.3f} ms")
    print(f"Slice time : {slice_time_s:.3f} s")
    print(f"Total scan : {total_scan_time_s:.3f} s")
    print(f"ADC count  : {total_adc}")
    print(f"Timing OK  : {result.timing_ok}")
    print(f"Summary    : {summary_path}")


if __name__ == "__main__":
    main()
