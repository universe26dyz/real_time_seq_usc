#!/usr/bin/env python3
"""
3层重叠层几何 + 短时间序列测试：3 slices、2 mm shift、5 frames/slice、7 arms/frame、FOV-z=10 mm（真实 physical support）。

诊断序列生成器。
建议放置到：
  /home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_generators/

它直接复用当前 production generator：
  src/write_rtspiral_svr_uih.py

仅在本 Python 进程中临时改变诊断参数，不修改 production 源文件。

保持不变：
- trueFISP
- FA = 100 deg
- spiral readout = 2.5 ms
- ADC dwell = 2 us
- golden-angle = 222.4969 deg
- M1-nulled gropt rewinder
- UIH dead times
- max_grad / max_slew
- TE/TR 生成逻辑
- 无 trigger

本测试有意改变：
- num_slices
- frames_per_slice
- arms_per_frame
- UIH FOV Definition 的第三维 FOV-z
- slice_shift_mm（本组均为 2 mm；保留为显式测试参数）
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

TEST_ID = "T09_3slice_overlap_fovz10_5frames"
PURPOSE = "3层重叠层几何 + 短时间序列测试：3 slices、2 mm shift、5 frames/slice、7 arms/frame、FOV-z=10 mm（真实 physical support）。"

NUM_SLICES = 3
SLICE_SHIFT_MM = 2.0
ARMS_PER_FRAME = 7
FRAMES_PER_SLICE = 5
TRS_PER_SLICE = ARMS_PER_FRAME * FRAMES_PER_SLICE
FOV_Z_MM = 10.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_loader import load_config
import write_rtspiral_svr_uih as generator


def diagnostic_constraints(config: dict) -> None:
    """
    保留 production generator 的关键平台和序列物理约束，
    只允许本诊断脚本改变 arms/frame、frames/slice、slice 数量和几何测试参数。
    """
    scanner = config["scanner_system"]
    expected_scanner = {
        "max_grad_mT_per_m": 15.0,
        "max_slew_T_per_m_per_s": 32.0,
        "grad_raster_time_s": 10e-6,
        "rf_raster_time_s": 1e-6,
        "rf_ringdown_time_s": 10e-6,
        "rf_dead_time_s": 400e-6,
        "adc_dead_time_s": 70e-6,
    }

    for key, expected in expected_scanner.items():
        if not np.isclose(scanner[key], expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"scanner_system.{key}={scanner[key]} but expected {expected}"
            )

    spiral = config["spiral"]
    if spiral["contrast"] != "trueFISP":
        raise ValueError("contrast must remain trueFISP")
    if spiral["arm_ordering"] != "ga":
        raise ValueError("arm_ordering must remain ga")
    if spiral["rewinder_method"] != "gropt":
        raise ValueError("rewinder_method must remain gropt")
    if not spiral["m1_nulling"]:
        raise ValueError("M1 nulling must remain enabled")
    if not spiral["rotate_grads"]:
        raise ValueError("rotate_grads must remain enabled")
    if not np.isclose(spiral["rewinder_time_s"], 3e-3, rtol=0.0, atol=1e-12):
        raise ValueError("requested rewinder ceiling must remain 3 ms")

    if config["rf"]["flip_angle_deg"] != 100.0:
        raise ValueError("flip angle must remain 100 deg")

    if (
        config["triggers"]["physio_trigger_enabled"]
        or config["triggers"]["external_ttl_enabled"]
    ):
        raise ValueError("all triggers must remain disabled")

    realtime = config["realtime"]
    if realtime["arms_per_frame"] != ARMS_PER_FRAME:
        raise ValueError(
            f"arms_per_frame={realtime['arms_per_frame']} "
            f"but expected {ARMS_PER_FRAME}"
        )
    if realtime["frames_per_slice"] != FRAMES_PER_SLICE:
        raise ValueError(
            f"frames_per_slice={realtime['frames_per_slice']} "
            f"but expected {FRAMES_PER_SLICE}"
        )
    if realtime["trs_per_slice"] != TRS_PER_SLICE:
        raise ValueError(
            f"trs_per_slice={realtime['trs_per_slice']} "
            f"but expected {TRS_PER_SLICE}"
        )

    geometry = config["geometry"]
    if geometry["num_slices"] != NUM_SLICES:
        raise ValueError("num_slices mismatch")
    if not np.isclose(geometry["slice_shift_mm"], SLICE_SHIFT_MM):
        raise ValueError("slice_shift_mm mismatch")


def diagnostic_apply_uih_definitions(
    seq,
    config: dict,
    actual_te_s: float,
    actual_tr_s: float,
    positions_m: list[float],
) -> None:
    """
    与正式 uih_definitions.py 保持一致，
    唯一有意暴露给 A/B test 的参数是 FOV 的第三维 FOV_Z_MM。
    """
    if not hasattr(seq, "set_definition"):
        raise TypeError("seq must expose set_definition")

    geometry = config["geometry"]
    fov_mm = geometry["fov_mm"]
    matrix = geometry["matrix"]
    thickness_m = geometry["slice_thickness_mm"] * 1e-3

    definitions = {
        "Dimension": 2,
        "FOV": [
            fov_mm[0] * 1e-3,
            fov_mm[1] * 1e-3,
            FOV_Z_MM * 1e-3,
        ],
        "SliceNumber": geometry["num_slices"],
        "SliceThickness": thickness_m,
        "Matrix": matrix,
        "Resolution": matrix,
        "Center": [matrix[0] / 2.0, matrix[1] / 2.0],
        "SlicePositions": positions_m,
        "Name": config["project"]["name"],
        "TE": actual_te_s,
        "TR": actual_tr_s,
        "FA": config["rf"]["flip_angle_deg"],
    }

    for key, value in definitions.items():
        seq.set_definition(key, value)


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json"
    config = copy.deepcopy(load_config(str(config_path)))

    # ------------------------------------------------------------------
    # Diagnostic overrides
    # ------------------------------------------------------------------
    config["project"]["name"] = TEST_ID

    config["geometry"]["num_slices"] = NUM_SLICES
    config["geometry"]["slice_shift_mm"] = SLICE_SHIFT_MM

    config["realtime"]["arms_per_frame"] = ARMS_PER_FRAME
    config["realtime"]["frames_per_slice"] = FRAMES_PER_SLICE
    config["realtime"]["trs_per_slice"] = TRS_PER_SLICE

    config["realtime"]["expected_frame_time_ms_at_target_tr"] = (
        ARMS_PER_FRAME * config["target_timing"]["tr_ms"]
    )
    config["realtime"]["expected_slice_time_s_at_target_tr"] = (
        TRS_PER_SLICE * config["target_timing"]["tr_ms"] * 1e-3
    )

    root = f"diagnostic_sequences/{TEST_ID}"
    config["output"]["seq_dir"] = f"{root}/out_seq"
    config["output"]["trajectory_dir"] = f"{root}/out_trajectory"
    config["output"]["report_dir"] = f"{root}/reports"

    # ------------------------------------------------------------------
    # Patch only in this Python process.
    # The production source files on disk remain untouched.
    # ------------------------------------------------------------------
    generator._require_generator_constraints = diagnostic_constraints
    generator.apply_uih_definitions = diagnostic_apply_uih_definitions

    physical_support_mm = (
        (NUM_SLICES - 1) * SLICE_SHIFT_MM
        + config["geometry"]["slice_thickness_mm"]
    )

    print("=" * 78)
    print(f"Diagnostic sequence: {TEST_ID}")
    print("=" * 78)
    print(f"Purpose              : {PURPOSE}")
    print(f"Num slices           : {NUM_SLICES}")
    print(f"Slice thickness      : {config['geometry']['slice_thickness_mm']:.3f} mm")
    print(f"Slice centre shift   : {SLICE_SHIFT_MM:.3f} mm")
    print(f"Physical support     : {physical_support_mm:.3f} mm")
    print(f"UIH FOV-z Definition : {FOV_Z_MM:.3f} mm")
    print(f"Arms/frame           : {ARMS_PER_FRAME}")
    print(f"Frames/slice         : {FRAMES_PER_SLICE}")
    print(f"TRs/slice            : {TRS_PER_SLICE}")
    print()

    result = generator.build_sequence(config, PROJECT_ROOT)

    total_adc = NUM_SLICES * TRS_PER_SLICE
    seq_size_mb = result.sequence_path.stat().st_size / (1024 ** 2)
    scan_time_s = result.actual_tr_s * total_adc
    frame_time_ms = result.actual_tr_s * ARMS_PER_FRAME * 1e3
    slice_time_s = result.actual_tr_s * TRS_PER_SLICE

    report_dir = PROJECT_ROOT / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "DIAGNOSTIC_SUMMARY.json"

    summary = {
        "test_id": TEST_ID,
        "purpose": PURPOSE,
        "base_config": str(config_path),
        "diagnostic_overrides": {
            "num_slices": NUM_SLICES,
            "slice_shift_mm": SLICE_SHIFT_MM,
            "arms_per_frame": ARMS_PER_FRAME,
            "frames_per_slice": FRAMES_PER_SLICE,
            "trs_per_slice": TRS_PER_SLICE,
            "uih_fov_z_mm": FOV_Z_MM,
        },
        "preserved_sequence_physics": {
            "contrast": config["spiral"]["contrast"],
            "spiral_readout_ms": config["spiral"]["readout_duration_s"] * 1e3,
            "adc_dwell_us": config["spiral"]["adc_dwell_s"] * 1e6,
            "ga_angle_deg": config["spiral"]["ga_angle_deg"],
            "flip_angle_deg": config["rf"]["flip_angle_deg"],
            "slice_thickness_mm": config["geometry"]["slice_thickness_mm"],
            "max_grad_mT_per_m": config["scanner_system"]["max_grad_mT_per_m"],
            "max_slew_T_per_m_per_s": config["scanner_system"]["max_slew_T_per_m_per_s"],
            "physio_trigger_enabled": config["triggers"]["physio_trigger_enabled"],
            "external_ttl_enabled": config["triggers"]["external_ttl_enabled"],
        },
        "geometry": {
            "physical_support_mm": physical_support_mm,
            "fov_xy_mm": config["geometry"]["fov_mm"],
            "fov_z_mm": FOV_Z_MM,
        },
        "resolved": {
            "actual_te_ms": result.actual_te_s * 1e3,
            "actual_tr_ms": result.actual_tr_s * 1e3,
            "frame_time_ms": frame_time_ms,
            "slice_time_s": slice_time_s,
            "total_adc_acquisitions": total_adc,
            "scan_time_s": scan_time_s,
            "sequence_signature": result.signature,
            "timing_ok": result.timing_ok,
            "seq_path": str(result.sequence_path),
            "seq_size_mb": seq_size_mb,
            "trajectory_path": str(result.trajectory_path),
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("Generated")
    print("=" * 78)
    print(f"SEQ        : {result.sequence_path}")
    print(f"SEQ size   : {seq_size_mb:.3f} MB")
    print(f"Trajectory : {result.trajectory_path}")
    print(f"Signature  : {result.signature}")
    print(
        f"TE/TR      : "
        f"{result.actual_te_s*1e3:.3f} / {result.actual_tr_s*1e3:.3f} ms"
    )
    print(f"Frame time : {frame_time_ms:.3f} ms")
    print(f"Slice time : {slice_time_s:.3f} s")
    print(f"ADC count  : {total_adc}")
    print(f"Scan time  : {scan_time_s:.3f} s")
    print(f"Timing OK  : {result.timing_ok}")
    print(f"Summary    : {summary_path}")


if __name__ == "__main__":
    main()
