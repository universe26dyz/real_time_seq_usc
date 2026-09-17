#!/usr/bin/env python3
"""
非重叠多层对照：3层、层厚6 mm、层中心间距6 mm、FOV-z=18 mm。

诊断序列生成器。
放到：
  /home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_generators/

它复用已经验证的 production generator：
  src/write_rtspiral_svr_uih.py

仅在本进程中临时改变诊断参数，不修改 production 源文件。
geometry-only 测试如果每层只有1帧，只用于 UIH 几何/加载排查，不用于图像质量评价。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import numpy as np

TEST_ID = "T05_3slice_nonoverlap_control"
PURPOSE = "非重叠多层对照：3层、层厚6 mm、层中心间距6 mm、FOV-z=18 mm。"
NUM_SLICES = 3
SLICE_SHIFT_MM = 6.0
ARMS_PER_FRAME = 7
FRAMES_PER_SLICE = 1
TRS_PER_SLICE = ARMS_PER_FRAME * FRAMES_PER_SLICE
FOV_Z_MM = 18.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config_loader import load_config
import write_rtspiral_svr_uih as generator


def diagnostic_constraints(config: dict) -> None:
    scanner = config["scanner_system"]
    expected = {
        "max_grad_mT_per_m": 15.0,
        "max_slew_T_per_m_per_s": 32.0,
        "grad_raster_time_s": 10e-6,
        "rf_raster_time_s": 1e-6,
        "rf_ringdown_time_s": 10e-6,
        "rf_dead_time_s": 400e-6,
        "adc_dead_time_s": 70e-6,
    }
    for key, value in expected.items():
        if not np.isclose(scanner[key], value, rtol=0.0, atol=1e-12):
            raise ValueError(f"{key}: got {scanner[key]}, expected {value}")

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
    if not np.isclose(spiral["rewinder_time_s"], 3e-3, atol=1e-12):
        raise ValueError("requested rewinder ceiling must remain 3 ms")

    if config["rf"]["flip_angle_deg"] != 100.0:
        raise ValueError("flip angle must remain 100 deg")

    if (config["triggers"]["physio_trigger_enabled"]
            or config["triggers"]["external_ttl_enabled"]):
        raise ValueError("all triggers must remain disabled")

    rt = config["realtime"]
    if rt["arms_per_frame"] != ARMS_PER_FRAME:
        raise ValueError("arms_per_frame must remain 7")
    if rt["frames_per_slice"] != FRAMES_PER_SLICE:
        raise ValueError("frames_per_slice mismatch")
    if rt["trs_per_slice"] != TRS_PER_SLICE:
        raise ValueError("trs_per_slice mismatch")

    geom = config["geometry"]
    if geom["num_slices"] != NUM_SLICES:
        raise ValueError("num_slices mismatch")
    if not np.isclose(geom["slice_shift_mm"], SLICE_SHIFT_MM):
        raise ValueError("slice_shift_mm mismatch")


def diagnostic_apply_uih_definitions(
    seq,
    config: dict,
    actual_te_s: float,
    actual_tr_s: float,
    positions_m: list[float],
) -> None:
    geom = config["geometry"]
    fov_mm = geom["fov_mm"]
    matrix = geom["matrix"]
    thickness_m = geom["slice_thickness_mm"] * 1e-3

    definitions = {
        "Dimension": 2,
        "FOV": [
            fov_mm[0] * 1e-3,
            fov_mm[1] * 1e-3,
            FOV_Z_MM * 1e-3,
        ],
        "SliceNumber": geom["num_slices"],
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

    generator._require_generator_constraints = diagnostic_constraints
    generator.apply_uih_definitions = diagnostic_apply_uih_definitions

    physical_support_mm = (
        (NUM_SLICES - 1) * SLICE_SHIFT_MM
        + config["geometry"]["slice_thickness_mm"]
    )

    print("=" * 72)
    print(TEST_ID)
    print("=" * 72)
    print("Purpose            :", PURPOSE)
    print("Num slices         :", NUM_SLICES)
    print("Slice thickness mm :", config["geometry"]["slice_thickness_mm"])
    print("Slice shift mm     :", SLICE_SHIFT_MM)
    print("Physical support mm:", physical_support_mm)
    print("UIH FOV-z mm       :", FOV_Z_MM)
    print("Arms/frame         :", ARMS_PER_FRAME)
    print("Frames/slice       :", FRAMES_PER_SLICE)
    print("TRs/slice          :", TRS_PER_SLICE)

    result = generator.build_sequence(config, PROJECT_ROOT)

    total_adc = NUM_SLICES * TRS_PER_SLICE
    seq_size_mb = result.sequence_path.stat().st_size / (1024 ** 2)
    scan_time_s = result.actual_tr_s * total_adc
    frame_time_ms = result.actual_tr_s * ARMS_PER_FRAME * 1e3

    report_dir = PROJECT_ROOT / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "DIAGNOSTIC_SUMMARY.json"

    summary = {
        "test_id": TEST_ID,
        "purpose": PURPOSE,
        "num_slices": NUM_SLICES,
        "slice_thickness_mm": config["geometry"]["slice_thickness_mm"],
        "slice_shift_mm": SLICE_SHIFT_MM,
        "physical_support_mm": physical_support_mm,
        "uih_fov_z_mm": FOV_Z_MM,
        "arms_per_frame": ARMS_PER_FRAME,
        "frames_per_slice": FRAMES_PER_SLICE,
        "trs_per_slice": TRS_PER_SLICE,
        "actual_te_ms": result.actual_te_s * 1e3,
        "actual_tr_ms": result.actual_tr_s * 1e3,
        "frame_time_ms": frame_time_ms,
        "total_adc_acquisitions": total_adc,
        "scan_time_s": scan_time_s,
        "seq_size_mb": seq_size_mb,
        "sequence_signature": result.signature,
        "seq_path": str(result.sequence_path),
        "trajectory_path": str(result.trajectory_path),
        "timing_ok": result.timing_ok,
        "note": (
            "1-frame-per-slice diagnostic sequences are geometry/loading tests, "
            "not image-quality tests. T02 is the intended 50-frame reconstruction smoke test."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Generated:")
    print("  SEQ       :", result.sequence_path)
    print("  size MB   :", f"{seq_size_mb:.3f}")
    print("  trajectory:", result.trajectory_path)
    print("  signature :", result.signature)
    print("  TE/TR ms  :", f"{result.actual_te_s*1e3:.3f} / {result.actual_tr_s*1e3:.3f}")
    print("  frame ms  :", f"{frame_time_ms:.3f}")
    print("  ADC count :", total_adc)
    print("  scan s    :", f"{scan_time_s:.3f}")
    print("  timing OK :", result.timing_ok)
    print("  summary   :", summary_path)


if __name__ == "__main__":
    main()
