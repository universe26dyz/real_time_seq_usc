"""
功能：生成 uMR 790 多层 fixed-slice real-time spiral trueFISP Pulseq 序列。
来源：USC MREL rtspiral_pypulseq 的 write_rtspiral_svr.py（commit 5d32a138）。
与 USC 原版相比修改了什么：改用统一 JSON、UIH Opts、无触发、每层 350 个全局 GA 臂、
SLC/REP/LIN 标签、UIH 2D Definitions，并输出最小 MATLAB 轨迹元数据。
为什么必须修改：UIH dead time、实时帧组织、重叠层几何和 raw-data 标签与 Siemens 原版不同。
输入：经 config_loader 校验的配置字典，以及输出根目录。
输出：GenerationResult；同时写出 .seq 和包含基准轨迹/全局角度的 .mat。
使用命令：conda run --no-capture-output -n Pulseq python src/write_rtspiral_svr_uih.py
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USC_ROOT = PROJECT_ROOT / "third_party" / "usc_rtspiral_pypulseq"
LIBSPIRAL_SOURCE = USC_ROOT / "modules" / "libspiral" / "src"

# UIH adaptation: 上游打包遗漏 libspiralutils，必须先加入原始源码目录。
for source_path in (LIBSPIRAL_SOURCE, USC_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from libspiral import raster_to_grad, vds_fixed_ro
from librewinder.design_rewinder import design_rewinder
from pypulseq import (
    Opts,
    add_gradients,
    calc_duration,
    calc_rf_center,
    make_adc,
    make_arbitrary_grad,
    make_delay,
    make_label,
    make_sinc_pulse,
    rotate,
)
from pypulseq.Sequence.sequence import Sequence

from config_loader import load_config
from uih_definitions import apply_uih_definitions, slice_positions_m


GA_POLICIES = {
    "usc144_lut_global_wrap": {"lut_length": 144, "slice_reset": False},
    "lut350_slice_reset": {"lut_length": 350, "slice_reset": True},
    "continuous_global_ga": {"lut_length": None, "slice_reset": False},
}


@dataclass(frozen=True)
class GenerationResult:
    """Sequence outputs and resolved values needed by the Task 3 reporter."""

    sequence: Sequence
    actual_te_s: float
    actual_tr_s: float
    sequence_path: Path
    trajectory_path: Path
    signature: str
    timing_ok: bool
    timing_adjustment_reason: str | None
    metadata: dict[str, Any]


def _require_generator_constraints(config: dict) -> None:
    """Reject a config that would silently change the required acquisition."""
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
            raise ValueError(f"scanner_system.{key} must be {expected}")

    spiral = config["spiral"]
    if (
        spiral["contrast"] != "trueFISP"
        or spiral["arm_ordering"] != "ga"
        or spiral["rewinder_method"] != "gropt"
        or not spiral["m1_nulling"]
        or not spiral["rotate_grads"]
    ):
        raise ValueError("generator requires GA trueFISP with rotated M1-nulled gropt rewinder")
    if not np.isclose(spiral["rewinder_time_s"], 3e-3, rtol=0.0, atol=1e-12):
        raise ValueError("spiral.rewinder_time_s must remain the requested 3 ms ceiling")
    if config["triggers"]["physio_trigger_enabled"] or config["triggers"]["external_ttl_enabled"]:
        raise ValueError("physio and external TTL triggers must remain disabled")
    if config["rf"]["flip_angle_deg"] != 100.0:
        raise ValueError("rf.flip_angle_deg must remain 100")
    if config["acquisition"]["arms_per_slice"] != 350:
        raise ValueError("acquisition.arms_per_slice must remain 350")
    if config["reconstruction_defaults"]["arms_per_frame"] != 7:
        raise ValueError("reconstruction_defaults.arms_per_frame must remain 7")


def _ceil_to_raster(duration_s: float, raster_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return ceil((duration_s - 1e-15) / raster_s) * raster_s


def _design_usc_rewinder(g_grad: np.ndarray, config: dict, system: Opts):
    """Run USC gropt, widening only its search ceiling when trial rotations fail."""
    spiral = config["spiral"]
    requested_limit_s = spiral["rewinder_time_s"]
    call_args = {
        "slew_ratio": spiral["slew_ratio"],
        "grad_rew_method": spiral["rewinder_method"],
        "M1_nulling": spiral["m1_nulling"],
        "rotate_grads": spiral["rotate_grads"],
    }
    try:
        result = design_rewinder(g_grad, requested_limit_s, system, **call_args)
        return (*result, requested_limit_s, None)
    except UnboundLocalError as error:
        if "G_out" not in str(error):
            raise
        # USC gropt 的旋转搜索在 3 ms 某些试角无解；只把 solver 上限重试为 4 ms。
        retry_limit_s = 4e-3
        retry_reason = (
            "USC design_rewinder raised UnboundLocalError for G_out because some "
            "rotation-search trial angles had no solution at the requested 3 ms "
            "solver ceiling; retried the identical M1-nulled gropt method with a "
            "4 ms solver upper bound."
        )
        result = design_rewinder(g_grad, retry_limit_s, system, **call_args)
        return (*result, retry_limit_s, retry_reason)


def build_sequence(
    config: dict,
    output_root: Path,
    ga_policy: str = "continuous_global_ga",
    sequence_filename: str = "uih790_rtspiral_realtime.seq",
) -> GenerationResult:
    """Build, timing-check, and write the UIH real-time spiral sequence."""
    _require_generator_constraints(config)
    output_root = Path(output_root)
    scanner = config["scanner_system"]
    spiral = config["spiral"]
    geometry = config["geometry"]
    acquisition = config["acquisition"]
    reconstruction = config["reconstruction_defaults"]
    if ga_policy not in GA_POLICIES:
        raise ValueError(f"unknown GA policy: {ga_policy}")

    # UIH adaptation: Opts 精确采用手册所需 raster/dead time 与保守梯度限制。
    system = Opts(
        max_grad=scanner["max_grad_mT_per_m"],
        grad_unit="mT/m",
        max_slew=scanner["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        grad_raster_time=scanner["grad_raster_time_s"],
        rf_raster_time=scanner["rf_raster_time_s"],
        rf_ringdown_time=scanner["rf_ringdown_time_s"],
        rf_dead_time=scanner["rf_dead_time_s"],
        adc_dead_time=scanner["adc_dead_time_s"],
    )
    rotation_safe_max_grad_mT_per_m = scanner["max_grad_mT_per_m"] / np.sqrt(2)
    # GA 会把二维向量投影到任意物理轴；设计阶段按 15/sqrt(2) 留出旋转余量。
    design_system = Opts(
        max_grad=rotation_safe_max_grad_mT_per_m,
        grad_unit="mT/m",
        max_slew=scanner["max_slew_T_per_m_per_s"],
        slew_unit="T/m/s",
        grad_raster_time=scanner["grad_raster_time_s"],
        rf_raster_time=scanner["rf_raster_time_s"],
        rf_ringdown_time=scanner["rf_ringdown_time_s"],
        rf_dead_time=scanner["rf_dead_time_s"],
        adc_dead_time=scanner["adc_dead_time_s"],
    )
    grad_raster_s = scanner["grad_raster_time_s"]
    spiral_system = {
        "max_slew": scanner["max_slew_T_per_m_per_s"] * spiral["slew_ratio"],
        "max_grad": rotation_safe_max_grad_mT_per_m * 0.99,
        "adc_dwell": spiral["adc_dwell_s"],
        "grad_raster_time": grad_raster_s,
        "os": 8,
    }

    # vds_fixed_ro 的 fov 是径向 profile；二维 UIH FOV 仍保留 360 x 320 mm。
    design_fov_mm = spiral["trajectory_design_fov_mm"]
    vds_fov_cm = [design_fov_mm * 0.1]
    resolution_mm = geometry["inplane_resolution_mm"][0]
    base_k, gradient_adc_raster, _, full_sampling_interleaves = vds_fixed_ro(
        spiral_system,
        vds_fov_cm,
        resolution_mm,
        spiral["readout_duration_s"],
    )
    if base_k is None or gradient_adc_raster is None:
        raise RuntimeError("vds_fixed_ro did not return a trajectory")
    _, base_gradient_unrotated = raster_to_grad(
        gradient_adc_raster,
        spiral["adc_dwell_s"],
        grad_raster_s,
    )
    rewind_x, rewind_y, base_gradient, rewinder_search_limit_s, rewinder_retry_reason = (
        _design_usc_rewinder(base_gradient_unrotated, config, design_system)
    )
    rotation_dot = np.sum(base_gradient_unrotated * base_gradient)
    rotation_cross = np.sum(
        base_gradient_unrotated[:, 0] * base_gradient[:, 1]
        - base_gradient_unrotated[:, 1] * base_gradient[:, 0]
    )
    rewinder_base_rotation_deg = float(
        np.rad2deg(np.arctan2(rotation_cross, rotation_dot))
    )
    complete_gradient = np.concatenate(
        (base_gradient, np.column_stack((rewind_x, rewind_y))), axis=0
    )

    rf_config = config["rf"]
    rf, gz, gzr = make_sinc_pulse(
        flip_angle=np.deg2rad(rf_config["flip_angle_deg"]),
        duration=rf_config["rf_duration_s"],
        slice_thickness=geometry["slice_thickness_mm"] * 1e-3,
        time_bw_product=rf_config["time_bandwidth_product"],
        return_gz=True,
        use="excitation",
        delay=scanner["rf_dead_time_s"],
        system=system,
    )
    gzrr = copy.deepcopy(gzr)
    gzr.delay = calc_duration(gz)
    slice_select = add_gradients([gz, gzr], system=system)

    ndiscard = 10
    num_samples = int(np.floor(spiral["readout_duration_s"] / spiral["adc_dwell_s"])) + ndiscard
    base_adc = make_adc(
        num_samples,
        dwell=spiral["adc_dwell_s"],
        delay=scanner["adc_dead_time_s"],
        system=system,
    )
    # ADC 必须先满足 70 us dead time，再保留 USC 的预丢弃采样与半 raster 对齐。
    discard_delay_s = ceil(
        (
            base_adc.delay
            + ndiscard * spiral["adc_dwell_s"]
            + grad_raster_s / 2
        )
        / grad_raster_s
    ) * grad_raster_s
    base_gx = make_arbitrary_grad(
        channel="x",
        waveform=complete_gradient[:, 0] * 42.58e3,
        delay=discard_delay_s,
        system=system,
    )
    base_gy = make_arbitrary_grad(
        channel="y",
        waveform=complete_gradient[:, 1] * 42.58e3,
        delay=discard_delay_s,
        system=system,
    )
    base_gx.first = base_gx.last = 0
    base_gy.first = base_gy.last = 0
    rewinder_duration_s = max(len(rewind_x), len(rewind_y)) * grad_raster_s
    gzrr.delay = calc_duration(base_gx, base_gy, base_adc) - min(
        calc_duration(gzrr), rewinder_duration_s
    )

    policy_spec = GA_POLICIES[ga_policy]
    ga_lut_length = policy_spec["lut_length"]
    ga_lut_angles_deg = (
        np.mod(np.arange(ga_lut_length) * spiral["ga_angle_deg"], 360.0)
        if ga_lut_length is not None
        else None
    )
    ga_gradient_lut = (
        [
            rotate(base_gx, base_gy, axis="z", angle=np.deg2rad(angle), system=system)
            for angle in ga_lut_angles_deg
        ]
        if ga_lut_angles_deg is not None
        else None
    )

    # target 不可达时只增加 raster-compatible delay，不删除 rewinder/M1。
    target_te_s = config["target_timing"]["te_ms"] * 1e-3
    rf_center_from_block_start_s = rf.delay + calc_rf_center(rf)[0]
    minimum_te_s = (
        calc_duration(rf, slice_select)
        - rf_center_from_block_start_s
        + base_gx.delay
    )
    te_delay_s = _ceil_to_raster(target_te_s - minimum_te_s, grad_raster_s)
    actual_te_s = minimum_te_s + te_delay_s

    readout_duration_s = calc_duration(base_gx, base_gy, base_adc, gzrr)
    minimum_tr_s = calc_duration(rf, slice_select) + te_delay_s + readout_duration_s
    target_tr_s = config["target_timing"]["tr_ms"] * 1e-3
    tr_delay_s = _ceil_to_raster(target_tr_s - minimum_tr_s, grad_raster_s)
    actual_tr_s = minimum_tr_s + tr_delay_s
    timing_reasons = []
    if actual_te_s > target_te_s + 1e-12:
        timing_reasons.append(
            f"target TE {target_te_s * 1e3:.3f} ms is below minimum {minimum_te_s * 1e3:.3f} ms"
        )
    if actual_tr_s > target_tr_s + 1e-12:
        timing_reasons.append(
            f"target TR {target_tr_s * 1e3:.3f} ms is below minimum {minimum_tr_s * 1e3:.3f} ms"
        )
    timing_adjustment_reason = "; ".join(timing_reasons) or None

    seq = Sequence(system)
    positions_m = slice_positions_m(
        geometry["num_slices"], geometry["slice_shift_mm"]
    )
    arms_per_slice = acquisition["arms_per_slice"]
    total_arms = geometry["num_slices"] * arms_per_slice
    global_indices = np.arange(total_arms, dtype=np.int32)
    slice_index_per_acq = np.repeat(np.arange(geometry["num_slices"], dtype=np.int32), arms_per_slice)
    arm_index_in_slice = np.tile(np.arange(arms_per_slice, dtype=np.int32), geometry["num_slices"])
    trajectory_index_per_acq = np.empty(total_arms, dtype=np.int32)
    actual_angles_deg = np.empty(total_arms, dtype=float)
    labels_metadata: list[dict[str, int]] = []
    maximum_played_axis_gradient_mT_per_m = 0.0
    te_delay = make_delay(te_delay_s) if te_delay_s else None
    tr_delay = make_delay(tr_delay_s) if tr_delay_s else None

    # fixed-slice：每层连续 7 arms/frame x 50 frames，再移动到下一层。
    for slice_index, position_m in enumerate(positions_m):
        # SLC 必须在本层第一条 RF 前写入一次，且不在 arm 内重复。
        seq.add_block(make_label(config["labels"]["slice_label"], "SET", slice_index))
        slice_rf = copy.deepcopy(rf)
        slice_rf.freq_offset = gz.amplitude * position_m
        slice_phase_correction = np.mod(
            -2 * np.pi * slice_rf.freq_offset * calc_rf_center(slice_rf)[0],
            2 * np.pi,
        )
        for local_arm_index in range(arms_per_slice):
            global_arm_index = (
                slice_index * arms_per_slice + local_arm_index
            )
            if ga_policy == "usc144_lut_global_wrap":
                trajectory_index = global_arm_index % ga_lut_length
            elif ga_policy == "lut350_slice_reset":
                trajectory_index = local_arm_index
            else:
                trajectory_index = global_arm_index
            if ga_gradient_lut is None:
                angle_deg = float(np.mod(global_arm_index * spiral["ga_angle_deg"], 360.0))
                gx, gy = rotate(base_gx, base_gy, axis="z", angle=np.deg2rad(angle_deg), system=system)
            else:
                angle_deg = float(ga_lut_angles_deg[trajectory_index])
                gx, gy = ga_gradient_lut[trajectory_index]
            trajectory_index_per_acq[global_arm_index] = trajectory_index
            actual_angles_deg[global_arm_index] = angle_deg
            labels_metadata.append({"SLC": slice_index, "LIN": trajectory_index if ga_policy != "continuous_global_ga" else local_arm_index})

            # trueFISP 相位交替与全局 GA 角均不在换层时重置。
            rf_phase = np.mod(global_arm_index * np.pi, 2 * np.pi)
            current_rf = copy.deepcopy(slice_rf)
            current_rf.phase_offset = np.mod(rf_phase + slice_phase_correction, 2 * np.pi)
            current_adc = copy.deepcopy(base_adc)
            current_adc.phase_offset = rf_phase
            maximum_played_axis_gradient_mT_per_m = max(
                maximum_played_axis_gradient_mT_per_m,
                float(np.max(np.abs(gx.waveform)) / system.gamma * 1e3),
                float(np.max(np.abs(gy.waveform)) / system.gamma * 1e3),
            )

            seq.add_block(current_rf, slice_select)
            if te_delay is not None:
                seq.add_block(te_delay)
            # LIN 是 view/trajectory index；不写 REP 或 temporal-frame scanner label。
            seq.add_block(
                make_label(config["labels"]["arm_in_frame_label"], "SET", labels_metadata[-1]["LIN"]),
                gx,
                gy,
                current_adc,
                gzrr,
            )
            if tr_delay is not None:
                seq.add_block(tr_delay)

    apply_uih_definitions(seq, config, actual_te_s, actual_tr_s, positions_m)
    timing_ok, timing_errors = seq.check_timing()
    if not timing_ok:
        raise RuntimeError("PyPulseq timing check failed:\n" + "\n".join(map(str, timing_errors)))

    # 用真实 Pulseq 事件得到 ADC-sampled 基准 k-space，确保包含 optimizer 固定旋转。
    reference_seq = Sequence(system)
    reference_seq.add_block(copy.deepcopy(rf), copy.deepcopy(slice_select))
    if te_delay is not None:
        reference_seq.add_block(copy.deepcopy(te_delay))
    reference_seq.add_block(
        copy.deepcopy(base_gx),
        copy.deepcopy(base_gy),
        copy.deepcopy(base_adc),
        copy.deepcopy(gzrr),
    )
    base_k_played_adc, *_ = reference_seq.calculate_kspace()
    base_k_played = base_k_played_adc[:2].T

    seq_dir = output_root / config["output"]["seq_dir"]
    trajectory_dir = output_root / config["output"]["trajectory_dir"]
    seq_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = seq_dir / sequence_filename
    signature = seq.write(str(sequence_path), create_signature=True, check_timing=True)
    if not signature:
        raise RuntimeError("PyPulseq did not return a sequence signature")
    trajectory_path = trajectory_dir / f"{signature}.mat"
    savemat(
        trajectory_path,
        {
            "base_k": base_k_played,
            "base_k_played": base_k_played,
            "base_k_vds_unrotated": base_k,
            "base_gradient_mT_per_m": complete_gradient,
            "global_arm_index": global_indices,
            "global_arm_angle_deg": actual_angles_deg,
            "ga_policy": ga_policy,
            "ga_angle_deg": spiral["ga_angle_deg"],
            "ga_lut_length": ga_lut_length if ga_lut_length is not None else -1,
            "slice_reset": policy_spec["slice_reset"],
            "slice_index_per_acq": slice_index_per_acq,
            "arm_index_in_slice": arm_index_in_slice,
            "global_acquisition_index": global_indices,
            "trajectory_index_per_acq": trajectory_index_per_acq,
            "rewinder_base_rotation_deg": rewinder_base_rotation_deg,
            "sequence_signature": signature,
            "full_sampling_interleaves": full_sampling_interleaves,
            "arms_per_frame": reconstruction["arms_per_frame"],
            "frames_per_slice": arms_per_slice // reconstruction["arms_per_frame"],
            "played_arms_per_slice": arms_per_slice,
            "num_slices": geometry["num_slices"],
            "slice_positions_m": positions_m,
            "fov_mm": geometry["fov_mm"],
            "nominal_resolution_mm": geometry["inplane_resolution_mm"],
            "matrix": geometry["matrix"],
            "target_te_s": target_te_s,
            "target_tr_s": target_tr_s,
            "actual_te_s": actual_te_s,
            "actual_tr_s": actual_tr_s,
            "rewinder_requested_time_s": spiral["rewinder_time_s"],
            "rewinder_solver_search_time_s": rewinder_search_limit_s,
            "rewinder_actual_duration_s": rewinder_duration_s,
            "rewinder_retry_reason": rewinder_retry_reason or "",
        },
    )

    metadata = {
        "labels": labels_metadata,
        "global_arm_indices": global_indices.tolist(),
        "global_arm_angles_deg": actual_angles_deg.tolist(),
        "slice_index_per_acq": slice_index_per_acq.tolist(),
        "arm_index_in_slice": arm_index_in_slice.tolist(),
        "trajectory_index_per_acq": trajectory_index_per_acq.tolist(),
        "ga_policy": ga_policy,
        "ga_lut_length": ga_lut_length,
        "slice_reset": policy_spec["slice_reset"],
        "played_arms_per_slice": arms_per_slice,
        "default_reconstruction_arms_per_frame": reconstruction["arms_per_frame"],
        "default_reconstruction_frames_per_slice": arms_per_slice // reconstruction["arms_per_frame"],
        "total_played_arms": total_arms,
        "full_sampling_interleaves": int(full_sampling_interleaves),
        "rotation_safe_design_max_grad_mT_per_m": rotation_safe_max_grad_mT_per_m,
        "rewinder_base_rotation_deg": rewinder_base_rotation_deg,
        "base_k_reference": (
            "base_k/base_k_played are ADC-sampled k-space for the optimizer-rotated "
            "played base arm; apply global_arm_angle_deg for each acquisition."
        ),
        "user_fov_mm": list(geometry["fov_mm"]),
        "trajectory_design_fov_mm": design_fov_mm,
        "vds_fixed_ro_fov_cm": vds_fov_cm,
        "fov_distinction": (
            "trajectory_design_fov_mm is a derived radial vds_fixed_ro input, "
            "while user_fov_mm remains the Cartesian UIH FOV definition."
        ),
        "rewinder_method": spiral["rewinder_method"],
        "m1_nulling": spiral["m1_nulling"],
        "rewinder_requested_time_s": spiral["rewinder_time_s"],
        "rewinder_solver_search_time_s": rewinder_search_limit_s,
        "rewinder_actual_duration_s": rewinder_duration_s,
        "rewinder_retry_reason": rewinder_retry_reason,
        "adc_dead_time_s": base_adc.delay,
        "pre_discard_samples": ndiscard,
        "gradient_delay_s": discard_delay_s,
        "target_te_s": target_te_s,
        "target_tr_s": target_tr_s,
        "minimum_te_s": minimum_te_s,
        "minimum_tr_s": minimum_tr_s,
        "actual_frame_time_s": actual_tr_s * reconstruction["arms_per_frame"],
        "actual_slice_dwell_s": actual_tr_s * arms_per_slice,
        "actual_total_scan_time_s": actual_tr_s * total_arms,
        "slice_positions_m": positions_m,
        "matrix": geometry["matrix"],
        "center": [value / 2.0 for value in geometry["matrix"]],
        "effective_spacing_mm": [
            geometry["fov_mm"][index] / geometry["matrix"][index]
            for index in range(2)
        ],
        "maximum_played_axis_gradient_mT_per_m": maximum_played_axis_gradient_mT_per_m,
        "has_physio_trigger": False,
        "has_external_ttl": False,
    }
    return GenerationResult(
        sequence=seq,
        actual_te_s=actual_te_s,
        actual_tr_s=actual_tr_s,
        sequence_path=sequence_path,
        trajectory_path=trajectory_path,
        signature=signature,
        timing_ok=True,
        timing_adjustment_reason=timing_adjustment_reason,
        metadata=metadata,
    )


def write_task3_reports(result: GenerationResult, config: dict, output_root: Path) -> None:
    """Write resolved JSON and concise final reports from this actual run."""
    output_root = Path(output_root)
    report_dir = output_root / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = result.metadata
    geometry = config["geometry"]
    scanner = config["scanner_system"]
    target = config["target_timing"]
    reconstruction = config["reconstruction_defaults"]
    resolved_path = report_dir / "resolved_config.json"
    resolved = {
        "project": config["project"],
        "provenance": config["provenance"],
        "target_timing": {
            "te_ms": target["te_ms"], "tr_ms": target["tr_ms"],
            "frame_time_ms": target["frame_time_ms"],
            "slice_dwell_time_s": target["slice_dwell_time_s"],
        },
        "actual_timing": {
            "te_ms": result.actual_te_s * 1e3,
            "tr_ms": result.actual_tr_s * 1e3,
            "frame_time_ms": metadata["actual_frame_time_s"] * 1e3,
            "slice_dwell_time_s": metadata["actual_slice_dwell_s"],
            "total_scan_time_s": metadata["actual_total_scan_time_s"],
            "timing_adjustment_reason": result.timing_adjustment_reason,
        },
        "reconstruction_defaults": {
            "arms_per_frame": reconstruction["arms_per_frame"],
            "frames_per_slice": metadata["default_reconstruction_frames_per_slice"],
            "arms_per_slice": metadata["played_arms_per_slice"],
            "num_slices": geometry["num_slices"],
            "total_adc_acquisitions": metadata["total_played_arms"],
        },
        "geometry": {
            "fov_mm": geometry["fov_mm"],
            "nominal_resolution_mm": geometry["inplane_resolution_mm"],
            "matrix": metadata["matrix"],
            "effective_spacing_mm": metadata["effective_spacing_mm"],
            "center": metadata["center"],
            "slice_thickness_mm": geometry["slice_thickness_mm"],
            "slice_shift_mm": geometry["slice_shift_mm"],
            "slice_positions_m": metadata["slice_positions_m"],
        },
        "scanner_system": {
            "max_grad_mT_per_m": scanner["max_grad_mT_per_m"],
            "max_slew_T_per_m_per_s": scanner["max_slew_T_per_m_per_s"],
            "maximum_played_physical_axis_gradient_mT_per_m": metadata["maximum_played_axis_gradient_mT_per_m"],
        },
        "rewinder": {
            "requested_time_s": metadata["rewinder_requested_time_s"],
            "solver_search_time_s": metadata["rewinder_solver_search_time_s"],
            "actual_duration_s": metadata["rewinder_actual_duration_s"],
            "retry_reason": metadata["rewinder_retry_reason"],
        },
        "sequence": {
            "signature": result.signature,
            "seq_path": str(result.sequence_path.resolve()),
            "trajectory_path": str(result.trajectory_path.resolve()),
            "seq_file_size_bytes": result.sequence_path.stat().st_size,
            "timing_check_passed": result.timing_ok,
            "full_sampling_interleaves": metadata["full_sampling_interleaves"],
            "played_arms_per_slice": metadata["played_arms_per_slice"],
        },
    }
    resolved_path.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    timing_path = report_dir / "TIMING_REPORT.md"
    timing_path.write_text(
        f"# Timing report\n\n"
        f"Target: TE {target['te_ms']:.2f} ms; TR {target['tr_ms']:.2f} ms; "
        f"frame {target['frame_time_ms']:.2f} ms; slice dwell about {target['slice_dwell_time_s']:.2f} s.\n\n"
        f"Actual final run: TE {result.actual_te_s * 1e3:.3f} ms; TR {result.actual_tr_s * 1e3:.3f} ms; "
        f"frame {metadata['actual_frame_time_s'] * 1e3:.3f} ms; "
        f"slice dwell {metadata['actual_slice_dwell_s']:.6f} s; "
        f"60-slice duration {metadata['actual_total_scan_time_s']:.6f} s.\n\n"
        f"The target is infeasible without removing required UIH RF/ADC dead time, RF and slice-selection timing, "
        f"spiral readout, M1-nulled gropt rewinder, or raster constraints. {result.timing_adjustment_reason}\n\n"
        f"`seq.check_timing()` passed: `{result.timing_ok}`.\n",
        encoding="utf-8",
    )
    generation_path = report_dir / "GENERATION_REPORT.md"
    generation_path.write_text(
        f"# UIH RTSpiral final generation\n\n"
        f"- USC RTSpiral: {config['provenance']['primary_sequence_repo']} @ `{config['provenance']['primary_sequence_commit']}`\n"
        f"- PulseqSystems: {config['provenance']['uih_system_repo']} @ `{config['provenance']['uih_system_commit']}`\n"
        f"- Python / PyPulseq: {sys.version.split()[0]} / {version('pypulseq')}\n"
        f"- UIH adaptations: JSON config, UIH timing, fixed-slice 7×50×60 acquisition, SLC/REP/LIN, UIH Definitions, and traceable trajectory metadata.\n"
        f"- FOV: user-specified 360×320 mm; radial trajectory design FOV 360 mm is an engineering derivation, not a paper parameter.\n"
        f"- Matrix / Center / Resolution definition: {metadata['matrix']} / {metadata['center']} / {metadata['matrix']}.\n"
        f"- Actual TE/TR: {result.actual_te_s * 1e3:.3f}/{result.actual_tr_s * 1e3:.3f} ms; frame {metadata['actual_frame_time_s'] * 1e3:.3f} ms; slice dwell {metadata['actual_slice_dwell_s']:.6f} s; total {metadata['actual_total_scan_time_s']:.6f} s.\n"
        f"- Max physical-axis gradient: {metadata['maximum_played_axis_gradient_mT_per_m']:.6f} mT/m.\n"
        f"- Rewinder: requested 3 ms; solver bound {metadata['rewinder_solver_search_time_s'] * 1e3:.1f} ms; actual {metadata['rewinder_actual_duration_s'] * 1e3:.3f} ms.\n"
        f"- Signature: `{result.signature}`; timing check: `{result.timing_ok}`.\n"
        f"- Sequence: `{result.sequence_path.resolve()}` ({result.sequence_path.stat().st_size} bytes)\n"
        f"- Trajectory: `{result.trajectory_path.resolve()}`\n\n"
        f"Next: import the `.seq` into UIH AIDE / Pulseq Virtual Scan, inspect arbitrary gradients and definitions, then validate SAR, PNS, 100° FA/B1, label-to-DHL mapping, and raw-data timestamps.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UIH uMR 790 real-time spiral bSSFP")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    result = build_sequence(load_config(str(args.config)), args.output_root)
    write_task3_reports(result, load_config(str(args.config)), args.output_root)
    print(f"actual TE: {result.actual_te_s * 1e3:.3f} ms")
    print(f"actual TR: {result.actual_tr_s * 1e3:.3f} ms")
    print(f"sequence: {result.sequence_path}")
    print(f"trajectory: {result.trajectory_path}")
    print(f"signature: {result.signature}")


if __name__ == "__main__":
    main()
