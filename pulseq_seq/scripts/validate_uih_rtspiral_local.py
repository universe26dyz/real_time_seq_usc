#!/usr/bin/env python3
"""
UIH RTSpiral 最终 .seq 的本地一致性验证脚本。

建议放置到：
    /home/universe/SVR/real_time_seq_usc/pulseq_seq/scripts/validate_uih_rtspiral_local.py

验证内容：
1. 最终 .seq 的 UIH Definitions
2. ADC 总数与 SLC/REP/LIN 组织
3. trueFISP RF/ADC phase cycling
4. 60 层 SlicePositions 是否中心对称、2 mm 等间隔
5. trajectory .mat 中 base arm / GA 旋转后的前 7 臂、前 350 臂可视化
6. trajectory metadata 与最终 .seq 前两条实际播放 arm 的方向一致性
7. 输出 CSV、PNG 和 LOCAL_VALIDATION_REPORT.md

说明：
- 本脚本直接读取“最终生成的 .seq”，不是重新 build sequence。
- 不替代 UIH AIDE / Virtual Scan、SAR、PNS 和真机 interpreter 检查。
- Pulseq .seq 为文本格式，写入/读回时会有有限小数位舍入。
  因此 phase / trajectory direction 检查使用严格但现实的数值容差，
  避免把 1e-5 度量级的序列化误差误判为真实物理错误。
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from pypulseq.Sequence.sequence import Sequence


# ============================================================================
# Numerical tolerances
# ============================================================================

# .seq 文本序列化后 phase 可能出现约 1e-6 ~ 1e-5 rad 的舍入误差。
# 1e-4 rad ≈ 0.00573 deg，仍然是非常严格的 trueFISP phase 检查。
PHASE_TOL_RAD = 1e-4

# trajectory endpoint direction 的 .seq 文本化误差通常约 1e-5 deg。
# 1e-3 deg 仍比实际观测误差大约严格一个数量级以上。
TRAJECTORY_DIRECTION_TOL_DEG = 1e-3

# Definition / slice-position 等确定性数值检查仍使用更严格容差。
DEFINITION_TOL = 1e-12


# ============================================================================
# Basic utilities
# ============================================================================

def wrap_phase(x: float | np.ndarray) -> float | np.ndarray:
    """将相位映射到 [0, 2π)。"""
    return np.mod(x, 2.0 * np.pi)


def circular_difference(a: float, b: float) -> float:
    """返回两个相位之间的最小圆周差，范围 [0, π]。"""
    return abs(np.angle(np.exp(1j * (a - b))))


def rotate_kspace(base_k: np.ndarray, angle_deg: float) -> np.ndarray:
    """将 [N, 2] k-space 轨迹逆时针旋转 angle_deg。"""
    theta = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    return base_k @ rotation.T


def normalize_direction(v: np.ndarray) -> np.ndarray:
    """将二维向量归一化为单位方向。"""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("zero-length direction vector")
    return v / norm


def as_flat_list(value: Any) -> list:
    """把 Definition 数组统一转成 Python flat list。"""
    return np.asarray(value).ravel().tolist()


# ============================================================================
# Read final Pulseq sequence / trajectory file
# ============================================================================

def load_final_sequence(seq_path: Path) -> Sequence:
    """直接读取最终写出的 .seq，不重新 build sequence。"""
    seq = Sequence()
    print(f"[INFO] Reading final .seq: {seq_path}")
    seq.read(str(seq_path))
    return seq


def find_trajectory_file(root: Path, explicit_path: Path | None) -> Path:
    """
    如果显式指定 trajectory，则严格使用指定文件；
    否则退化为 out_trajectory/ 中最新的 .mat。

    推荐正式验证时始终通过 --traj 显式指定与当前 sequence signature 对应的文件。
    """
    if explicit_path is not None:
        return explicit_path

    candidates = sorted(
        (root / "out_trajectory").glob("*.mat"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No .mat found in {root / 'out_trajectory'}")

    if len(candidates) > 1:
        print(
            f"[WARN] Found {len(candidates)} trajectory files; "
            f"using newest: {candidates[0].name}"
        )
        print(
            "[WARN] For a version-locked validation, pass "
            "--traj out_trajectory/<sequence_signature>.mat explicitly."
        )
    return candidates[0]


# ============================================================================
# Extract acquisition timeline from final .seq
# ============================================================================

def _apply_label_event(label_state: dict[str, int], event: Any) -> None:
    """
    更新 Pulseq label 状态。
    当前项目使用 SET，但这里同时兼容 INC，方便完整解析最终 .seq。
    """
    label = str(event.label)
    value = int(event.value)

    event_type = str(getattr(event, "type", "labelset")).lower()
    if "inc" in event_type:
        label_state[label] = int(label_state.get(label, 0)) + value
    else:
        label_state[label] = value


def extract_acquisitions(seq: Sequence) -> tuple[list[dict[str, Any]], int]:
    """
    顺序遍历最终 .seq。

    Pulseq label 是有状态的：
    - 遇到 label event 时更新当前 SLC/REP/LIN 状态
    - 遇到 ADC 时记录当时的 label 状态
    - 同时记录最近一次 RF phase 和 ADC phase
    """
    label_state: dict[str, int] = {}
    acquisitions: list[dict[str, Any]] = []
    trigger_count = 0
    last_rf_phase: float | None = None

    for block_id in seq.block_events:
        block = seq.get_block(block_id)

        if getattr(block, "trigger", None) is not None:
            trigger_count += 1

        labels = getattr(block, "label", None)
        if labels:
            for event in labels.values():
                _apply_label_event(label_state, event)

        rf = getattr(block, "rf", None)
        if rf is not None:
            last_rf_phase = float(getattr(rf, "phase_offset", 0.0))

        adc = getattr(block, "adc", None)
        if adc is None:
            continue

        acquisitions.append(
            {
                "acq_index": len(acquisitions),
                "block_id": int(block_id),
                "SLC": int(label_state.get("SLC", -1)),
                "REP": int(label_state.get("REP", -1)),
                "LIN": int(label_state.get("LIN", -1)),
                "rf_phase_rad": (
                    float(last_rf_phase)
                    if last_rf_phase is not None
                    else np.nan
                ),
                "adc_phase_rad": float(getattr(adc, "phase_offset", 0.0)),
                "block": block,
            }
        )

    return acquisitions, trigger_count


# ============================================================================
# Result collector
# ============================================================================

class Validator:
    """集中记录所有 PASS / FAIL，最后生成 Markdown 报告。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        condition = bool(condition)
        self.rows.append((name, condition, detail))
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}: {detail}")

    @property
    def all_passed(self) -> bool:
        return all(row[1] for row in self.rows)


# ============================================================================
# Check A: UIH Definitions
# ============================================================================

def validate_definitions(
    seq: Sequence,
    validator: Validator,
    expected_te_ms: float,
    expected_tr_ms: float,
) -> dict[str, Any]:
    """检查最终 .seq 中实际写入的 UIH Definitions。"""
    keys = [
        "Dimension",
        "FOV",
        "SliceNumber",
        "SliceThickness",
        "SlicePositions",
        "Matrix",
        "Resolution",
        "Center",
        "TE",
        "TR",
        "FA",
        "Name",
    ]

    definitions: dict[str, Any] = {}

    print("\n=== UIH Definitions ===")
    for key in keys:
        try:
            value = seq.get_definition(key)
        except Exception:
            value = None
        definitions[key] = value
        print(f"{key:16s}: {value}")

    validator.check(
        "Dimension",
        definitions["Dimension"] == 2,
        f"actual={definitions['Dimension']}, expected=2",
    )

    validator.check(
        "FOV",
        np.allclose(
            np.asarray(definitions["FOV"], dtype=float),
            [0.36, 0.32, 0.006],
            atol=DEFINITION_TOL,
        ),
        f"actual={definitions['FOV']}, expected=[0.36, 0.32, 0.006] m",
    )

    validator.check(
        "SliceNumber",
        int(definitions["SliceNumber"]) == 60,
        f"actual={definitions['SliceNumber']}, expected=60",
    )

    validator.check(
        "SliceThickness",
        math.isclose(
            float(definitions["SliceThickness"]),
            0.006,
            abs_tol=DEFINITION_TOL,
        ),
        f"actual={definitions['SliceThickness']}, expected=0.006 m",
    )

    validator.check(
        "Matrix",
        as_flat_list(definitions["Matrix"]) == [240, 213],
        f"actual={definitions['Matrix']}, expected=[240, 213]",
    )

    validator.check(
        "Resolution",
        as_flat_list(definitions["Resolution"]) == [240, 213],
        f"actual={definitions['Resolution']}, expected=[240, 213]",
    )

    validator.check(
        "Center",
        np.allclose(
            np.asarray(definitions["Center"], dtype=float),
            [120.0, 106.5],
            atol=DEFINITION_TOL,
        ),
        f"actual={definitions['Center']}, expected=[120.0, 106.5]",
    )

    actual_te_ms = float(definitions["TE"]) * 1e3
    actual_tr_ms = float(definitions["TR"]) * 1e3

    validator.check(
        "TE definition",
        math.isclose(
            actual_te_ms,
            expected_te_ms,
            abs_tol=1e-6,
        ),
        f"actual={actual_te_ms:.6f} ms, expected={expected_te_ms:.6f} ms",
    )

    validator.check(
        "TR definition",
        math.isclose(
            actual_tr_ms,
            expected_tr_ms,
            abs_tol=1e-6,
        ),
        f"actual={actual_tr_ms:.6f} ms, expected={expected_tr_ms:.6f} ms",
    )

    validator.check(
        "Flip angle",
        math.isclose(
            float(definitions["FA"]),
            100.0,
            abs_tol=DEFINITION_TOL,
        ),
        f"actual={definitions['FA']}, expected=100 deg",
    )

    return definitions


# ============================================================================
# Check B: ADC + SLC / REP / LIN
# ============================================================================

def validate_labels_and_adc(
    acquisitions: list[dict[str, Any]],
    trigger_count: int,
    validator: Validator,
    num_slices: int = 60,
    frames_per_slice: int = 50,
    arms_per_frame: int = 7,
) -> None:
    """验证最终 acquisition 顺序：slice -> frame -> arm。"""
    arms_per_slice = frames_per_slice * arms_per_frame
    expected_adc = num_slices * arms_per_slice

    validator.check(
        "ADC count",
        len(acquisitions) == expected_adc,
        f"actual={len(acquisitions)}, expected={expected_adc}",
    )

    validator.check(
        "Trigger count",
        trigger_count == 0,
        f"actual={trigger_count}, expected=0",
    )

    if not acquisitions:
        return

    slc = np.asarray([a["SLC"] for a in acquisitions], dtype=int)
    rep = np.asarray([a["REP"] for a in acquisitions], dtype=int)
    lin = np.asarray([a["LIN"] for a in acquisitions], dtype=int)

    validator.check(
        "SLC range",
        slc.min() == 0 and slc.max() == num_slices - 1,
        f"actual={slc.min()}..{slc.max()}, expected=0..{num_slices - 1}",
    )

    validator.check(
        "REP range",
        rep.min() == 0 and rep.max() == frames_per_slice - 1,
        f"actual={rep.min()}..{rep.max()}, expected=0..{frames_per_slice - 1}",
    )

    validator.check(
        "LIN range",
        lin.min() == 0 and lin.max() == arms_per_frame - 1,
        f"actual={lin.min()}..{lin.max()}, expected=0..{arms_per_frame - 1}",
    )

    ordering_ok = True
    first_bad = None

    for i, acq in enumerate(acquisitions):
        expected_slc = i // arms_per_slice
        local_arm = i % arms_per_slice
        expected_rep = local_arm // arms_per_frame
        expected_lin = local_arm % arms_per_frame

        actual_tuple = (
            acq["SLC"],
            acq["REP"],
            acq["LIN"],
        )
        expected_tuple = (
            expected_slc,
            expected_rep,
            expected_lin,
        )

        if actual_tuple != expected_tuple:
            ordering_ok = False
            first_bad = (
                i,
                actual_tuple,
                expected_tuple,
            )
            break

    validator.check(
        "Exact acquisition ordering",
        ordering_ok,
        (
            "all acquisitions follow SLC -> REP -> LIN"
            if ordering_ok
            else f"first mismatch={first_bad}"
        ),
    )

    per_slice_ok = all(
        np.sum(slc == s) == arms_per_slice
        for s in range(num_slices)
    )

    validator.check(
        "350 ADC per slice",
        per_slice_ok,
        f"expected={arms_per_slice} ADC for every slice",
    )

    per_frame_ok = True
    for s in range(num_slices):
        for f in range(frames_per_slice):
            count = np.sum(
                (slc == s)
                & (rep == f)
            )
            if count != arms_per_frame:
                per_frame_ok = False
                break

        if not per_frame_ok:
            break

    validator.check(
        "7 ADC per real-time frame",
        per_frame_ok,
        f"expected={arms_per_frame} arms for each (SLC, REP)",
    )


# ============================================================================
# Check C: trueFISP phase cycling
# ============================================================================

def validate_phase_cycling(
    acquisitions: list[dict[str, Any]],
    validator: Validator,
    output_dir: Path,
    arms_per_slice: int = 350,
) -> None:
    """
    检查 trueFISP 相位交替。

    注意：
    Pulseq .seq 文本保存后 phase 会出现约 1e-6 ~ 1e-5 rad 的舍入，
    因此这里使用 PHASE_TOL_RAD，而不是 1e-8 这种超过文本精度的阈值。
    """
    if len(acquisitions) < 2:
        validator.check(
            "trueFISP phase cycling",
            False,
            "fewer than two acquisitions",
        )
        return

    # ------------------------------------------------------------------
    # ADC phase: 理论上应为 0, pi, 0, pi, ...
    # ------------------------------------------------------------------
    adc_ok = True
    first_adc_bad = None
    max_adc_error_rad = 0.0

    for i, acq in enumerate(acquisitions):
        expected = float(wrap_phase(i * np.pi))
        actual = float(wrap_phase(acq["adc_phase_rad"]))
        error = circular_difference(actual, expected)

        max_adc_error_rad = max(
            max_adc_error_rad,
            error,
        )

        if error > PHASE_TOL_RAD:
            adc_ok = False
            first_adc_bad = (
                i,
                actual,
                expected,
                error,
            )
            break

    validator.check(
        "ADC phase alternation",
        adc_ok,
        (
            f"0/pi alternation for all acquisitions; "
            f"max_error={max_adc_error_rad:.3e} rad "
            f"({np.rad2deg(max_adc_error_rad):.6f} deg), "
            f"tolerance={PHASE_TOL_RAD:.1e} rad"
            if adc_ok
            else (
                f"first mismatch={first_adc_bad}; "
                f"tolerance={PHASE_TOL_RAD:.1e} rad"
            )
        ),
    )

    # ------------------------------------------------------------------
    # RF phase:
    # absolute RF phase 还包含 slice-frequency phase correction，
    # 所以这里不直接检查绝对 0/pi，而检查同一 slice 相邻 RF 相位差≈pi。
    # ------------------------------------------------------------------
    rf_ok = True
    first_rf_bad = None
    max_rf_error_rad = 0.0

    for slice_idx in range(60):
        start = slice_idx * arms_per_slice
        stop = min(
            start + arms_per_slice,
            len(acquisitions),
        )

        if stop - start < 2:
            continue

        phases = np.asarray(
            [
                a["rf_phase_rad"]
                for a in acquisitions[start:stop]
            ],
            dtype=float,
        )

        for j in range(len(phases) - 1):
            delta = float(
                wrap_phase(
                    phases[j + 1]
                    - phases[j]
                )
            )
            error = circular_difference(
                delta,
                np.pi,
            )

            max_rf_error_rad = max(
                max_rf_error_rad,
                error,
            )

            if error > PHASE_TOL_RAD:
                rf_ok = False
                first_rf_bad = (
                    slice_idx,
                    j,
                    delta,
                    error,
                )
                break

        if not rf_ok:
            break

    validator.check(
        "RF phase alternation",
        rf_ok,
        (
            f"adjacent RF phase differs by pi within every slice; "
            f"max_error={max_rf_error_rad:.3e} rad "
            f"({np.rad2deg(max_rf_error_rad):.6f} deg), "
            f"tolerance={PHASE_TOL_RAD:.1e} rad"
            if rf_ok
            else (
                f"first mismatch={first_rf_bad}; "
                f"tolerance={PHASE_TOL_RAD:.1e} rad"
            )
        ),
    )

    # ------------------------------------------------------------------
    # 可视化前 14 arms = 两个 frame
    # ------------------------------------------------------------------
    n = min(
        14,
        len(acquisitions),
    )
    indices = np.arange(n)

    rf_phase = np.asarray(
        [
            wrap_phase(a["rf_phase_rad"])
            for a in acquisitions[:n]
        ]
    )

    adc_phase = np.asarray(
        [
            wrap_phase(a["adc_phase_rad"])
            for a in acquisitions[:n]
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        indices,
        rf_phase,
        marker="o",
        label="RF phase",
    )

    ax.plot(
        indices,
        adc_phase,
        marker="x",
        label="ADC phase",
    )

    ax.set_xlabel(
        "Acquisition index"
    )

    ax.set_ylabel(
        "Phase (rad)"
    )

    ax.set_title(
        "trueFISP phase cycling: first 14 arms"
    )

    ax.set_xticks(indices)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "truefisp_phase_first14.png",
        dpi=160,
    )

    plt.close(fig)


# ============================================================================
# Check D: SlicePositions
# ============================================================================

def validate_slice_positions(
    definitions: dict[str, Any],
    validator: Validator,
    output_dir: Path,
) -> None:
    """验证 60 层、2 mm spacing、中心对称。"""
    positions = np.asarray(
        definitions["SlicePositions"],
        dtype=float,
    ).ravel()

    validator.check(
        "SlicePositions length",
        len(positions) == 60,
        f"actual={len(positions)}, expected=60",
    )

    if len(positions) < 2:
        return

    spacing = np.diff(positions)

    validator.check(
        "SlicePositions spacing",
        np.allclose(
            spacing,
            0.002,
            atol=DEFINITION_TOL,
        ),
        (
            f"mean spacing={spacing.mean() * 1e3:.6f} mm, "
            "expected=2.000000 mm"
        ),
    )

    validator.check(
        "SlicePositions symmetry",
        np.allclose(
            positions,
            -positions[::-1],
            atol=DEFINITION_TOL,
        ),
        (
            f"first={positions[0] * 1e3:.3f} mm, "
            f"last={positions[-1] * 1e3:.3f} mm"
        ),
    )

    validator.check(
        "SlicePositions extent",
        math.isclose(
            positions[0],
            -59e-3,
            abs_tol=DEFINITION_TOL,
        )
        and math.isclose(
            positions[-1],
            59e-3,
            abs_tol=DEFINITION_TOL,
        ),
        (
            f"centres={positions[0] * 1e3:.3f}.."
            f"{positions[-1] * 1e3:.3f} mm; "
            "with 6 mm thickness -> physical support about -62..62 mm"
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        np.arange(len(positions)),
        positions * 1e3,
        marker=".",
    )

    ax.set_xlabel(
        "Slice index"
    )

    ax.set_ylabel(
        "Slice centre (mm)"
    )

    ax.set_title(
        "UIH SlicePositions"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "slice_positions.png",
        dpi=160,
    )

    plt.close(fig)


# ============================================================================
# Check E: trajectory .mat + GA coverage
# ============================================================================

def validate_trajectory_metadata(
    traj_path: Path,
    validator: Validator,
    output_dir: Path,
) -> dict[str, Any]:
    """检查 trajectory metadata，并画 7 arms / 350 arms 覆盖图。"""
    print(
        f"\n[INFO] Reading trajectory metadata: {traj_path}"
    )

    mat = loadmat(
        traj_path
    )

    required = [
        "base_k_played",
        "global_arm_index",
        "global_arm_angle_deg",
        "sequence_signature",
    ]

    for key in required:
        validator.check(
            f"trajectory field: {key}",
            key in mat,
            "present" if key in mat else "missing",
        )

    if (
        "base_k_played" not in mat
        or "global_arm_angle_deg" not in mat
    ):
        return mat

    base_k = np.asarray(
        mat["base_k_played"],
        dtype=float,
    )

    angles = np.asarray(
        mat["global_arm_angle_deg"],
        dtype=float,
    ).ravel()

    validator.check(
        "base_k_played shape",
        base_k.ndim == 2
        and base_k.shape[1] == 2,
        (
            f"shape={base_k.shape}, "
            "expected=[N,2]"
        ),
    )

    validator.check(
        "global arm metadata length",
        len(angles) == 60 * 350,
        (
            f"actual={len(angles)}, "
            f"expected={60 * 350}"
        ),
    )

    if len(angles) >= 2:
        expected_second = np.mod(
            222.4969,
            360.0,
        )

        validator.check(
            "GA second-arm angle",
            math.isclose(
                float(angles[1]),
                expected_second,
                abs_tol=1e-8,
            ),
            (
                f"actual={angles[1]:.8f} deg, "
                f"expected={expected_second:.8f} deg"
            ),
        )

    # ------------------------------------------------------------------
    # 7 arms = 1 frame
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    for arm_idx in range(
        min(
            7,
            len(angles),
        )
    ):
        arm = rotate_kspace(
            base_k,
            float(angles[arm_idx]),
        )

        ax.plot(
            arm[:, 0],
            arm[:, 1],
            linewidth=0.8,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    ax.set_title(
        "First 7 golden-angle arms = one real-time frame"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "kspace_first7_arms.png",
        dpi=180,
    )

    plt.close(fig)

    # ------------------------------------------------------------------
    # 350 arms = 1 slice
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    for arm_idx in range(
        min(
            350,
            len(angles),
        )
    ):
        arm = rotate_kspace(
            base_k,
            float(angles[arm_idx]),
        )

        ax.plot(
            arm[:, 0],
            arm[:, 1],
            linewidth=0.25,
            alpha=0.35,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("kx")
    ax.set_ylabel("ky")

    ax.set_title(
        "First 350 golden-angle arms = one slice"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "kspace_first350_arms.png",
        dpi=180,
    )

    plt.close(fig)

    return mat


# ============================================================================
# Check F: final .seq first two arms vs trajectory metadata
# ============================================================================

def _calculate_kspace_for_one_adc_block(
    seq: Sequence,
    block: Any,
) -> np.ndarray:
    """
    将一个实际 ADC readout block 复制到 mini Sequence 中，
    避免对 228 MB / 21000 acquisitions 的整条序列重复 calculate_kspace()。
    """
    mini = Sequence(
        seq.system
    )

    events = []

    for attr in (
        "gx",
        "gy",
        "gz",
        "adc",
    ):
        event = getattr(
            block,
            attr,
            None,
        )

        if event is not None:
            events.append(
                copy.deepcopy(event)
            )

    if not events:
        raise RuntimeError(
            "ADC block contains no playable events"
        )

    mini.add_block(
        *events
    )

    k_adc, *_ = mini.calculate_kspace()

    return np.asarray(
        k_adc[:2, :].T,
        dtype=float,
    )


def validate_seq_vs_trajectory(
    seq: Sequence,
    acquisitions: list[dict[str, Any]],
    mat: dict[str, Any],
    validator: Validator,
) -> None:
    """
    比较最终 .seq 的实际 played gradient 推出的 k-space
    与 trajectory .mat 预测的 arm 方向。

    仅比较前两条 arm：
    - arm 0: base played arm
    - arm 1: base arm + first GA rotation

    方向容差使用 TRAJECTORY_DIRECTION_TOL_DEG，
    以适应 .seq 文本序列化后的 1e-5 deg 量级舍入。
    """
    if (
        len(acquisitions) < 2
        or "base_k_played" not in mat
    ):
        validator.check(
            "final .seq vs trajectory",
            False,
            (
                "insufficient acquisitions "
                "or missing base_k_played"
            ),
        )
        return

    base_k = np.asarray(
        mat["base_k_played"],
        dtype=float,
    )

    angles = np.asarray(
        mat["global_arm_angle_deg"],
        dtype=float,
    ).ravel()

    for arm_idx in (0, 1):
        actual = _calculate_kspace_for_one_adc_block(
            seq,
            acquisitions[arm_idx]["block"],
        )

        predicted = rotate_kspace(
            base_k,
            float(angles[arm_idx]),
        )

        if actual.shape[0] != predicted.shape[0]:
            validator.check(
                f"arm {arm_idx}: sample count",
                False,
                (
                    f"seq={actual.shape[0]}, "
                    f"trajectory={predicted.shape[0]}"
                ),
            )
            continue

        validator.check(
            f"arm {arm_idx}: sample count",
            True,
            f"N={actual.shape[0]}",
        )

        actual_dir = normalize_direction(
            actual[-1]
        )

        predicted_dir = normalize_direction(
            predicted[-1]
        )

        angular_cosine = float(
            np.clip(
                np.dot(
                    actual_dir,
                    predicted_dir,
                ),
                -1.0,
                1.0,
            )
        )

        angle_error_deg = float(
            np.rad2deg(
                np.arccos(
                    angular_cosine
                )
            )
        )

        validator.check(
            f"arm {arm_idx}: endpoint direction",
            angle_error_deg
            < TRAJECTORY_DIRECTION_TOL_DEG,
            (
                f"direction error={angle_error_deg:.9f} deg, "
                f"tolerance={TRAJECTORY_DIRECTION_TOL_DEG:.6f} deg"
            ),
        )


# ============================================================================
# Output files
# ============================================================================

def write_acquisition_csv(
    acquisitions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """保存最终 21000 次 acquisition 的 label / phase timeline。"""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "acq_index",
                "block_id",
                "SLC",
                "REP",
                "LIN",
                "rf_phase_rad",
                "adc_phase_rad",
            ]
        )

        for a in acquisitions:
            writer.writerow(
                [
                    a["acq_index"],
                    a["block_id"],
                    a["SLC"],
                    a["REP"],
                    a["LIN"],
                    a["rf_phase_rad"],
                    a["adc_phase_rad"],
                ]
            )


def write_report(
    validator: Validator,
    output_path: Path,
    seq_path: Path,
    traj_path: Path,
) -> None:
    """生成最终本地验证 Markdown 报告。"""
    lines = [
        "# Local validation report",
        "",
        f"- Sequence: `{seq_path}`",
        f"- Trajectory: `{traj_path}`",
        f"- Phase tolerance: `{PHASE_TOL_RAD:.1e} rad` "
        f"({np.rad2deg(PHASE_TOL_RAD):.6f} deg)",
        f"- Trajectory direction tolerance: "
        f"`{TRAJECTORY_DIRECTION_TOL_DEG:.6f} deg`",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]

    for name, passed, detail in validator.rows:
        detail = detail.replace(
            "|",
            "\\|",
        )

        lines.append(
            f"| {name} | "
            f"{'PASS' if passed else 'FAIL'} | "
            f"{detail} |"
        )

    lines += [
        "",
        "## Overall",
        "",
        (
            "**PASS**"
            if validator.all_passed
            else (
                "**FAIL — inspect failed items "
                "before AIDE/Virtual Scan.**"
            )
        ),
        "",
        (
            "本地检查不能替代 UIH AIDE / Virtual Scan、SAR、PNS、"
            "100° FA/B1 和 scanner interpreter 验证。"
        ),
        "",
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq",
        type=Path,
        default=None,
        help=(
            "Final .seq path. "
            "Default: <project>/out_seq/uih790_rtspiral_realtime.seq"
        ),
    )

    parser.add_argument(
        "--traj",
        type=Path,
        default=None,
        help=(
            "Trajectory .mat path. "
            "Recommended: explicitly pass the signature-matched .mat. "
            "Fallback: newest .mat in out_trajectory/."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. "
            "Default: <project>/reports/local_validation"
        ),
    )

    parser.add_argument(
        "--expected-te-ms",
        type=float,
        default=1.560,
    )

    parser.add_argument(
        "--expected-tr-ms",
        type=float,
        default=7.640,
    )

    args = parser.parse_args()

    # 本脚本默认位于：
    #   pulseq_seq/scripts/validate_uih_rtspiral_local.py
    project_root = Path(
        __file__
    ).resolve().parents[1]

    seq_path = (
        args.seq
        if args.seq is not None
        else (
            project_root
            / "out_seq"
            / "uih790_rtspiral_realtime.seq"
        )
    )

    traj_path = find_trajectory_file(
        project_root,
        args.traj,
    )

    output_dir = (
        args.out_dir
        if args.out_dir is not None
        else (
            project_root
            / "reports"
            / "local_validation"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not seq_path.is_file():
        raise FileNotFoundError(
            seq_path
        )

    if not traj_path.is_file():
        raise FileNotFoundError(
            traj_path
        )

    validator = Validator()

    # ------------------------------------------------------------------
    # 1. Read final .seq
    # ------------------------------------------------------------------
    seq = load_final_sequence(
        seq_path
    )

    # ------------------------------------------------------------------
    # 2. Definitions
    # ------------------------------------------------------------------
    definitions = validate_definitions(
        seq,
        validator,
        expected_te_ms=args.expected_te_ms,
        expected_tr_ms=args.expected_tr_ms,
    )

    # ------------------------------------------------------------------
    # 3. Extract acquisition timeline
    # ------------------------------------------------------------------
    print(
        "\n[INFO] Extracting ADC / label timeline from final .seq ..."
    )

    acquisitions, trigger_count = extract_acquisitions(
        seq
    )

    # ------------------------------------------------------------------
    # 4. ADC + labels
    # ------------------------------------------------------------------
    validate_labels_and_adc(
        acquisitions,
        trigger_count,
        validator,
        num_slices=60,
        frames_per_slice=50,
        arms_per_frame=7,
    )

    # ------------------------------------------------------------------
    # 5. trueFISP phase
    # ------------------------------------------------------------------
    validate_phase_cycling(
        acquisitions,
        validator,
        output_dir,
        arms_per_slice=350,
    )

    # ------------------------------------------------------------------
    # 6. Slice positions
    # ------------------------------------------------------------------
    validate_slice_positions(
        definitions,
        validator,
        output_dir,
    )

    # ------------------------------------------------------------------
    # 7. trajectory metadata + k-space plots
    # ------------------------------------------------------------------
    mat = validate_trajectory_metadata(
        traj_path,
        validator,
        output_dir,
    )

    # ------------------------------------------------------------------
    # 8. final .seq vs trajectory
    # ------------------------------------------------------------------
    validate_seq_vs_trajectory(
        seq,
        acquisitions,
        mat,
        validator,
    )

    # ------------------------------------------------------------------
    # 9. CSV
    # ------------------------------------------------------------------
    write_acquisition_csv(
        acquisitions,
        output_dir
        / "acquisition_timeline.csv",
    )

    # ------------------------------------------------------------------
    # 10. Markdown report
    # ------------------------------------------------------------------
    report_path = (
        output_dir
        / "LOCAL_VALIDATION_REPORT.md"
    )

    write_report(
        validator,
        report_path,
        seq_path,
        traj_path,
    )

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    print(
        "\n=== Generated outputs ==="
    )

    print(
        output_dir
        / "truefisp_phase_first14.png"
    )

    print(
        output_dir
        / "slice_positions.png"
    )

    print(
        output_dir
        / "kspace_first7_arms.png"
    )

    print(
        output_dir
        / "kspace_first350_arms.png"
    )

    print(
        output_dir
        / "acquisition_timeline.csv"
    )

    print(
        report_path
    )

    if not validator.all_passed:
        raise SystemExit(
            "\nLocal validation FAILED. "
            "Inspect LOCAL_VALIDATION_REPORT.md before AIDE."
        )

    print(
        "\nLocal validation PASSED."
    )


if __name__ == "__main__":
    main()
