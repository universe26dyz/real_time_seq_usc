#!/usr/bin/env python3
"""
Read-only reconstruction preflight for UIH -> ISMRMRD RT-Spiral data.

Designed first for:
  H5: T13_5 / testdata.h5
  Trajectory: the matching T13_5 diagnostic trajectory .mat

Checks before any NUFFT reconstruction:
  - ISMRMRD acquisition count/order, scan_counter, timestamps
  - SLC/LIN/REP semantics and 350 arms/slice
  - actual data shape, finite/zero samples, per-coil RMS
  - geometry: position/read_dir/phase_dir/slice_dir and ~2 mm slice spacing
  - whether H5 contains an embedded trajectory
  - external Pulseq trajectory metadata and arm-to-trajectory mapping
  - raw-sample count vs base_k_played sample count (e.g. 2520 vs 1260)
  - leading near-zero points in base_k_played (important for pre-discard interpretation)

It DOES NOT resample data and DOES NOT reconstruct images.  It only decides
whether the data/trajectory pair is internally consistent enough to proceed.

Example:
  python preflight_rtspiral_recon.py \
    --h5 /media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5/UID_7685991886275844011_pulseq_T13_5/testdata.h5 \
    --project-root /home/universe/SVR/real_time_seq_usc/pulseq_seq \
    --protocol T13_5 \
    --out-dir /home/universe/SVR/real_time_seq_usc/seq_recon/preflight_T13_5

If auto-discovery cannot uniquely select a trajectory, pass it explicitly:
  --trajectory /path/to/<sequence_signature>.mat

Dependencies:
  python -m pip install numpy scipy h5py ismrmrd
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _scalar(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(x.item())
        except Exception:
            return default


def _vec3(x: Any) -> np.ndarray:
    try:
        return np.asarray([float(x[0]), float(x[1]), float(x[2])], dtype=float)
    except Exception:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(np.asarray(x).squeeze())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _mat_scalar(x: Any) -> Any:
    """Convert common scipy.io.loadmat scalars/strings to Python objects."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return None
    if a.dtype.kind in {"U", "S"}:
        return "".join(a.reshape(-1).astype(str)).strip()
    if a.dtype == object:
        try:
            return _mat_scalar(a.reshape(-1)[0])
        except Exception:
            return str(a)
    if a.size == 1:
        val = a.reshape(-1)[0]
        if isinstance(val, np.generic):
            val = val.item()
        return val
    return a


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _open_ismrmrd(path: Path):
    import h5py
    import ismrmrd

    with h5py.File(path, "r") as f:
        top = list(f.keys())

    candidates = []
    if "dataset" in top:
        candidates.append("dataset")
    candidates += [k for k in top if k not in candidates]

    errors = {}
    for group in candidates:
        try:
            ds = ismrmrd.Dataset(str(path), group, create_if_needed=False)
            n = _get_nacq(ds)
            return ds, group, n, top, errors
        except Exception as exc:
            errors[group] = repr(exc)
    raise RuntimeError(f"Could not open {path} as ISMRMRD. Attempts: {errors}")


def _get_nacq(ds) -> int:
    for name in ("number_of_acquisitions", "getNumberOfAcquisitions"):
        fn = getattr(ds, name, None)
        if fn is not None:
            return int(fn())
    raise AttributeError("No acquisition-count method found on ismrmrd.Dataset")


def _read_acq(ds, i: int):
    for name in ("read_acquisition", "readAcquisition"):
        fn = getattr(ds, name, None)
        if fn is not None:
            return fn(i)
    raise AttributeError("No acquisition reader found on ismrmrd.Dataset")


def _protocol_spec(protocol: str) -> dict[str, Any]:
    specs = {
        "T13_5": {
            "family": "T13", "num_slices": 5, "arms_per_slice": 350,
            "ga_policy": "usc144_lut_global_wrap", "lin_mode": "global_mod_144",
        },
        "T13_60": {
            "family": "T13", "num_slices": 60, "arms_per_slice": 350,
            "ga_policy": "usc144_lut_global_wrap", "lin_mode": "global_mod_144",
        },
        "T14_5": {
            "family": "T14", "num_slices": 5, "arms_per_slice": 350,
            "ga_policy": "lut350_slice_reset", "lin_mode": "local_0_349",
        },
        "T14_60": {
            "family": "T14", "num_slices": 60, "arms_per_slice": 350,
            "ga_policy": "lut350_slice_reset", "lin_mode": "local_0_349",
        },
        "T15_5": {
            "family": "T15", "num_slices": 5, "arms_per_slice": 350,
            "ga_policy": "continuous_global_ga", "lin_mode": "local_0_349",
        },
        "T15_60": {
            "family": "T15", "num_slices": 60, "arms_per_slice": 350,
            "ga_policy": "continuous_global_ga", "lin_mode": "local_0_349",
        },
    }
    if protocol not in specs:
        raise ValueError(f"Unsupported protocol {protocol!r}. Choose one of {sorted(specs)}")
    return specs[protocol]


def _diagnostic_id(protocol: str) -> str:
    table = {
        "T13_5": "T13_5slice_usc144_lut",
        "T13_60": "T13_60slice_usc144_lut",
        "T14_5": "T14_5slice_350lut_slice_reset",
        "T14_60": "T14_60slice_350lut_slice_reset",
        "T15_5": "T15_5slice_continuous_global_ga",
        "T15_60": "T15_60slice_continuous_global_ga",
    }
    return table[protocol]


def _trajectory_meta(path: Path) -> dict[str, Any]:
    m = loadmat(path, squeeze_me=False, struct_as_record=False)

    def arr(name: str):
        if name not in m:
            return None
        return np.asarray(m[name]).squeeze()

    def val(name: str):
        if name not in m:
            return None
        return _mat_scalar(m[name])

    base = arr("base_k_played")
    if base is None:
        base = arr("base_k")
    if base is not None:
        base = np.asarray(base)
        if base.ndim == 1:
            base = base.reshape(-1, 1)
        if base.ndim == 2 and base.shape[0] in (2, 3) and base.shape[1] > base.shape[0]:
            base = base.T

    out = {
        "path": str(path.resolve()),
        "sequence_signature": val("sequence_signature"),
        "ga_policy": val("ga_policy"),
        "ga_angle_deg": _safe_float(val("ga_angle_deg")),
        "ga_lut_length": _safe_float(val("ga_lut_length")),
        "slice_reset": val("slice_reset"),
        "played_arms_per_slice": _safe_float(val("played_arms_per_slice")),
        "arms_per_frame": _safe_float(val("arms_per_frame")),
        "frames_per_slice": _safe_float(val("frames_per_slice")),
        "num_slices": _safe_float(val("num_slices")),
        "actual_tr_s": _safe_float(val("actual_tr_s")),
        "actual_te_s": _safe_float(val("actual_te_s")),
        "base_k_played_shape": list(base.shape) if base is not None else None,
    }

    for name in (
        "global_arm_index", "global_acquisition_index", "global_arm_angle_deg",
        "slice_index_per_acq", "arm_index_in_slice", "trajectory_index_per_acq",
        "slice_positions_m",
    ):
        a = arr(name)
        out[name] = a
        out[f"{name}_length"] = int(a.size) if a is not None else None

    out["base_k_played"] = base

    # Count ADC-sampled trajectory points at/near k=0 from the beginning.
    if base is not None and base.size:
        norms = np.linalg.norm(np.real(base[:, : min(2, base.shape[1])]), axis=1)
        scale = max(float(np.nanmax(np.abs(norms))), 1.0)
        tol = scale * 1e-10
        leading_zero = 0
        for v in norms:
            if abs(float(v)) <= tol:
                leading_zero += 1
            else:
                break
        out["base_k_leading_near_zero_points"] = leading_zero
        out["base_k_norm_first_20"] = norms[:20]
    else:
        out["base_k_leading_near_zero_points"] = None
        out["base_k_norm_first_20"] = None
    return out


def _score_trajectory_candidate(path: Path, spec: dict[str, Any]) -> tuple[int, list[str]]:
    reasons = []
    score = 0
    try:
        t = _trajectory_meta(path)
    except Exception as exc:
        return -999, [f"cannot read: {exc}"]

    if t.get("ga_policy") == spec["ga_policy"]:
        score += 10
        reasons.append("ga_policy matches")
    nsl = t.get("num_slices")
    if nsl is not None and int(round(nsl)) == spec["num_slices"]:
        score += 10
        reasons.append("num_slices matches")
    aps = t.get("played_arms_per_slice")
    if aps is not None and int(round(aps)) == spec["arms_per_slice"]:
        score += 10
        reasons.append("arms_per_slice matches")
    n = t.get("global_acquisition_index_length") or t.get("global_arm_index_length")
    expected = spec["num_slices"] * spec["arms_per_slice"]
    if n == expected:
        score += 20
        reasons.append("global acquisition length matches")
    return score, reasons


def _discover_trajectory(project_root: Path, protocol: str, spec: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    diagnostic = _diagnostic_id(protocol)
    search_roots = [
        project_root / "diagnostic_sequences" / diagnostic,
        project_root / "diagnostic_sequences",
        project_root / "out_trajectory",
    ]
    seen = set()
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.mat"):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            score, reasons = _score_trajectory_candidate(rp, spec)
            candidates.append({"path": str(rp), "score": score, "reasons": reasons})

    candidates.sort(key=lambda x: (-x["score"], x["path"]))
    if not candidates:
        raise FileNotFoundError(
            f"No trajectory .mat found under {project_root}. Pass --trajectory explicitly."
        )
    if len(candidates) > 1 and candidates[0]["score"] == candidates[1]["score"]:
        raise RuntimeError(
            "Trajectory auto-discovery is ambiguous. Pass --trajectory explicitly.\n" +
            "\n".join(f"  score={c['score']:>3} {c['path']}" for c in candidates[:10])
        )
    return Path(candidates[0]["path"]), candidates


def _flags_for_acq(acq) -> list[str]:
    import ismrmrd
    names = [
        "ACQ_FIRST_IN_SLICE", "ACQ_LAST_IN_SLICE", "ACQ_IS_NOISE_MEASUREMENT",
        "ACQ_IS_PARALLEL_CALIBRATION", "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",
        "ACQ_IS_NAVIGATION_DATA", "ACQ_IS_PHASECORR_DATA", "ACQ_LAST_IN_MEASUREMENT",
        "ACQ_IS_DUMMYSCAN_DATA", "ACQ_IS_RTFEEDBACK_DATA",
    ]
    out = []
    for name in names:
        if not hasattr(ismrmrd, name):
            continue
        flag = getattr(ismrmrd, name)
        ok = False
        for fn_name in ("isFlagSet", "is_flag_set"):
            fn = getattr(acq, fn_name, None)
            if fn is not None:
                try:
                    ok = bool(fn(flag))
                except Exception:
                    ok = False
                break
        if ok:
            out.append(name)
    return out


def _data_channels_samples(acq) -> np.ndarray:
    data = np.asarray(acq.data)
    ns = _scalar(acq.number_of_samples)
    nc = _scalar(acq.active_channels)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape == (nc, ns):
        return data
    if data.shape == (ns, nc):
        return data.T
    if data.size == nc * ns:
        return data.reshape(nc, ns)
    return data


def _wrapped_uint32_diffs(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return np.asarray([], dtype=np.int64)
    a = values.astype(np.uint64)
    d = (a[1:] - a[:-1]) % (2 ** 32)
    return d.astype(np.uint64)


def audit_h5(path: Path, spec: dict[str, Any], sample_stride_for_signal: int = 1) -> dict[str, Any]:
    ds, group, n, top, open_errors = _open_ismrmrd(path)

    rows = []
    coil_sumsq = None
    coil_n = None
    coil_nonfinite = None
    coil_zero = None
    embedded_traj_nonempty = 0
    embedded_traj_shapes = Counter()
    data_shapes = Counter()
    flag_hist = Counter()

    slice_positions: dict[int, list[np.ndarray]] = defaultdict(list)
    slice_dirs: dict[int, list[np.ndarray]] = defaultdict(list)
    read_dirs: dict[int, list[np.ndarray]] = defaultdict(list)
    phase_dirs: dict[int, list[np.ndarray]] = defaultdict(list)

    for i in range(n):
        acq = _read_acq(ds, i)
        idx = acq.idx
        slc = _scalar(idx.slice)
        lin = _scalar(idx.kspace_encode_step_1)
        rep = _scalar(idx.repetition)
        flags = _flags_for_acq(acq)
        for f in flags:
            flag_hist[f] += 1

        data = _data_channels_samples(acq)
        data_shapes[str(tuple(data.shape))] += 1

        # Initialize signal statistics from observed channel count.
        if data.ndim == 2:
            nc = data.shape[0]
            if coil_sumsq is None:
                coil_sumsq = np.zeros(nc, dtype=np.float64)
                coil_n = np.zeros(nc, dtype=np.int64)
                coil_nonfinite = np.zeros(nc, dtype=np.int64)
                coil_zero = np.zeros(nc, dtype=np.int64)
            if i % max(sample_stride_for_signal, 1) == 0 and nc == len(coil_sumsq):
                for c in range(nc):
                    x = np.asarray(data[c]).reshape(-1)
                    finite = np.isfinite(x.real) & np.isfinite(x.imag)
                    coil_nonfinite[c] += int((~finite).sum())
                    xf = x[finite]
                    if xf.size:
                        coil_sumsq[c] += float(np.sum(np.abs(xf) ** 2))
                        coil_n[c] += int(xf.size)
                        coil_zero[c] += int(np.count_nonzero(xf == 0))

        traj = np.asarray(getattr(acq, "traj", np.asarray([])))
        if traj.size:
            embedded_traj_nonempty += 1
            embedded_traj_shapes[str(tuple(traj.shape))] += 1

        pos = _vec3(acq.position)
        rd = _vec3(acq.read_dir)
        pd = _vec3(acq.phase_dir)
        sd = _vec3(acq.slice_dir)
        slice_positions[slc].append(pos)
        read_dirs[slc].append(rd)
        phase_dirs[slc].append(pd)
        slice_dirs[slc].append(sd)

        rows.append({
            "i": i,
            "scan_counter": _scalar(acq.scan_counter),
            "timestamp": _scalar(acq.acquisition_time_stamp),
            "slice": slc,
            "lin": lin,
            "rep": rep,
            "samples": _scalar(acq.number_of_samples),
            "channels": _scalar(acq.active_channels),
            "trajectory_dimensions": _scalar(acq.trajectory_dimensions),
            "position": pos,
            "read_dir": rd,
            "phase_dir": pd,
            "slice_dir": sd,
            "flags": flags,
        })

    scan = np.asarray([r["scan_counter"] for r in rows], dtype=np.int64)
    ts = np.asarray([r["timestamp"] for r in rows], dtype=np.uint64)
    slc = np.asarray([r["slice"] for r in rows], dtype=np.int64)
    lin = np.asarray([r["lin"] for r in rows], dtype=np.int64)
    rep = np.asarray([r["rep"] for r in rows], dtype=np.int64)
    nsamp = np.asarray([r["samples"] for r in rows], dtype=np.int64)
    nch = np.asarray([r["channels"] for r in rows], dtype=np.int64)
    trajdims = np.asarray([r["trajectory_dimensions"] for r in rows], dtype=np.int64)

    expected_n = spec["num_slices"] * spec["arms_per_slice"]
    slice_hist = Counter(slc.tolist())

    expected_slices = np.repeat(np.arange(spec["num_slices"]), spec["arms_per_slice"])
    expected_local = np.tile(np.arange(spec["arms_per_slice"]), spec["num_slices"])
    if spec["lin_mode"] == "global_mod_144":
        expected_lin = np.arange(expected_n) % 144
    else:
        expected_lin = expected_local

    timestamp_d = _wrapped_uint32_diffs(ts)
    positive_timestamp_d = timestamp_d[timestamp_d > 0]
    timestamp_hist = Counter(positive_timestamp_d.tolist())

    # Geometry summary: median per slice; then adjacent spacing projected on slice direction.
    geom_per_slice = {}
    slice_medians = []
    slice_dir_medians = []
    for s in sorted(slice_positions):
        P = np.vstack(slice_positions[s])
        R = np.vstack(read_dirs[s])
        Q = np.vstack(phase_dirs[s])
        S = np.vstack(slice_dirs[s])
        p = np.nanmedian(P, axis=0)
        r = np.nanmedian(R, axis=0)
        q = np.nanmedian(Q, axis=0)
        z = np.nanmedian(S, axis=0)
        slice_medians.append((s, p))
        slice_dir_medians.append((s, z))
        geom_per_slice[str(s)] = {
            "position_median": p,
            "position_range_peak_to_peak": np.nanmax(P, axis=0) - np.nanmin(P, axis=0),
            "read_dir_median": r,
            "phase_dir_median": q,
            "slice_dir_median": z,
            "norm_read": float(np.linalg.norm(r)),
            "norm_phase": float(np.linalg.norm(q)),
            "norm_slice": float(np.linalg.norm(z)),
            "dot_read_phase": float(np.dot(r, q)),
            "dot_read_slice": float(np.dot(r, z)),
            "dot_phase_slice": float(np.dot(q, z)),
        }

    spacing = []
    for j in range(len(slice_medians) - 1):
        s0, p0 = slice_medians[j]
        s1, p1 = slice_medians[j + 1]
        _, z0 = slice_dir_medians[j]
        z_norm = np.linalg.norm(z0)
        dp = p1 - p0
        spacing.append({
            "slice0": s0,
            "slice1": s1,
            "delta_position_norm_mm": float(np.linalg.norm(dp)),
            "delta_projected_on_slice_dir_mm": float(np.dot(dp, z0 / z_norm)) if z_norm > 0 else None,
        })

    coil_stats = []
    if coil_sumsq is not None:
        for c in range(len(coil_sumsq)):
            rms = math.sqrt(coil_sumsq[c] / coil_n[c]) if coil_n[c] else None
            coil_stats.append({
                "coil": c,
                "rms": rms,
                "n_values_checked": int(coil_n[c]),
                "nonfinite_values": int(coil_nonfinite[c]),
                "zero_values": int(coil_zero[c]),
                "zero_fraction": float(coil_zero[c] / coil_n[c]) if coil_n[c] else None,
            })

    return {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "ismrmrd_group": group,
        "top_level_h5_keys": top,
        "open_errors_before_success": open_errors,
        "total_acquisitions": n,
        "expected_acquisitions": expected_n,
        "acquisition_count_ok": n == expected_n,
        "scan_counter_first": int(scan[0]) if scan.size else None,
        "scan_counter_last": int(scan[-1]) if scan.size else None,
        "scan_counter_exact_0_to_n_minus_1": bool(np.array_equal(scan, np.arange(n))),
        "timestamp_first": int(ts[0]) if ts.size else None,
        "timestamp_last": int(ts[-1]) if ts.size else None,
        "timestamp_positive_delta_histogram": dict(timestamp_hist),
        "timestamp_nonincreasing_count": int(np.count_nonzero(timestamp_d == 0)),
        "slice_histogram": dict(sorted(slice_hist.items())),
        "slice_sequence_exact_expected": bool(n == expected_n and np.array_equal(slc, expected_slices)),
        "lin_sequence_exact_expected": bool(n == expected_n and np.array_equal(lin, expected_lin)),
        "rep_unique": sorted(set(rep.tolist())),
        "rep_all_zero": bool(np.all(rep == 0)),
        "samples_histogram": dict(Counter(nsamp.tolist())),
        "channels_histogram": dict(Counter(nch.tolist())),
        "trajectory_dimensions_histogram": dict(Counter(trajdims.tolist())),
        "embedded_trajectory_nonempty_acquisitions": embedded_traj_nonempty,
        "embedded_trajectory_shapes": dict(embedded_traj_shapes),
        "data_shapes": dict(data_shapes),
        "flags_histogram": dict(flag_hist),
        "coil_signal_stats": coil_stats,
        "geometry_per_slice": geom_per_slice,
        "adjacent_slice_spacing": spacing,
        "arrays_for_crosscheck": {
            "slice": slc,
            "lin": lin,
            "rep": rep,
            "scan_counter": scan,
            "timestamp": ts,
        },
    }


def crosscheck(h5: dict[str, Any], traj: dict[str, Any], spec: dict[str, Any], nominal_pre_discard: int) -> dict[str, Any]:
    a = h5["arrays_for_crosscheck"]
    n = h5["total_acquisitions"]

    global_idx = traj.get("global_acquisition_index")
    if global_idx is None:
        global_idx = traj.get("global_arm_index")
    slice_idx = traj.get("slice_index_per_acq")
    local_idx = traj.get("arm_index_in_slice")
    traj_idx = traj.get("trajectory_index_per_acq")
    angles = traj.get("global_arm_angle_deg")

    def flat_int(x):
        if x is None:
            return None
        return np.asarray(x).reshape(-1).astype(np.int64)

    gi = flat_int(global_idx)
    si = flat_int(slice_idx)
    li = flat_int(local_idx)
    ti = flat_int(traj_idx)

    mapping = {
        "global_index_length_ok": gi is not None and gi.size == n,
        "global_index_exact_0_to_n_minus_1": bool(gi is not None and gi.size == n and np.array_equal(gi, np.arange(n))),
        "slice_index_length_ok": si is not None and si.size == n,
        "slice_index_matches_h5": bool(si is not None and si.size == n and np.array_equal(si, a["slice"])),
        "arm_index_in_slice_length_ok": li is not None and li.size == n,
        "arm_index_in_slice_expected_0_349": bool(
            li is not None and li.size == n and
            np.array_equal(li, np.tile(np.arange(spec["arms_per_slice"]), spec["num_slices"]))
        ),
        "trajectory_index_length_ok": ti is not None and ti.size == n,
        "global_arm_angle_length_ok": angles is not None and np.asarray(angles).size == n,
    }

    if spec["family"] in {"T13", "T14"}:
        mapping["trajectory_index_matches_h5_lin"] = bool(
            ti is not None and ti.size == n and np.array_equal(ti, a["lin"])
        )
    else:
        mapping["trajectory_index_matches_h5_lin"] = None
        mapping["t15_note"] = (
            "For T15, H5 LIN is the local 0..349 arm index; trajectory_index_per_acq is global. "
            "They are intentionally not expected to match."
        )

    raw_samples = None
    if h5["samples_histogram"]:
        keys = [int(k) for k in h5["samples_histogram"].keys()]
        if len(keys) == 1:
            raw_samples = keys[0]
    base_shape = traj.get("base_k_played_shape")
    nominal_samples = int(base_shape[0]) if base_shape else None
    ratio = (raw_samples / nominal_samples) if raw_samples and nominal_samples else None

    sample_relation = {
        "raw_samples_per_arm": raw_samples,
        "trajectory_samples_per_arm": nominal_samples,
        "raw_to_trajectory_sample_ratio": ratio,
        "ratio_is_exactly_2": bool(ratio is not None and abs(ratio - 2.0) < 1e-12),
        "nominal_pre_discard_samples_from_sequence_design": nominal_pre_discard,
        "naive_2x_equivalent_raw_pre_discard_samples": 2 * nominal_pre_discard if ratio == 2 else None,
        "base_k_leading_near_zero_points": traj.get("base_k_leading_near_zero_points"),
        "IMPORTANT": (
            "This preflight does NOT discard or decimate samples. A 2:1 count is reported as a candidate "
            "UIH ADC-oversampling relationship only. The reconstruction step must explicitly choose and "
            "document either actual-dwell trajectory resampling or a validated oversampling-removal method."
        ),
    }

    actual_tr = traj.get("actual_tr_s")
    deltas = []
    for k, v in h5["timestamp_positive_delta_histogram"].items():
        try:
            deltas += [int(k)] * int(v)
        except Exception:
            pass
    timestamp_inference = {
        "actual_tr_s_from_trajectory_metadata": actual_tr,
        "median_timestamp_delta_raw_units": float(np.median(deltas)) if deltas else None,
        "approx_seconds_per_timestamp_unit_if_one_delta_equals_one_TR": None,
    }
    if actual_tr and deltas and np.median(deltas) > 0:
        timestamp_inference["approx_seconds_per_timestamp_unit_if_one_delta_equals_one_TR"] = float(
            actual_tr / np.median(deltas)
        )

    critical = {
        "h5_acquisition_count": h5["acquisition_count_ok"],
        "h5_slice_order": h5["slice_sequence_exact_expected"],
        "h5_lin_order": h5["lin_sequence_exact_expected"],
        "h5_rep_all_zero": h5["rep_all_zero"],
        "scan_counter": h5["scan_counter_exact_0_to_n_minus_1"],
        "trajectory_slice_mapping": mapping["slice_index_matches_h5"],
        "trajectory_global_index": mapping["global_index_exact_0_to_n_minus_1"],
        "trajectory_arm_local_index": mapping["arm_index_in_slice_expected_0_349"],
    }
    ready_for_sampling_decision = all(v is True for v in critical.values())

    return {
        "mapping_checks": mapping,
        "sample_count_relationship": sample_relation,
        "timestamp_inference": timestamp_inference,
        "critical_checks": critical,
        "ready_for_sampling_strategy_decision": ready_for_sampling_decision,
        "ready_for_nufft_reconstruction": False,
        "why_not_nufft_yet": (
            "The 2520-vs-base_k sample relationship must be resolved first, and the final per-sample "
            "trajectory used for reconstruction must be explicitly constructed/validated."
        ),
    }


def write_csvs(out_dir: Path, h5: dict[str, Any]) -> dict[str, str]:
    out = {}
    geom_csv = out_dir / "slice_geometry.csv"
    with geom_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "slice", "px", "py", "pz", "rx", "ry", "rz", "qx", "qy", "qz", "sx", "sy", "sz",
            "norm_read", "norm_phase", "norm_slice", "dot_read_phase", "dot_read_slice", "dot_phase_slice",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s, g in h5["geometry_per_slice"].items():
            p, r, q, z = g["position_median"], g["read_dir_median"], g["phase_dir_median"], g["slice_dir_median"]
            w.writerow({
                "slice": s, "px": p[0], "py": p[1], "pz": p[2],
                "rx": r[0], "ry": r[1], "rz": r[2],
                "qx": q[0], "qy": q[1], "qz": q[2],
                "sx": z[0], "sy": z[1], "sz": z[2],
                "norm_read": g["norm_read"], "norm_phase": g["norm_phase"], "norm_slice": g["norm_slice"],
                "dot_read_phase": g["dot_read_phase"], "dot_read_slice": g["dot_read_slice"], "dot_phase_slice": g["dot_phase_slice"],
            })
    out["slice_geometry_csv"] = str(geom_csv.resolve())

    spacing_csv = out_dir / "slice_spacing.csv"
    with spacing_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["slice0", "slice1", "delta_position_norm_mm", "delta_projected_on_slice_dir_mm"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(h5["adjacent_slice_spacing"])
    out["slice_spacing_csv"] = str(spacing_csv.resolve())

    coil_csv = out_dir / "coil_signal_stats.csv"
    with coil_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["coil", "rms", "n_values_checked", "nonfinite_values", "zero_values", "zero_fraction"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(h5["coil_signal_stats"])
    out["coil_signal_stats_csv"] = str(coil_csv.resolve())
    return out


def strip_large_arrays(obj: dict[str, Any]) -> dict[str, Any]:
    """Keep report compact: remove full acquisition-length arrays and full base trajectory."""
    h = dict(obj["h5"])
    h.pop("arrays_for_crosscheck", None)
    t = dict(obj["trajectory"])
    for k in (
        "base_k_played", "global_arm_index", "global_acquisition_index", "global_arm_angle_deg",
        "slice_index_per_acq", "arm_index_in_slice", "trajectory_index_per_acq", "slice_positions_m",
        "base_k_norm_first_20",
    ):
        if k in t:
            if k == "base_k_norm_first_20" and t[k] is not None:
                t[k] = np.asarray(t[k]).tolist()
            else:
                t.pop(k, None)
    return {"h5": h, "trajectory": t, "crosscheck": obj["crosscheck"], "outputs": obj.get("outputs", {})}


def make_text_report(compact: dict[str, Any]) -> str:
    h = compact["h5"]
    t = compact["trajectory"]
    c = compact["crosscheck"]
    lines = []
    lines.append("=== RTSPIRAL RECONSTRUCTION PREFLIGHT ===")
    lines.append(f"H5: {h['path']}")
    lines.append(f"Trajectory: {t['path']}")
    lines.append("")
    lines.append("H5 structure")
    lines.append(f"  acquisitions: {h['total_acquisitions']} / expected {h['expected_acquisitions']} -> {h['acquisition_count_ok']}")
    lines.append(f"  SLC sequence exact: {h['slice_sequence_exact_expected']}")
    lines.append(f"  LIN sequence exact: {h['lin_sequence_exact_expected']}")
    lines.append(f"  REP all zero: {h['rep_all_zero']}")
    lines.append(f"  scan_counter 0..N-1: {h['scan_counter_exact_0_to_n_minus_1']}")
    lines.append(f"  samples: {h['samples_histogram']}")
    lines.append(f"  channels: {h['channels_histogram']}")
    lines.append(f"  data shapes: {h['data_shapes']}")
    lines.append(f"  trajectory_dimensions: {h['trajectory_dimensions_histogram']}")
    lines.append(f"  embedded trajectory nonempty acq: {h['embedded_trajectory_nonempty_acquisitions']}")
    lines.append("")
    lines.append("External trajectory")
    lines.append(f"  sequence signature: {t.get('sequence_signature')}")
    lines.append(f"  GA policy: {t.get('ga_policy')}")
    lines.append(f"  base_k_played shape: {t.get('base_k_played_shape')}")
    lines.append(f"  actual TR: {t.get('actual_tr_s')} s")
    lines.append(f"  leading near-zero base-k points: {t.get('base_k_leading_near_zero_points')}")
    lines.append("")
    lines.append("Arm mapping")
    for k, v in c["mapping_checks"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Raw/trajectory sample relationship")
    for k, v in c["sample_count_relationship"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Timestamp")
    for k, v in c["timestamp_inference"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Ready for sampling-strategy decision: {c['ready_for_sampling_strategy_decision']}")
    lines.append(f"Ready for NUFFT now: {c['ready_for_nufft_reconstruction']}")
    lines.append(f"Reason: {c['why_not_nufft_yet']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--protocol", required=True, choices=["T13_5", "T13_60", "T14_5", "T14_60", "T15_5", "T15_60"])
    ap.add_argument("--trajectory", type=Path, default=None)
    ap.add_argument("--project-root", type=Path, default=Path("/home/universe/SVR/real_time_seq_usc/pulseq_seq"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--nominal-pre-discard", type=int, default=10)
    ap.add_argument("--signal-stat-stride", type=int, default=1, help="1 = inspect signal values from every acquisition")
    args = ap.parse_args()

    h5_path = args.h5.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = _protocol_spec(args.protocol)

    candidate_log = []
    if args.trajectory is not None:
        traj_path = args.trajectory.expanduser().resolve()
    else:
        traj_path, candidate_log = _discover_trajectory(project_root, args.protocol, spec)

    print(f"H5         : {h5_path}")
    print(f"Protocol   : {args.protocol}")
    print(f"Trajectory : {traj_path}")
    print("Reading H5 acquisitions (read-only)...")

    h5 = audit_h5(h5_path, spec, sample_stride_for_signal=args.signal_stat_stride)
    traj = _trajectory_meta(traj_path)
    cross = crosscheck(h5, traj, spec, nominal_pre_discard=args.nominal_pre_discard)

    result = {"h5": h5, "trajectory": traj, "crosscheck": cross}
    outputs = write_csvs(out_dir, h5)
    outputs["trajectory_autodiscovery_candidates"] = candidate_log
    result["outputs"] = outputs

    compact = strip_large_arrays(result)
    summary_path = out_dir / "preflight_summary.json"
    summary_path.write_text(json.dumps(_jsonable(compact), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = out_dir / "preflight_report.txt"
    report_path.write_text(make_text_report(_jsonable(compact)), encoding="utf-8")

    print("\n" + report_path.read_text(encoding="utf-8"))
    print(f"Summary: {summary_path}")
    print(f"Report : {report_path}")
    print("Send me preflight_summary.json first. If geometry is suspicious, also send slice_geometry.csv and slice_spacing.csv.")


if __name__ == "__main__":
    main()
