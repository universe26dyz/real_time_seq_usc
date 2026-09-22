#!/usr/bin/env python3
"""
T13_5 slice-wise RT-Spiral NUFFT baseline for UIH->ISMRMRD data.

Purpose
-------
This is a deliberately simple reconstruction/QC baseline before temporal-TV,
GIRF, B0 correction, or sophisticated coil sensitivity estimation.

It reconstructs:
  1) a 350-arm static image for one slice (trajectory sanity check)
  2) a 50-frame cine using 7 consecutive arms/frame (temporal sanity check)
  3) optional static comparison of UIH 2x-ADC-oversampling removal strategies

Current project assumptions, validated upstream:
  - H5 acquisition order matches Pulseq trajectory metadata
  - 350 arms/slice
  - 7 arms/frame -> 50 frames/slice
  - raw acquisition data shape = [2 coils, 2520 samples]
  - nominal Pulseq trajectory = 1260 ADC samples/arm
  - UIH mandatory 2x ADC oversampling gives exactly 2520/1260 = 2
  - nominal pre_discard = 10 samples
  - T13 arm angles come from global_arm_angle_deg in the trajectory .mat

Default ADC handling
--------------------
"pair_mean":
    raw[..., 0::2] and raw[..., 1::2] are complex-averaged
    -> 2520 becomes 1260 nominal 2-us samples
    -> discard first 10 nominal samples
    -> 1250 usable samples aligned with base_k_played[10:]

This is a baseline, not the final reconstruction. It does NOT use the H5
sample_time_us field, which is known to be inconsistent with the Pulseq event
timing for this converted dataset.

Dependencies
------------
    pip install numpy scipy matplotlib h5py ismrmrd sigpy

Examples
--------
python reconstruct_t13_slice0_baseline.py \
  --h5 /media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5/UID_7685991886275844011_pulseq_T13_5/testdata.h5 \
  --traj /home/universe/SVR/real_time_seq_usc/pulseq_seq/diagnostic_sequences/T13_5slice_usc144_lut/out_trajectory/b83a6a3dd57599fa51d5a6134d4fd590.mat \
  --slice 0 \
  --out /home/universe/SVR/real_time_seq_usc/seq_recon/baseline_T13_5_slice0 \
  --device -1

Use --device 0 only if SigPy/CuPy GPU support is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import ismrmrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

try:
    import sigpy as sp
except Exception as e:
    raise SystemExit(
        "SigPy is required. Install with: python -m pip install sigpy\n"
        f"Import error: {e}"
    )


def _scalar_mat(x):
    a = np.asarray(x).squeeze()
    if a.size == 1:
        return a.item()
    return a


def _mat_string(x):
    a = np.asarray(x).squeeze()
    if a.dtype.kind in ("U", "S"):
        return str(a.tolist() if a.ndim else a.item())
    if a.dtype == object:
        v = a.item() if a.size == 1 else a.flat[0]
        return str(v)
    return str(a)


def _candidate_groups(path: Path):
    out = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                score = int("data" in obj) * 3 + int("xml" in obj) * 3
                out.append((score, key))
    return [x[1] for x in sorted(out, reverse=True)]


def _open_ismrmrd(path: Path):
    names = _candidate_groups(path)
    if "dataset" in names:
        names = ["dataset"] + [n for n in names if n != "dataset"]
    if not names:
        names = ["dataset"]

    errors = {}
    for name in names:
        try:
            ds = ismrmrd.Dataset(str(path), name, create_if_needed=False)
            _nacq(ds)
            return ds, name, errors
        except Exception as e:
            errors[name] = repr(e)
    raise RuntimeError(f"Could not open ISMRMRD dataset: {errors}")


def _nacq(ds):
    for name in ("number_of_acquisitions", "getNumberOfAcquisitions"):
        fn = getattr(ds, name, None)
        if fn is not None:
            return int(fn())
    raise AttributeError("No ISMRMRD acquisition count API found")


def _read_acq(ds, i):
    for name in ("read_acquisition", "readAcquisition"):
        fn = getattr(ds, name, None)
        if fn is not None:
            return fn(i)
    raise AttributeError("No ISMRMRD acquisition read API found")


def _oversampling_reduce(data: np.ndarray, strategy: str) -> np.ndarray:
    """
    data: [n_arms, n_coils, 2520]
    returns [n_arms, n_coils, 1260]
    """
    if data.shape[-1] % 2:
        raise ValueError(f"Expected even raw sample count, got {data.shape[-1]}")
    if strategy == "pair_mean":
        return 0.5 * (data[..., 0::2] + data[..., 1::2])
    if strategy == "even":
        return data[..., 0::2]
    if strategy == "odd":
        return data[..., 1::2]
    raise ValueError(strategy)


def _rotate_trajectory(base_k: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    base_k: [samples, 2]
    angles_deg: [arms]
    returns [arms, samples, 2], physical kx/ky before USC recon convention.
    """
    th = np.deg2rad(angles_deg.astype(np.float64))
    c = np.cos(th)[:, None]
    s = np.sin(th)[:, None]
    x = base_k[None, :, 0]
    y = base_k[None, :, 1]
    xr = c * x - s * y
    yr = s * x + c * y
    return np.stack((xr, yr), axis=-1)


def _hoge_dcf_full(base_k_full: np.ndarray, dwell_s: float, pre_discard: int) -> np.ndarray:
    """
    Reproduce the USC rtspiral_pypulseq analytical DCF logic closely.
    Input is the full nominal trajectory including pre-discard samples.
    Output corresponds to base_k_full[pre_discard:].
    Overall physical scaling cancels during normalization.
    """
    kx = np.asarray(base_k_full[:, 0], dtype=np.float64)
    ky = np.asarray(base_k_full[:, 1], dtype=np.float64)
    gx = np.diff(np.concatenate(([0.0], kx))) / dwell_s / 42.58e6
    gy = np.diff(np.concatenate(([0.0], ky))) / dwell_s / 42.58e6

    # Match USC implementation, including its atan2 argument order.
    cosgk = np.cos(np.arctan2(kx, ky) - np.arctan2(gx, gy))
    w = np.sqrt(kx * kx + ky * ky) * np.sqrt(gx * gx + gy * gy) * np.abs(cosgk)
    w = w[pre_discard:].copy()

    n_full = len(base_k_full)
    n_tail = int(n_full // 2)
    if len(w) > n_tail and n_tail > 0:
        w[-n_tail:] = w[-n_tail]

    finite = np.isfinite(w)
    w[~finite] = 0
    maxw = float(np.max(w)) if w.size else 0.0
    if maxw <= 0:
        # Safe fallback. Should not normally be used.
        kr = np.linalg.norm(base_k_full[pre_discard:], axis=1)
        maxkr = float(np.max(kr)) if kr.size else 1.0
        w = kr / max(maxkr, 1e-12)
    else:
        w /= maxw
    return w.astype(np.float32)


def _sigpy_coord(kxy: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """
    Convert physical kx/ky to SigPy coordinate units.

    USC reconstruction convention uses [kx, -ky].
    SigPy coordinate order follows output array axes [y, x], so:
        coord[..., 0] = -ky scaled to Ny
        coord[..., 1] =  kx scaled to Nx
    """
    ny, nx = shape_yx
    kmax = float(np.max(np.sqrt(kxy[..., 0] ** 2 + kxy[..., 1] ** 2)))
    if not np.isfinite(kmax) or kmax <= 0:
        raise ValueError(f"Invalid kmax={kmax}")
    out = np.empty_like(kxy, dtype=np.float32)
    out[..., 0] = (-kxy[..., 1] / kmax * (ny / 2.0)).astype(np.float32)
    out[..., 1] = ( kxy[..., 0] / kmax * (nx / 2.0)).astype(np.float32)
    return out


def _adjoint_coils(data_acs: np.ndarray, coord_acs2: np.ndarray,
                   dcf_s: np.ndarray, shape_yx: tuple[int, int],
                   device_id: int) -> np.ndarray:
    """
    data_acs:  [arms, coils, samples]
    coord:     [arms, samples, 2]
    dcf:       [samples]
    returns complex coil images [coils, y, x]
    """
    dev = sp.Device(device_id)
    n_arms, n_coils, n_samples = data_acs.shape
    coord_flat = coord_acs2.reshape(-1, 2)
    weights = np.tile(dcf_s[None, :], (n_arms, 1)).reshape(-1).astype(np.float32)

    imgs = np.empty((n_coils, shape_yx[0], shape_yx[1]), dtype=np.complex64)
    with dev:
        c_gpu = sp.to_device(coord_flat, dev)
        w_gpu = sp.to_device(weights, dev)
        for c in range(n_coils):
            y = data_acs[:, c, :].reshape(-1).astype(np.complex64)
            y_gpu = sp.to_device(y, dev) * w_gpu
            im_gpu = sp.nufft_adjoint(y_gpu, c_gpu, oshape=shape_yx)
            imgs[c] = sp.to_device(im_gpu, sp.cpu_device)
    return imgs


def _rss(coil_images: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0)).astype(np.float32)


def _norm_for_png(im: np.ndarray, lo=1.0, hi=99.5) -> np.ndarray:
    x = np.asarray(im, dtype=np.float32)
    p0, p1 = np.percentile(x[np.isfinite(x)], [lo, hi])
    if not np.isfinite(p1) or p1 <= p0:
        p0, p1 = float(np.min(x)), float(np.max(x))
    return np.clip((x - p0) / max(p1 - p0, 1e-12), 0, 1)


def _save_image(path: Path, im: np.ndarray, title: str):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(_norm_for_png(im), cmap="gray", origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_montage(path: Path, cine: np.ndarray, title: str):
    # 10 representative frames. Avoid using only the very first frames because
    # bSSFP transient/steady-state effects may be strongest there.
    n = cine.shape[0]
    candidates = np.linspace(min(5, n - 1), n - 1, 10).round().astype(int)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    common_hi = np.percentile(cine[np.isfinite(cine)], 99.5)
    common_lo = np.percentile(cine[np.isfinite(cine)], 1.0)
    for ax, idx in zip(axes.ravel(), candidates):
        x = np.clip((cine[idx] - common_lo) / max(common_hi - common_lo, 1e-12), 0, 1)
        ax.imshow(x, cmap="gray", origin="upper", vmin=0, vmax=1)
        ax.set_title(f"frame {idx}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_os_compare(path: Path, ims: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, len(ims), figsize=(5 * len(ims), 5))
    if len(ims) == 1:
        axes = [axes]
    for ax, (name, im) in zip(axes, ims.items()):
        ax.imshow(_norm_for_png(im), cmap="gray", origin="upper")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("UIH 2x ADC oversampling removal: 350-arm static comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _simple_sharpness(im: np.ndarray) -> float:
    x = _norm_for_png(im)
    gy, gx = np.gradient(x)
    return float(np.mean(gx * gx + gy * gy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pre-discard", type=int, default=10)
    ap.add_argument("--arms-per-frame", type=int, default=7)
    ap.add_argument("--nominal-dwell-us", type=float, default=2.0)
    ap.add_argument("--matrix-x", type=int, default=240)
    ap.add_argument("--matrix-y", type=int, default=213)
    ap.add_argument("--device", type=int, default=-1,
                    help="-1=CPU, 0=first GPU if CuPy/SigPy GPU support exists")
    ap.add_argument("--oversampling-strategy",
                    choices=("pair_mean", "even", "odd"),
                    default="pair_mean")
    ap.add_argument("--skip-os-compare", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    h5_path = args.h5.expanduser().resolve()
    traj_path = args.traj.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Read trajectory metadata
    # -------------------------------------------------------------------------
    m = loadmat(traj_path, squeeze_me=False, struct_as_record=False)
    required = [
        "base_k_played", "global_arm_angle_deg",
        "slice_index_per_acq", "arm_index_in_slice",
        "trajectory_index_per_acq"
    ]
    missing = [k for k in required if k not in m]
    if missing:
        raise SystemExit(f"Trajectory MAT missing required keys: {missing}")

    base_k_full = np.asarray(m["base_k_played"], dtype=np.float64)
    if base_k_full.ndim != 2 or base_k_full.shape[1] != 2:
        raise SystemExit(f"Unexpected base_k_played shape: {base_k_full.shape}")

    angles_all = np.asarray(m["global_arm_angle_deg"]).squeeze().astype(np.float64)
    traj_slice_all = np.asarray(m["slice_index_per_acq"]).squeeze().astype(np.int64)
    traj_arm_local_all = np.asarray(m["arm_index_in_slice"]).squeeze().astype(np.int64)
    traj_index_all = np.asarray(m["trajectory_index_per_acq"]).squeeze().astype(np.int64)

    actual_tr_s = float(_scalar_mat(m.get("actual_tr_s", np.array([[np.nan]]))))
    actual_te_s = float(_scalar_mat(m.get("actual_te_s", np.array([[np.nan]]))))
    seq_sig = _mat_string(m.get("sequence_signature", ""))

    # -------------------------------------------------------------------------
    # Read exactly one slice from H5, preserving file acquisition ordinals
    # -------------------------------------------------------------------------
    ds, group_name, open_errors = _open_ismrmrd(h5_path)
    n_total = _nacq(ds)

    acq_ordinals = []
    lins = []
    reps = []
    timestamps = []
    raw_list = []

    for i in range(n_total):
        a = _read_acq(ds, i)
        if int(a.idx.slice) != args.slice:
            continue
        arr = np.asarray(a.data)
        if arr.ndim != 2:
            raise SystemExit(f"Unexpected acquisition data shape at {i}: {arr.shape}")
        acq_ordinals.append(i)
        lins.append(int(a.idx.kspace_encode_step_1))
        reps.append(int(a.idx.repetition))
        timestamps.append(int(a.acquisition_time_stamp))
        raw_list.append(arr.astype(np.complex64, copy=False))

    if not raw_list:
        raise SystemExit(f"No acquisitions found for slice {args.slice}")

    raw = np.stack(raw_list, axis=0)  # [arms, coils, samples]
    acq_ordinals = np.asarray(acq_ordinals, dtype=np.int64)
    lins = np.asarray(lins, dtype=np.int64)
    reps = np.asarray(reps, dtype=np.int64)

    n_arms, n_coils, n_raw_samples = raw.shape
    if n_arms != 350:
        raise SystemExit(f"Expected 350 arms for the selected slice, got {n_arms}")
    if n_raw_samples != 2 * base_k_full.shape[0]:
        raise SystemExit(
            f"Expected exact 2x raw/trajectory sample relationship, got "
            f"raw={n_raw_samples}, trajectory={base_k_full.shape[0]}"
        )

    # Metadata cross-checks for this slice.
    if int(np.max(acq_ordinals)) >= len(angles_all):
        raise SystemExit("H5 acquisition ordinal exceeds trajectory metadata length")
    if not np.array_equal(traj_slice_all[acq_ordinals], np.full(n_arms, args.slice)):
        raise SystemExit("Trajectory slice_index_per_acq does not match H5 selected slice")
    if not np.array_equal(traj_index_all[acq_ordinals], lins):
        raise SystemExit("Trajectory trajectory_index_per_acq does not match H5 LIN")
    if not np.array_equal(traj_arm_local_all[acq_ordinals], np.arange(n_arms)):
        raise SystemExit("arm_index_in_slice is not exactly 0..349")
    if np.any(reps != 0):
        raise SystemExit("REP is not all zero; this baseline assumes no scanner temporal-frame label")

    angles = angles_all[acq_ordinals]

    # -------------------------------------------------------------------------
    # Nominal-sampling data + trajectory
    # -------------------------------------------------------------------------
    data_nominal = _oversampling_reduce(raw, args.oversampling_strategy)
    n_nominal = data_nominal.shape[-1]
    if n_nominal != base_k_full.shape[0]:
        raise SystemExit(
            f"After oversampling reduction: data={n_nominal}, traj={base_k_full.shape[0]}"
        )

    pd = args.pre_discard
    if not (0 <= pd < n_nominal):
        raise SystemExit(f"Invalid pre-discard {pd} for {n_nominal} samples")

    data_use = data_nominal[..., pd:].astype(np.complex64, copy=False)
    base_k_use = base_k_full[pd:, :]
    dcf = _hoge_dcf_full(base_k_full, args.nominal_dwell_us * 1e-6, pd)

    if data_use.shape[-1] != len(base_k_use) or len(base_k_use) != len(dcf):
        raise SystemExit(
            f"Post-discard mismatch: data={data_use.shape[-1]}, "
            f"traj={len(base_k_use)}, dcf={len(dcf)}"
        )

    kxy = _rotate_trajectory(base_k_use, angles)  # [350, samples, 2]

    # Reconstruct directly into requested UIH-like [y,x] matrix.
    shape_yx = (args.matrix_y, args.matrix_x)
    coord = _sigpy_coord(kxy, shape_yx)

    # -------------------------------------------------------------------------
    # 350-arm static reconstruction
    # -------------------------------------------------------------------------
    print(f"[1/3] Static 350-arm adjoint NUFFT, shape={shape_yx}, device={args.device}")
    static_coils = _adjoint_coils(data_use, coord, dcf, shape_yx, args.device)
    static_rss = _rss(static_coils)

    np.save(out / "static_350arm_coils_complex.npy", static_coils)
    np.save(out / "static_350arm_rss.npy", static_rss)
    _save_image(
        out / "static_350arm_rss.png",
        static_rss,
        f"T13_5 slice {args.slice}: 350-arm static RSS ({args.oversampling_strategy})"
    )

    # -------------------------------------------------------------------------
    # Optional ADC-oversampling strategy comparison using the static image
    # -------------------------------------------------------------------------
    os_compare = {}
    if not args.skip_os_compare:
        print("[2/3] Oversampling-removal comparison: pair_mean / even / odd")
        for strategy in ("pair_mean", "even", "odd"):
            dn = _oversampling_reduce(raw, strategy)[..., pd:].astype(np.complex64, copy=False)
            ci = _adjoint_coils(dn, coord, dcf, shape_yx, args.device)
            ri = _rss(ci)
            os_compare[strategy] = ri
            np.save(out / f"static_350arm_rss_{strategy}.npy", ri)
        _save_os_compare(out / "static_oversampling_compare.png", os_compare)
    else:
        os_compare[args.oversampling_strategy] = static_rss

    # -------------------------------------------------------------------------
    # 50-frame 7-arm/frame cine
    # -------------------------------------------------------------------------
    apf = args.arms_per_frame
    n_frames = n_arms // apf
    if n_frames * apf != n_arms:
        raise SystemExit(f"{n_arms} arms not divisible by arms_per_frame={apf}")

    print(f"[3/3] Cine adjoint NUFFT: {n_frames} frames x {apf} arms/frame")
    cine = np.empty((n_frames, shape_yx[0], shape_yx[1]), dtype=np.float32)

    for f in range(n_frames):
        sl = slice(f * apf, (f + 1) * apf)
        ci = _adjoint_coils(data_use[sl], coord[sl], dcf, shape_yx, args.device)
        cine[f] = _rss(ci)
        if (f + 1) % 10 == 0 or f == 0:
            print(f"  frame {f+1}/{n_frames}")

    np.save(out / "cine_50frames_7arms_rss.npy", cine)
    _save_montage(
        out / "cine_montage_10frames.png",
        cine,
        f"T13_5 slice {args.slice}: adjoint cine, {apf} arms/frame"
    )

    # Save individual normalized PNGs for quick inspection.
    frame_dir = out / "cine_png"
    frame_dir.mkdir(exist_ok=True)
    common_lo = float(np.percentile(cine, 1.0))
    common_hi = float(np.percentile(cine, 99.5))
    for f in range(n_frames):
        x = np.clip((cine[f] - common_lo) / max(common_hi - common_lo, 1e-12), 0, 1)
        plt.imsave(frame_dir / f"frame_{f:03d}.png", x, cmap="gray", vmin=0, vmax=1)

    sharp = {k: _simple_sharpness(v) for k, v in os_compare.items()}

    summary = {
        "inputs": {
            "h5": str(h5_path),
            "ismrmrd_group": group_name,
            "trajectory_mat": str(traj_path),
            "sequence_signature": seq_sig,
            "slice": args.slice,
            "device": args.device,
        },
        "validated_structure": {
            "total_h5_acquisitions": n_total,
            "selected_slice_arms": n_arms,
            "coils": n_coils,
            "raw_samples_per_arm": n_raw_samples,
            "nominal_trajectory_samples_per_arm": int(base_k_full.shape[0]),
            "raw_to_nominal_ratio": float(n_raw_samples / base_k_full.shape[0]),
            "trajectory_slice_mapping_ok": True,
            "trajectory_lin_mapping_ok": True,
            "local_arm_index_0_349_ok": True,
            "rep_all_zero": True,
        },
        "adc_handling": {
            "h5_sample_time_us_is_not_used": True,
            "oversampling_strategy": args.oversampling_strategy,
            "raw_samples": n_raw_samples,
            "after_2x_os_reduction": n_nominal,
            "nominal_pre_discard": pd,
            "usable_samples_per_arm": int(data_use.shape[-1]),
            "nominal_dwell_us_used_for_dcf": args.nominal_dwell_us,
        },
        "trajectory": {
            "actual_tr_ms": actual_tr_s * 1e3 if np.isfinite(actual_tr_s) else None,
            "actual_te_ms": actual_te_s * 1e3 if np.isfinite(actual_te_s) else None,
            "global_angle_first_10_deg": angles[:10].tolist(),
            "base_k_full_shape": list(base_k_full.shape),
            "base_k_used_shape": list(base_k_use.shape),
            "sigpy_coordinate_shape": list(coord.shape),
            "dcf_min": float(np.min(dcf)),
            "dcf_max": float(np.max(dcf)),
        },
        "reconstruction": {
            "matrix_yx": list(shape_yx),
            "matrix_xy": [args.matrix_x, args.matrix_y],
            "arms_per_frame": apf,
            "n_frames": n_frames,
            "frame_duration_ms": (
                actual_tr_s * apf * 1e3 if np.isfinite(actual_tr_s) else None
            ),
            "coil_combine": "RSS after per-coil adjoint NUFFT",
            "regularization": "none",
            "GIRF": False,
            "B0_correction": False,
            "density_compensation": "USC-like analytical Hoge DCF",
        },
        "diagnostics": {
            "oversampling_strategy_static_sharpness": sharp,
            "timestamp_first": int(timestamps[0]),
            "timestamp_last": int(timestamps[-1]),
        },
        "outputs": {
            "static_rss_png": str((out / "static_350arm_rss.png").resolve()),
            "os_compare_png": (
                str((out / "static_oversampling_compare.png").resolve())
                if not args.skip_os_compare else None
            ),
            "cine_montage_png": str((out / "cine_montage_10frames.png").resolve()),
            "cine_npy": str((out / "cine_50frames_7arms_rss.npy").resolve()),
            "cine_png_dir": str(frame_dir.resolve()),
        },
        "interpretation_guardrail": (
            "This is an adjoint-NUFFT QC baseline. Seven arms/frame are severely "
            "undersampled, so streaking in cine frames is expected. The 350-arm "
            "static image is the primary trajectory/orientation sanity check. "
            "Do not tune temporal-TV/GIRF/B0 correction until the static image is coherent."
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    (out / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nDONE")
    print(f"Static image : {out / 'static_350arm_rss.png'}")
    if not args.skip_os_compare:
        print(f"OS compare   : {out / 'static_oversampling_compare.png'}")
    print(f"Cine montage : {out / 'cine_montage_10frames.png'}")
    print(f"Summary      : {out / 'baseline_summary.json'}")
    print("\nImportant: inspect the 350-arm static image first. "
          "Do not judge final cine quality from this unregularized 7-arm baseline.")


if __name__ == "__main__":
    main()
