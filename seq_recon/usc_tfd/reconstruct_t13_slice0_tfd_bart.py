#!/usr/bin/env python3
"""Offline USC BART TFD dry-run and deliberate real-execution runner."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from bart_tfd import (
    NLINV_COMMAND,
    USC_KSPACE_PREFACTOR,
    USC_REFERENCE_COMMIT,
    build_usc_bart_inputs,
    center_crop_native_xy,
    chunk_frame_indices,
    effective_temporal_lambda,
    estimate_scale_bart,
    nlinv_sensitivity_maps,
    pics_command,
    scale_frame_indices,
    trim_chunk_overlap,
    usc_display_transform_native_xy,
)
from t13_offline_adapter import load_adapter_config


USC_SOURCE_REPOSITORY = "https://github.com/usc-mrel/python-ismrmrd-server"
BART_EXPECTED_VERSION = "v0.9.00"


def _shape(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def _load_prepared(prepared_npz: Path, input_summary: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(prepared_npz) as prepared:
        kspace = prepared["kspace_frame_arm_coil_sample"]
        trajectory = prepared["trajectory_frame_arm_sample_kx_negative_ky"]
    return kspace, trajectory, json.loads(input_summary.read_text(encoding="utf-8"))


def _assert_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")


def _array_report(array: np.ndarray) -> dict[str, Any]:
    magnitude = np.abs(array)
    return {
        "shape": _shape(array),
        "dtype": str(array.dtype),
        "finite": bool(np.all(np.isfinite(array))),
        "magnitude_min": float(np.min(magnitude)),
        "magnitude_max": float(np.max(magnitude)),
    }


def _save_qc_images(output: Path, sens_map: np.ndarray, display_magnitude: np.ndarray) -> None:
    """Write visualization-only QC images; no displayed values feed the solver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sens_magnitude = np.abs(np.squeeze(sens_map))
    if sens_magnitude.ndim > 2:
        sens_magnitude = np.max(sens_magnitude, axis=tuple(range(2, sens_magnitude.ndim)))
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(sens_magnitude.T, cmap="gray", origin="lower")
    axis.set(title="Sensitivity-map magnitude QC", axis="off")
    figure.savefig(output / "sens_map_magnitude.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    frame_indices = np.unique(np.linspace(0, display_magnitude.shape[2] - 1, 6, dtype=int))
    figure, axes = plt.subplots(1, len(frame_indices), figsize=(3 * len(frame_indices), 3))
    axes = np.atleast_1d(axes)
    display_max = float(np.max(display_magnitude))
    for axis, frame in zip(axes, frame_indices):
        axis.imshow(display_magnitude[:, :, frame], cmap="gray", origin="lower", vmin=0, vmax=display_max)
        axis.set(title=f"frame {frame}", axis="off")
    figure.suptitle("USC display magnitude QC (shared visualization scale)")
    figure.savefig(output / "tfd_selected_frames.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def _configure_and_import_bart(bart_toolbox_path: Path, gpu_device: str | None) -> Any:
    bart_root = bart_toolbox_path.resolve()
    if not (bart_root / "bart").is_file() or not (bart_root / "python").is_dir():
        raise ValueError("--bart-toolbox-path must contain bart and python/")
    os.environ["BART_TOOLBOX_PATH"] = str(bart_root)
    os.environ["TOOLBOX_PATH"] = str(bart_root)
    os.environ["PYTHONPATH"] = f"{bart_root / 'python'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    if gpu_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    sys.path.insert(0, str(bart_root / "python"))
    module = importlib.import_module("bart")
    if not callable(getattr(module, "bart", None)):
        raise RuntimeError("BART Python module lacks callable bart.bart")
    return module


def build_dry_run_plan(prepared_npz: Path, input_summary: Path, config_path: Path, gpu_device: str | None) -> dict[str, Any]:
    config = load_adapter_config(config_path)
    kspace, trajectory, summary = _load_prepared(prepared_npz, input_summary)
    inputs = build_usc_bart_inputs(kspace, trajectory)
    frames_per_chunk = config.num_recon_frames // config.num_chunks
    chunks = chunk_frame_indices(config.num_recon_frames, config.num_chunks)
    effective_lambda = effective_temporal_lambda(config.reg_lambda_temporal, frames_per_chunk)
    if tuple(kspace.shape[:2]) != (config.num_recon_frames, config.arms_per_frame):
        raise ValueError("prepared frame/arm dimensions do not match reconstruction TOML")
    return {
        "mode": "dry-run: no BART module import or BART command execution",
        "usc_source_repository": USC_SOURCE_REPOSITORY,
        "usc_reference_commit": USC_REFERENCE_COMMIT,
        "bart_expected_version": BART_EXPECTED_VERSION,
        "source_input_summary": str(input_summary.resolve()),
        "prepared_npz": str(prepared_npz.resolve()),
        "input_summary_number_of_frames": summary.get("number_of_frames"),
        "input_shapes": {"kspace_frame_arm_coil_sample": _shape(kspace), "trajectory_frame_arm_sample_kx_negative_ky": _shape(trajectory)},
        "bart_shapes": {"kdata_semantic_sample_arm_coil_time": _shape(inputs.kdata_semantic), "kdata_bart": _shape(inputs.kdata_bart), "kloc_semantic_xyz_sample_arm_time": _shape(inputs.kloc_semantic), "kloc_bart": _shape(inputs.kloc_bart), "ksp_all_nlinv": _shape(inputs.ksp_all), "traj_all_nlinv": _shape(inputs.traj_all)},
        "native_bart_working_matrix_xy": [config.working_matrix, config.working_matrix],
        "native_bart_crop_xy": [config.crop_matrix_yx[1], config.crop_matrix_yx[0]],
        "usc_display_crop_yx": list(config.crop_matrix_yx),
        "native_bart_spatial_axes": {"axis_0": "READ_DIM / x", "axis_1": "PHS1_DIM / y"},
        "usc_kspace_prefactor": USC_KSPACE_PREFACTOR,
        "trajectory_convention": "Step-1 supplied [kx, -ky, 0]; no DCF; GIRF disabled",
        "nlinv_command": NLINV_COMMAND,
        "nlinv_input_shapes": {"traj_all": _shape(inputs.traj_all), "ksp_all": _shape(inputs.ksp_all)},
        "nlinv_merged_ordering": "arm-major / frame-fast, matching pinned USC source",
        "sensitivity_postprocess": ["IFFT axes 0,1", "resize -c 0 720 1 720", "FFT axes 0,1", "resize -c 0 360 1 360", "normalize 8"],
        "scale_estimation": {"frames_per_chunk": frames_per_chunk, "frames": scale_frame_indices(frames_per_chunk).tolist(), "range": f"5:{frames_per_chunk}", "command": "nufft -g -x 360:360:1 -a", "usc_conditional": "p90 if (max-p90) < 2*(p90-median) else max"},
        "reg_lambda_temporal_config": config.reg_lambda_temporal,
        "effective_temporal_lambda_bart": effective_lambda,
        "reg_lambda_spatial": config.reg_lambda_spatial,
        "pics_command_template": pics_command("<selected_scale_directly_from_estimate_scale_bart>", effective_lambda, config.num_iter, config.reg_lambda_spatial),
        "pics_input_shapes": {"kloc": _shape(inputs.kloc_bart), "kdata": _shape(inputs.kdata_bart), "sens_map": "created by nlinv during execute mode"},
        "chunks": [{"index": index, "frames": values.tolist()} for index, values in enumerate(chunks)],
        "apply_girf": config.apply_girf,
        "spatial_regularization_active": bool(config.reg_lambda_spatial > 0),
        "gpu_device_requested": gpu_device,
    }


def execute_reconstruction(
    prepared_npz: Path,
    input_summary: Path,
    config_path: Path,
    output: Path,
    bart: Any,
    gpu_device: str | None,
    bart_toolbox_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the source-locked BART pipeline; caller must supply runtime-imported bart."""
    total_start = time.perf_counter()
    config = load_adapter_config(config_path)
    kspace, trajectory, input_metadata = _load_prepared(prepared_npz, input_summary)
    inputs = build_usc_bart_inputs(kspace, trajectory)
    if tuple(kspace.shape[:2]) != (config.num_recon_frames, config.arms_per_frame):
        raise ValueError("prepared frame/arm dimensions do not match reconstruction TOML")
    output.mkdir(parents=True, exist_ok=True)
    frames_per_chunk = config.num_recon_frames // config.num_chunks
    scale_indices = scale_frame_indices(frames_per_chunk)
    effective_lambda = effective_temporal_lambda(config.reg_lambda_temporal, frames_per_chunk)

    nlinv_start = time.perf_counter()
    sens_map = nlinv_sensitivity_maps(bart, inputs.traj_all, inputs.ksp_all, config.working_matrix)
    nlinv_seconds = time.perf_counter() - nlinv_start
    _assert_finite(sens_map, "sensitivity map")
    np.save(output / "sens_map.npy", sens_map)

    scale_start = time.perf_counter()
    selected_scale = estimate_scale_bart(bart, inputs.kloc_bart[..., 5:frames_per_chunk], inputs.kdata_bart[..., 5:frames_per_chunk], sens_map)
    scale_seconds = time.perf_counter() - scale_start
    scale_command = f"nufft -g -x {sens_map.shape[0]}:{sens_map.shape[1]}:1 -a"

    pics_start = time.perf_counter()
    chunk_outputs: list[np.ndarray] = []
    raw_pics_shapes: list[list[int]] = []
    command = pics_command(selected_scale.selected, effective_lambda, config.num_iter, config.reg_lambda_spatial)
    for chunk_index, frame_indices in enumerate(chunk_frame_indices(config.num_recon_frames, config.num_chunks)):
        raw_pics = bart.bart(1, command, inputs.kloc_bart[..., frame_indices], inputs.kdata_bart[..., frame_indices], sens_map)
        raw_pics_shapes.append(_shape(np.asarray(raw_pics)))
        squeezed = np.squeeze(raw_pics)
        expected_chunk_shape = (config.working_matrix, config.working_matrix, len(frame_indices))
        if squeezed.shape != expected_chunk_shape:
            raise ValueError(f"raw BART PICS output shape {raw_pics_shapes[-1]} squeezes to {squeezed.shape}; expected {expected_chunk_shape}")
        _assert_finite(squeezed, "PICS output")
        chunk_outputs.append(trim_chunk_overlap(squeezed, chunk_index, config.num_chunks))
    pics_seconds = time.perf_counter() - pics_start
    native_complex = np.concatenate(chunk_outputs, axis=2)
    expected_native_shape = (config.working_matrix, config.working_matrix, config.num_recon_frames)
    if native_complex.shape != expected_native_shape:
        raise ValueError(f"native reconstructed shape {native_complex.shape}; expected {expected_native_shape}")
    _assert_finite(native_complex, "native complex reconstruction")
    native_magnitude = np.abs(native_complex)
    _assert_finite(native_magnitude, "native magnitude reconstruction")
    crop_xy = (config.crop_matrix_yx[1], config.crop_matrix_yx[0])
    native_crop_complex = center_crop_native_xy(native_complex, crop_xy)
    native_crop_magnitude = np.abs(native_crop_complex)
    display_magnitude = usc_display_transform_native_xy(native_crop_magnitude)
    if native_crop_complex.shape != (crop_xy[0], crop_xy[1], config.num_recon_frames):
        raise ValueError(f"native crop shape {native_crop_complex.shape}; expected {(crop_xy[0], crop_xy[1], config.num_recon_frames)}")
    if display_magnitude.shape != (*config.crop_matrix_yx, config.num_recon_frames):
        raise ValueError(f"USC display shape {display_magnitude.shape}; expected {(*config.crop_matrix_yx, config.num_recon_frames)}")
    _assert_finite(native_crop_complex, "native crop")
    _assert_finite(display_magnitude, "USC display magnitude")

    np.save(output / "tfd_bart_native_complex.npy", native_complex)
    np.save(output / "tfd_bart_native_magnitude.npy", native_magnitude)
    np.save(output / "tfd_native_crop_complex.npy", native_crop_complex)
    np.save(output / "tfd_native_crop_magnitude.npy", native_crop_magnitude)
    np.save(output / "tfd_usc_display_magnitude.npy", display_magnitude)
    _save_qc_images(output, sens_map, display_magnitude)

    summary: dict[str, Any] = {
        "status": "PASS", "mode": "execute", "usc_source_repository": USC_SOURCE_REPOSITORY,
        "usc_reference_commit": USC_REFERENCE_COMMIT, "bart_expected_version": BART_EXPECTED_VERSION,
        "bart_toolbox_path": str(bart_toolbox_path.resolve()) if bart_toolbox_path else None,
        "gpu_device_requested": gpu_device, "prepared_npz": str(prepared_npz.resolve()),
        "input_summary": str(input_summary.resolve()), "config": str(config_path.resolve()),
        "input_summary_number_of_frames": input_metadata.get("number_of_frames"),
        "input_shapes": {"kspace_frame_arm_coil_sample": _shape(kspace), "trajectory_frame_arm_sample_kx_negative_ky": _shape(trajectory)},
        "bart_input_shapes": {"kdata": _shape(inputs.kdata_bart), "kloc": _shape(inputs.kloc_bart), "ksp_all": _shape(inputs.ksp_all), "traj_all": _shape(inputs.traj_all)},
        "nlinv_command": NLINV_COMMAND, "nlinv_input_shapes": {"traj_all": _shape(inputs.traj_all), "ksp_all": _shape(inputs.ksp_all)},
        "sensitivity_map": _array_report(sens_map), "scale_frame_range": f"5:{frames_per_chunk}",
        "scale_nufft_command": scale_command, "scale_median": selected_scale.median,
        "scale_p90": selected_scale.p90, "scale_maximum": selected_scale.maximum,
        "selected_scale": selected_scale.selected, "scale_selection_rule": selected_scale.selection_rule,
        "selected_scale_pics_note": "selected scale is passed directly to pics -w; no reciprocal",
        "pics_command": command, "effective_temporal_lambda_bart": effective_lambda,
        "raw_bart_pics_output_shapes": raw_pics_shapes, "native_squeezed_shape": _shape(native_complex),
        "native_complex": _array_report(native_complex), "native_magnitude": _array_report(native_magnitude),
        "native_crop_shape": _shape(native_crop_complex), "native_crop_complex": _array_report(native_crop_complex),
        "native_crop_magnitude": _array_report(native_crop_magnitude), "usc_display_shape": _shape(display_magnitude),
        "usc_display_magnitude": _array_report(display_magnitude), "apply_girf": config.apply_girf,
        "spatial_regularization_active": bool(config.reg_lambda_spatial > 0),
        "output_filenames": ["sens_map.npy", "sens_map_magnitude.png", "tfd_bart_native_complex.npy", "tfd_bart_native_magnitude.npy", "tfd_native_crop_complex.npy", "tfd_native_crop_magnitude.npy", "tfd_usc_display_magnitude.npy", "tfd_selected_frames.png"],
        "timing_seconds": {"nlinv": nlinv_seconds, "scale": scale_seconds, "pics": pics_seconds, "total": time.perf_counter() - total_start},
    }
    (output / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-npz", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bart-toolbox-path", type=Path)
    parser.add_argument("--gpu-device", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate arrays and write a plan without importing BART.")
    mode.add_argument("--execute", action="store_true", help="Deliberately run BART nlinv, scale, and pics on the configured server.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = build_dry_run_plan(args.prepared_npz.resolve(), args.input_summary.resolve(), args.config.resolve(), args.gpu_device)
        if args.bart_toolbox_path:
            plan["bart_toolbox_path_requested"] = str(args.bart_toolbox_path.resolve())
        target = args.out / "tfd_dry_run_plan.json"
        target.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.bart_toolbox_path is None:
        parser.error("--execute requires --bart-toolbox-path")
    try:
        bart = _configure_and_import_bart(args.bart_toolbox_path, args.gpu_device)
        summary = execute_reconstruction(args.prepared_npz.resolve(), args.input_summary.resolve(), args.config.resolve(), args.out, bart, args.gpu_device, args.bart_toolbox_path)
    except Exception as error:
        failure = {"status": "FAIL", "mode": "execute", "error": repr(error)}
        (args.out / "reconstruction_summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    print(args.out / "reconstruction_summary.json")


if __name__ == "__main__":
    main()
