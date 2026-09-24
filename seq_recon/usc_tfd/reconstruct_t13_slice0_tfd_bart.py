#!/usr/bin/env python3
"""Build and record an offline USC BART TFD plan from a validated Step-1 NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bart_tfd import (
    NLINV_COMMAND,
    USC_KSPACE_PREFACTOR,
    USC_REFERENCE_COMMIT,
    build_usc_bart_inputs,
    chunk_frame_indices,
    effective_temporal_lambda,
    pics_command,
    scale_frame_indices,
)
from t13_offline_adapter import load_adapter_config


def _shape(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.shape]


def build_dry_run_plan(prepared_npz: Path, input_summary: Path, config_path: Path, gpu_device: str | None) -> dict:
    config = load_adapter_config(config_path)
    with np.load(prepared_npz) as prepared:
        kspace = prepared["kspace_frame_arm_coil_sample"]
        trajectory = prepared["trajectory_frame_arm_sample_kx_negative_ky"]
    summary = json.loads(input_summary.read_text(encoding="utf-8"))
    inputs = build_usc_bart_inputs(kspace, trajectory)
    frames_per_chunk = config.num_recon_frames // config.num_chunks
    chunks = chunk_frame_indices(config.num_recon_frames, config.num_chunks)
    effective_lambda = effective_temporal_lambda(config.reg_lambda_temporal, frames_per_chunk)
    if tuple(kspace.shape[:2]) != (config.num_recon_frames, config.arms_per_frame):
        raise ValueError("prepared frame/arm dimensions do not match reconstruction TOML")
    return {
        "mode": "dry-run: no BART module import or BART command execution",
        "usc_reference_commit": USC_REFERENCE_COMMIT,
        "source_input_summary": str(input_summary.resolve()),
        "prepared_npz": str(prepared_npz.resolve()),
        "input_summary_number_of_frames": summary.get("number_of_frames"),
        "input_shapes": {"kspace_frame_arm_coil_sample": _shape(kspace), "trajectory_frame_arm_sample_kx_negative_ky": _shape(trajectory)},
        "bart_shapes": {
            "kdata_semantic_sample_arm_coil_time": _shape(inputs.kdata_semantic),
            "kdata_bart": _shape(inputs.kdata_bart),
            "kloc_semantic_xyz_sample_arm_time": _shape(inputs.kloc_semantic),
            "kloc_bart": _shape(inputs.kloc_bart),
            "ksp_all_nlinv": _shape(inputs.ksp_all),
            "traj_all_nlinv": _shape(inputs.traj_all),
        },
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
        "pics_command_template": pics_command("<scale_from_estimate_scale_bart>", effective_lambda, config.num_iter, config.reg_lambda_spatial),
        "pics_input_shapes": {"kloc": _shape(inputs.kloc_bart), "kdata": _shape(inputs.kdata_bart), "sens_map": "created by nlinv during Step 2B"},
        "chunks": [{"index": index, "frames": values.tolist()} for index, values in enumerate(chunks)],
        "apply_girf": config.apply_girf,
        "spatial_regularization_active": bool(config.reg_lambda_spatial > 0),
        "gpu_device_requested": gpu_device,
        "future_outputs": ["sens_map.npy", "sens_map_magnitude.png", "tfd_bart_native_complex.npy [x,y,time]", "tfd_bart_native_magnitude.npy [x,y,time]", "native center crop [x,y]=[240,213]", "USC display transform (flip x, swap x/y) -> [y,x]=[213,240]"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-npz", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bart-toolbox-path", type=Path, help="Recorded only in dry-run; BART is not imported.")
    parser.add_argument("--gpu-device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Required for Step 2A; validates and writes a plan only.")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("Step 2A intentionally permits only --dry-run; execute BART later in Step 2B.")
    plan = build_dry_run_plan(args.prepared_npz.resolve(), args.input_summary.resolve(), args.config.resolve(), args.gpu_device)
    if args.bart_toolbox_path:
        plan["bart_toolbox_path_requested"] = str(args.bart_toolbox_path.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "tfd_dry_run_plan.json"
    target.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
