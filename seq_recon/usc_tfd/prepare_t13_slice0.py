#!/usr/bin/env python3
"""Prepare one real UIH T13_5 slice for a future USC BART TFD reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from t13_offline_adapter import load_adapter_config, prepare_t13_slice0


def _trajectory_preview(path: Path, trajectory: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    for arm in np.linspace(0, trajectory.shape[0] - 1, 12, dtype=int):
        axis.plot(trajectory[arm, :, 0], trajectory[arm, :, 1], linewidth=0.5)
    axis.set(title="T13 physical trajectory: selected rotated arms", xlabel="kx", ylabel="ky", aspect="equal")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _frame_preview(path: Path, ordinals: np.ndarray, arms_per_frame: int) -> None:
    arm = np.arange(ordinals.size)
    fig, axis = plt.subplots(figsize=(8, 3))
    scatter = axis.scatter(arm, ordinals, c=arm // arms_per_frame, s=9, cmap="tab20")
    axis.set(title="Consecutive H5 acquisition ordinals grouped for reconstruction", xlabel="local arm index", ylabel="H5 acquisition ordinal")
    fig.colorbar(scatter, ax=axis, label="reconstruction frame")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_adapter_config(args.config)
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_t13_slice0(args.h5.resolve(), args.traj.resolve(), args.slice, config)
    np.savez_compressed(
        output / "t13_slice0_usc_tfd_input.npz",
        kspace_frame_arm_coil_sample=prepared["kspace_frames"],
        trajectory_frame_arm_sample_kx_negative_ky=prepared["bart_trajectory_frames"],
        physical_trajectory_frame_arm_sample_kx_ky=prepared["physical_trajectory"].reshape(prepared["bart_trajectory_frames"].shape),
        acquisition_ordinals=prepared["ordinals"], lin=prepared["lins"], global_arm_angle_deg=prepared["angles"],
    )
    _trajectory_preview(output / "trajectory_preview.png", prepared["physical_trajectory"])
    _frame_preview(output / "frame_grouping_preview.png", prepared["ordinals"], config.arms_per_frame)
    summary = {
        "input_h5_path": str(args.h5.resolve()), "trajectory_mat_path": str(args.traj.resolve()),
        "sequence_signature": prepared["sequence_signature"], "selected_slice": args.slice,
        "ismrmrd_group": prepared["ismrmrd_group"], "acquisition_ordinals": prepared["ordinals"].tolist(),
        "number_of_arms": int(prepared["kspace"].shape[0]), "number_of_coils": int(prepared["kspace"].shape[1]),
        "raw_samples_per_arm": 2520, "adc_reduction_method": config.adc_reduction,
        "nominal_samples_per_arm": prepared["nominal_samples"], "nominal_pre_discard": config.pre_discard,
        "usable_samples_per_arm": prepared["usable_samples"], "arms_per_frame": config.arms_per_frame,
        "number_of_frames": int(prepared["kspace_frames"].shape[0]), "base_matrix": config.base_matrix,
        "working_matrix": config.working_matrix, "fov_oversampling": config.fov_oversampling,
        "crop_matrix_yx": list(config.crop_matrix_yx), "trajectory_kmax_before_bart_scaling": prepared["kmax"],
        "trajectory_physical_shape": list(prepared["physical_trajectory"].shape),
        "trajectory_bart_shape": list(prepared["bart_trajectory_frames"].shape),
        "trajectory_bart_convention": "[kx, -ky], normalized by measured kmax and scaled to working_matrix/2",
        "kspace_shape": list(prepared["kspace_frames"].shape), "LIN_checks": prepared["mapping_checks"]["trajectory_lin_mapping_ok"],
        "REP_checks": prepared["mapping_checks"]["rep_all_zero"], "T13_mapping_checks": prepared["mapping_checks"],
        "step_2_settings_not_executed": {"reg_lambda_temporal": config.reg_lambda_temporal, "reg_lambda_spatial": config.reg_lambda_spatial, "num_iter": config.num_iter, "num_chunks": config.num_chunks, "apply_girf": config.apply_girf},
        "outputs": {"prepared_npz": str((output / "t13_slice0_usc_tfd_input.npz").resolve()), "trajectory_preview_png": str((output / "trajectory_preview.png").resolve()), "frame_grouping_preview_png": str((output / "frame_grouping_preview.png").resolve())},
        "guardrail": "Prepare-only CPU adapter. No BART, DCF, coil maps, TFD optimization, GIRF, B0 correction, spatial TV, or reconstructed image was produced.",
    }
    (output / "input_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output / "input_summary.json")


if __name__ == "__main__":
    main()
