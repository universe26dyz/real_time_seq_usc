"""CPU-only UIH T13_5 offline adapter for the later USC BART TFD step.

This module prepares and validates data only.  It intentionally does not call
BART, compute a DCF, estimate sensitivity maps, or reconstruct an image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import ismrmrd
import numpy as np
from scipy.io import loadmat
import tomli


@dataclass(frozen=True)
class AdapterConfig:
    arms_per_frame: int
    num_recon_frames: int
    pre_discard: int
    base_matrix: int
    fov_oversampling: float
    crop_matrix_yx: tuple[int, int]
    adc_reduction: str
    reg_lambda_temporal: float
    reg_lambda_spatial: float
    num_iter: int
    num_chunks: int
    apply_girf: bool

    @property
    def working_matrix(self) -> int:
        return int(round(self.base_matrix * self.fov_oversampling))


def load_adapter_config(path: Path) -> AdapterConfig:
    with path.open("rb") as handle:
        raw = tomli.load(handle)
    reconstruction = raw["reconstruction"]
    input_cfg = raw["input"]
    return AdapterConfig(
        arms_per_frame=int(reconstruction["arms_per_frame"]),
        num_recon_frames=int(reconstruction["num_recon_frames"]),
        pre_discard=int(input_cfg["nominal_pre_discard"]),
        base_matrix=int(reconstruction["base_matrix"]),
        fov_oversampling=float(reconstruction["fov_oversampling"]),
        crop_matrix_yx=(int(reconstruction["crop_matrix_y"]), int(reconstruction["crop_matrix_x"])),
        adc_reduction=str(input_cfg["adc_reduction"]),
        reg_lambda_temporal=float(reconstruction["reg_lambda_temporal"]),
        reg_lambda_spatial=float(reconstruction["reg_lambda_spatial"]),
        num_iter=int(reconstruction["num_iter"]),
        num_chunks=int(reconstruction["num_chunks"]),
        apply_girf=bool(reconstruction["apply_girf"]),
    )


def reduce_adc_oversampling(raw: np.ndarray, method: str) -> np.ndarray:
    """Explicitly reduce an even-length raw ADC axis; no physical claim is made."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[-1] % 2:
        raise ValueError(f"expected [arm, coil, even_samples] raw k-space, got {raw.shape}")
    if method == "pair_mean":
        return (0.5 * (raw[..., 0::2] + raw[..., 1::2])).astype(np.complex64, copy=False)
    if method == "even":
        return raw[..., 0::2].astype(np.complex64, copy=False)
    if method == "odd":
        return raw[..., 1::2].astype(np.complex64, copy=False)
    raise ValueError(f"unsupported ADC reduction method {method!r}")


def validate_raw_kspace_shape(raw: np.ndarray, arms: int, coils: int, samples: int) -> None:
    expected = (arms, coils, samples)
    if tuple(np.asarray(raw).shape) != expected:
        raise ValueError(f"expected raw k-space shape {expected}, got {np.asarray(raw).shape}")


def apply_pre_discard(kspace: np.ndarray, trajectory: np.ndarray, pre_discard: int) -> tuple[np.ndarray, np.ndarray]:
    if kspace.ndim != 3 or trajectory.ndim != 3 or trajectory.shape[-1] != 2:
        raise ValueError("expected k-space [arm, coil, sample] and trajectory [arm, sample, 2]")
    if kspace.shape[0] != trajectory.shape[0] or kspace.shape[-1] != trajectory.shape[1]:
        raise ValueError("k-space and trajectory arm/sample dimensions must match before pre-discard")
    if not 0 <= pre_discard < kspace.shape[-1]:
        raise ValueError(f"invalid pre-discard {pre_discard} for {kspace.shape[-1]} samples")
    return kspace[..., pre_discard:], trajectory[:, pre_discard:, :]


def rotate_base_trajectory(base_k: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rotate [sample, kx/ky] by the actual per-acquisition MAT angles."""
    base_k = np.asarray(base_k, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64).reshape(-1)
    if base_k.ndim != 2 or base_k.shape[1] != 2:
        raise ValueError(f"expected base trajectory [sample, 2], got {base_k.shape}")
    radians = np.deg2rad(angles_deg)
    cosine, sine = np.cos(radians)[:, None], np.sin(radians)[:, None]
    x, y = base_k[None, :, 0], base_k[None, :, 1]
    return np.stack((cosine * x - sine * y, sine * x + cosine * y), axis=-1)


def to_usc_bart_trajectory(physical_kxy: np.ndarray, working_matrix: int) -> tuple[np.ndarray, float]:
    """Use USC's [kx, -ky] convention, normalized by measured trajectory kmax."""
    physical_kxy = np.asarray(physical_kxy, dtype=np.float64)
    if physical_kxy.ndim != 3 or physical_kxy.shape[-1] != 2:
        raise ValueError(f"expected physical trajectory [arm, sample, 2], got {physical_kxy.shape}")
    kmax = float(np.max(np.linalg.norm(physical_kxy, axis=-1)))
    if not np.isfinite(kmax) or kmax <= 0:
        raise ValueError(f"invalid trajectory kmax {kmax}")
    result = np.empty_like(physical_kxy, dtype=np.float32)
    result[..., 0] = physical_kxy[..., 0] / kmax * (working_matrix / 2.0)
    result[..., 1] = -physical_kxy[..., 1] / kmax * (working_matrix / 2.0)
    return result, kmax


def group_consecutive_arms(kspace: np.ndarray, trajectory: np.ndarray, arms_per_frame: int) -> tuple[np.ndarray, np.ndarray]:
    if arms_per_frame <= 0 or kspace.shape[0] != trajectory.shape[0] or kspace.shape[0] % arms_per_frame:
        raise ValueError("arm count must be divisible by a positive arms_per_frame")
    frames = kspace.shape[0] // arms_per_frame
    return (
        kspace.reshape(frames, arms_per_frame, *kspace.shape[1:]),
        trajectory.reshape(frames, arms_per_frame, *trajectory.shape[1:]),
    )


def _mat_scalar(value: Any) -> Any:
    array = np.asarray(value).squeeze()
    if array.size == 1:
        return array.item()
    return array


def _mat_string(value: Any) -> str:
    array = np.asarray(value).squeeze()
    if array.dtype.kind in ("U", "S"):
        return str(array.tolist() if array.ndim else array.item())
    if array.dtype == object:
        return str(array.item() if array.size == 1 else array.flat[0])
    return str(array)


def _candidate_groups(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        candidates = [
            (int("data" in value) * 3 + int("xml" in value) * 3, key)
            for key, value in handle.items() if isinstance(value, h5py.Group)
        ]
    return [name for _, name in sorted(candidates, reverse=True)]


def _open_ismrmrd(path: Path):
    groups = _candidate_groups(path)
    if "dataset" in groups:
        groups = ["dataset"] + [group for group in groups if group != "dataset"]
    errors: dict[str, str] = {}
    for group in groups or ["dataset"]:
        try:
            dataset = ismrmrd.Dataset(str(path), group, create_if_needed=False)
            return dataset, group
        except Exception as error:
            errors[group] = repr(error)
    raise RuntimeError(f"could not open ISMRMRD dataset: {errors}")


def _nacq(dataset: Any) -> int:
    for name in ("number_of_acquisitions", "getNumberOfAcquisitions"):
        function = getattr(dataset, name, None)
        if function:
            return int(function())
    raise AttributeError("ISMRMRD dataset has no acquisition count API")


def _read_acq(dataset: Any, index: int) -> Any:
    for name in ("read_acquisition", "readAcquisition"):
        function = getattr(dataset, name, None)
        if function:
            return function(index)
    raise AttributeError("ISMRMRD dataset has no acquisition reader API")


def _canonical_data(acquisition: Any) -> np.ndarray:
    data = np.asarray(acquisition.data)
    samples, coils = int(acquisition.number_of_samples), int(acquisition.active_channels)
    if data.shape == (coils, samples):
        return data.astype(np.complex64, copy=False)
    if data.shape == (samples, coils):
        return data.T.astype(np.complex64, copy=False)
    if data.size == coils * samples:
        return data.reshape(coils, samples).astype(np.complex64, copy=False)
    raise ValueError(f"unexpected ISMRMRD acquisition data shape {data.shape}")


def _trajectory_metadata(path: Path) -> dict[str, Any]:
    fields = ("base_k_played", "global_arm_angle_deg", "slice_index_per_acq", "arm_index_in_slice", "trajectory_index_per_acq")
    metadata = loadmat(path, squeeze_me=False, struct_as_record=False)
    missing = [field for field in fields if field not in metadata]
    if missing:
        raise ValueError(f"trajectory MAT missing required fields: {missing}")
    base = np.asarray(metadata["base_k_played"], dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"unexpected base_k_played shape {base.shape}")
    return {
        "base": base,
        "angles": np.asarray(metadata["global_arm_angle_deg"]).squeeze().astype(np.float64),
        "slice": np.asarray(metadata["slice_index_per_acq"]).squeeze().astype(np.int64),
        "local_arm": np.asarray(metadata["arm_index_in_slice"]).squeeze().astype(np.int64),
        "trajectory_index": np.asarray(metadata["trajectory_index_per_acq"]).squeeze().astype(np.int64),
        "sequence_signature": _mat_string(metadata.get("sequence_signature", "")),
    }


def prepare_t13_slice0(h5_path: Path, trajectory_path: Path, slice_index: int, config: AdapterConfig) -> dict[str, Any]:
    """Read real T13 data and return validated arrays for a later BART TFD solver."""
    trajectory_meta = _trajectory_metadata(trajectory_path)
    dataset, group = _open_ismrmrd(h5_path)
    ordinals: list[int] = []
    lins: list[int] = []
    reps: list[int] = []
    raw: list[np.ndarray] = []
    for ordinal in range(_nacq(dataset)):
        acquisition = _read_acq(dataset, ordinal)
        if int(acquisition.idx.slice) != slice_index:
            continue
        ordinals.append(ordinal)
        lins.append(int(acquisition.idx.kspace_encode_step_1))
        reps.append(int(acquisition.idx.repetition))
        raw.append(_canonical_data(acquisition))
    if not raw:
        raise ValueError(f"no acquisitions found for slice {slice_index}")

    raw_array = np.stack(raw, axis=0)
    ordinal_array = np.asarray(ordinals, dtype=np.int64)
    lin_array, rep_array = np.asarray(lins, dtype=np.int64), np.asarray(reps, dtype=np.int64)
    validate_raw_kspace_shape(raw_array, arms=350, coils=2, samples=2520)
    if raw_array.shape[-1] != 2 * trajectory_meta["base"].shape[0]:
        raise ValueError("raw/nominal sample ratio is not exactly 2 for this input")
    if ordinal_array.max() >= trajectory_meta["angles"].size:
        raise ValueError("H5 acquisition ordinal exceeds trajectory metadata")
    mapping_checks = {
        "trajectory_slice_mapping_ok": bool(np.array_equal(trajectory_meta["slice"][ordinal_array], np.full(350, slice_index))),
        "trajectory_lin_mapping_ok": bool(np.array_equal(trajectory_meta["trajectory_index"][ordinal_array], lin_array)),
        "local_arm_index_0_349_ok": bool(np.array_equal(trajectory_meta["local_arm"][ordinal_array], np.arange(350))),
        "rep_all_zero": bool(np.all(rep_array == 0)),
    }
    if not all(mapping_checks.values()):
        raise ValueError(f"T13 H5/MAT mapping validation failed: {mapping_checks}")

    nominal = reduce_adc_oversampling(raw_array, config.adc_reduction)
    angles = trajectory_meta["angles"][ordinal_array]
    physical = rotate_base_trajectory(trajectory_meta["base"], angles)
    kspace, physical = apply_pre_discard(nominal, physical, config.pre_discard)
    bart, kmax = to_usc_bart_trajectory(physical, config.working_matrix)
    kspace_frames, bart_frames = group_consecutive_arms(kspace, bart, config.arms_per_frame)
    if kspace_frames.shape[0] != config.num_recon_frames:
        raise ValueError(f"expected {config.num_recon_frames} frames, got {kspace_frames.shape[0]}")
    return {
        "kspace": kspace, "physical_trajectory": physical, "bart_trajectory": bart,
        "kspace_frames": kspace_frames, "bart_trajectory_frames": bart_frames,
        "ordinals": ordinal_array, "lins": lin_array, "reps": rep_array, "angles": angles,
        "nominal_samples": int(nominal.shape[-1]), "usable_samples": int(kspace.shape[-1]),
        "kmax": kmax, "ismrmrd_group": group, "sequence_signature": trajectory_meta["sequence_signature"],
        "mapping_checks": mapping_checks,
    }
