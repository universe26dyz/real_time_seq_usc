"""Offline port of USC's BART TFD array and solver preparation.

The functions that invoke BART are deliberately separate from the CPU-only
dry-run helpers.  Step 2A builds and validates the exact solver inputs without
importing BART or starting a reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


USC_REFERENCE_COMMIT = "faaf0f3e44bf8f2557ac4d7da90a62a5451d7b96"
USC_KSPACE_PREFACTOR = 1e3
NLINV_COMMAND = "nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t"


@dataclass(frozen=True)
class BartInputs:
    """USC layouts; BART arrays use 11 dimensions with time at dimension 10."""

    kdata_semantic: np.ndarray  # [sample, arm, coil, time]
    kdata_bart: np.ndarray      # [1, sample, arm, coil, 1, 1, 1, 1, 1, 1, time]
    kloc_semantic: np.ndarray   # [xyz, sample, arm, time]
    kloc_bart: np.ndarray       # [xyz, sample, arm, 1, 1, 1, 1, 1, 1, 1, time]
    ksp_all: np.ndarray         # [1, sample, arm_major_frame_fast, coil]
    traj_all: np.ndarray        # [xyz, sample, arm_major_frame_fast]


def _validate_prepared_shapes(kspace: np.ndarray, trajectory: np.ndarray) -> tuple[int, int, int, int]:
    if kspace.ndim != 4 or trajectory.ndim != 4 or trajectory.shape[-1] != 2:
        raise ValueError("expected k-space [frame, arm, coil, sample] and trajectory [frame, arm, sample, 2]")
    frames, arms, coils, samples = kspace.shape
    if trajectory.shape[:3] != (frames, arms, samples):
        raise ValueError(f"k-space shape {kspace.shape} and trajectory shape {trajectory.shape} do not align")
    if min(frames, arms, coils, samples) <= 0:
        raise ValueError("prepared k-space dimensions must be non-zero")
    return frames, arms, coils, samples


def build_usc_bart_inputs(kspace_frames: np.ndarray, trajectory_frames: np.ndarray) -> BartInputs:
    """Build USC BART arrays from Step-1 [F,A,C,S] and [F,A,S,2] inputs.

    The dynamic arrays follow the pinned source's [S,A,C,F] and [3,S,A,F]
    layout verbatim.  The static nlinv arrays are derived from those dynamic
    layouts exactly as in USC: arm-major, frame-fast (arm 0 frames 0..F-1,
    then arm 1).  K-space and trajectory use the same merged ordering.
    """
    kspace = np.asarray(kspace_frames)
    trajectory = np.asarray(trajectory_frames)
    frames, arms, coils, samples = _validate_prepared_shapes(kspace, trajectory)

    # Pinned USC code applies this formal prefactor immediately before BART
    # organisation.  Multiplication returns a new array and leaves Step 1 data untouched.
    formal_data = np.asarray(kspace * USC_KSPACE_PREFACTOR, dtype=np.complex64)
    kdata_semantic = np.transpose(formal_data, (3, 1, 2, 0))
    kdata_bart = kdata_semantic[None, :, :, :, None, None, None, None, None, None, :]

    kz = np.zeros((frames, arms, samples, 1), dtype=np.float32)
    kloc_frame_arm_sample_xyz = np.concatenate((trajectory.astype(np.float32, copy=False), kz), axis=3)
    kloc_semantic = np.transpose(kloc_frame_arm_sample_xyz, (3, 2, 1, 0))
    kloc_bart = kloc_semantic[:, :, :, None, None, None, None, None, None, None, :]

    # USC source: transpose [S,A,C,F] -> [S,A,F,C], then reshape A*F.
    # The result is arm-major/frame-fast, not Step-1's frame-major ordering.
    ksp_all = np.transpose(kdata_semantic, (0, 1, 3, 2)).reshape(1, samples, arms * frames, coils)
    # USC source: reshape the existing [3,S,A,F] dynamic trajectory directly.
    traj_all = kloc_semantic.reshape(3, samples, arms * frames)

    expected_dynamic = (samples, arms, coils, frames)
    if kdata_semantic.shape != expected_dynamic or kloc_semantic.shape != (3, samples, arms, frames):
        raise AssertionError("USC dynamic BART dimension construction failed")
    if ksp_all.shape != (1, samples, frames * arms, coils) or traj_all.shape != (3, samples, frames * arms):
        raise AssertionError("USC nlinv input dimension construction failed")
    return BartInputs(kdata_semantic, kdata_bart, kloc_semantic, kloc_bart, ksp_all, traj_all)


def effective_temporal_lambda(reg_lambda_temporal: float, frames_per_chunk: int) -> float:
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be positive")
    return float(reg_lambda_temporal) * frames_per_chunk


def scale_frame_indices(frames_per_chunk: int, excluded_initial_frames: int = 5) -> np.ndarray:
    """USC scale-estimation frames 5:n_frame_per_chunk from the first chunk."""
    if frames_per_chunk <= excluded_initial_frames:
        raise ValueError("not enough frames after excluding initial scale-estimation frames")
    return np.arange(excluded_initial_frames, frames_per_chunk, dtype=np.int64)


def chunk_frame_indices(num_frames: int, num_chunks: int, overlap: int = 3) -> list[np.ndarray]:
    """Port USC's +3/-3 frame chunk selection, with divisibility a hard error."""
    if num_chunks <= 0 or num_frames % num_chunks:
        raise ValueError("number of reconstruction frames must be divisible by number of chunks")
    frames_per_chunk = num_frames // num_chunks
    if num_chunks > 1 and frames_per_chunk <= overlap:
        raise ValueError("frames per chunk must exceed the USC overlap")
    chunks: list[np.ndarray] = []
    for index in range(num_chunks):
        start, stop = index * frames_per_chunk, (index + 1) * frames_per_chunk
        if num_chunks == 1:
            chunks.append(np.arange(start, stop))
        elif index == 0:
            chunks.append(np.arange(start, stop + overlap))
        elif index == num_chunks - 1:
            chunks.append(np.arange(start - overlap, stop))
        else:
            chunks.append(np.arange(start - overlap, stop + overlap))
    return chunks


def trim_chunk_overlap(image: np.ndarray, chunk_index: int, num_chunks: int, overlap: int = 3) -> np.ndarray:
    """Remove the same temporal overlap frames as the pinned USC source."""
    if num_chunks == 1:
        return image
    if chunk_index == 0:
        return image[..., :-overlap]
    if chunk_index == num_chunks - 1:
        return image[..., overlap:]
    return image[..., overlap:-overlap]


def pics_command(scale: str | float, temporal_lambda: float, num_iter: int, spatial_lambda: float = 0.0) -> str:
    command = (
        f"pics -g -m -w {scale} -e -S -R T:1024:1024:{temporal_lambda} "
        f"-d 4 -i {num_iter} -t"
    )
    if spatial_lambda > 0:
        command = (
            f"pics -g -m -w {scale} -e -S -R T:1024:1024:{temporal_lambda} "
            f"-R W:3:0:{spatial_lambda} -d 4 -i {num_iter} -t"
        )
    return command


def select_usc_scale(median: float, p90: float, maximum: float) -> float:
    """Exact conditional selection in USC's estimate_scale_bart()."""
    return p90 if (maximum - p90) < 2 * (p90 - median) else maximum


def estimate_scale_bart(bart: Any, traj: np.ndarray, ksp: np.ndarray, sens_map: np.ndarray) -> float:
    """BART-dependent, exact USC nufft-adjoint scale estimate; not called by dry-run."""
    first_it = bart.bart(1, f"nufft -g -x {sens_map.shape[0]}:{sens_map.shape[1]}:1 -a ", traj, ksp)
    first_it = np.sum(
        first_it * np.conj(sens_map[:, :, :, :, None, None, None, None, None, None, None]), axis=3
    )
    magnitudes = np.abs(first_it[:])
    return float(select_usc_scale(np.median(magnitudes), np.percentile(magnitudes, 90), np.max(magnitudes)))


def nlinv_sensitivity_maps(bart: Any, traj_all: np.ndarray, ksp_all: np.ndarray, msize: int) -> np.ndarray:
    """Port the USC nlinv → resize → normalize sensitivity-map pathway."""
    _, rtnlinv_sens_32 = bart.bart(2, NLINV_COMMAND, traj_all, ksp_all)
    sens_ksp = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(rtnlinv_sens_32, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
    )
    sens_ksp = bart.bart(1, f"resize -c 0 {msize * 2} 1 {msize * 2}", sens_ksp)
    sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sens_ksp, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sens = bart.bart(1, f"resize -c 0 {msize} 1 {msize}", sens)
    return bart.bart(1, "normalize 8", sens)


def center_crop_native_xy(native: np.ndarray, crop_xy: Sequence[int]) -> np.ndarray:
    """Crop native complex BART [x/read, y/phase, ...] without display orientation."""
    crop_x, crop_y = (int(crop_xy[0]), int(crop_xy[1]))
    if native.ndim < 2 or crop_x > native.shape[0] or crop_y > native.shape[1]:
        raise ValueError("crop must fit the first two native BART dimensions")
    start_x, start_y = (native.shape[0] - crop_x) // 2, (native.shape[1] - crop_y) // 2
    return native[start_x:start_x + crop_x, start_y:start_y + crop_y, ...]


def usc_display_transform_native_xy(native: np.ndarray) -> np.ndarray:
    """USC process_group spatial transform: native [x,y,...] -> display [y,x,...]."""
    if native.ndim < 2:
        raise ValueError("native BART result must have x and y axes")
    return np.flip(native, axis=0).swapaxes(0, 1)
