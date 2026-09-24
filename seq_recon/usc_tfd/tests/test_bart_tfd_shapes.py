from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from bart_tfd import (
    USC_KSPACE_PREFACTOR,
    build_usc_bart_inputs,
    center_crop_native_xy,
    chunk_frame_indices,
    effective_temporal_lambda,
    scale_frame_indices,
    trim_chunk_overlap,
    usc_display_transform_native_xy,
)


def _synthetic_inputs():
    frames, arms, coils, samples = 50, 7, 2, 1250
    arm_time = np.arange(frames * arms, dtype=np.float32).reshape(frames, arms)
    kspace = np.zeros((frames, arms, coils, samples), dtype=np.complex64)
    kspace[:, :, 0, 0] = arm_time
    kspace[:, :, 1, 0] = arm_time + 1000j
    trajectory = np.zeros((frames, arms, samples, 2), dtype=np.float32)
    trajectory[:, :, 0, 0] = arm_time
    trajectory[:, :, 0, 1] = -arm_time
    return kspace, trajectory


def test_usc_bart_layouts_keep_arm_major_frame_fast_order_and_prefactor():
    kspace, trajectory = _synthetic_inputs()
    inputs = build_usc_bart_inputs(kspace, trajectory)

    assert inputs.kdata_semantic.shape == (1250, 7, 2, 50)
    assert inputs.kdata_bart.shape == (1, 1250, 7, 2, 1, 1, 1, 1, 1, 1, 50)
    assert inputs.kloc_semantic.shape == (3, 1250, 7, 50)
    assert inputs.kloc_bart.shape == (3, 1250, 7, 1, 1, 1, 1, 1, 1, 1, 50)
    assert inputs.ksp_all.shape == (1, 1250, 350, 2)
    assert inputs.traj_all.shape == (3, 1250, 350)
    expected_usc_order = np.arange(350).reshape(50, 7).T.reshape(-1)
    np.testing.assert_array_equal(inputs.ksp_all[0, 0, :, 0], expected_usc_order * USC_KSPACE_PREFACTOR)
    np.testing.assert_array_equal(inputs.ksp_all[0, 0, :, 1], expected_usc_order * USC_KSPACE_PREFACTOR + 1000j * USC_KSPACE_PREFACTOR)
    np.testing.assert_array_equal(inputs.traj_all[0, 0], expected_usc_order)
    np.testing.assert_array_equal(inputs.traj_all[1, 0], -expected_usc_order)
    np.testing.assert_array_equal(inputs.traj_all[2], 0)
    assert kspace[1, 0, 0, 0] == 7


def test_temporal_lambda_scale_frames_and_usc_chunk_overlap():
    assert effective_temporal_lambda(0.0002, 50) == pytest.approx(0.01)
    np.testing.assert_array_equal(scale_frame_indices(50), np.arange(5, 50))
    one = chunk_frame_indices(50, 1)
    assert len(one) == 1
    np.testing.assert_array_equal(one[0], np.arange(50))
    assert trim_chunk_overlap(np.ones((2, 2, 50)), 0, 1).shape[-1] == 50

    chunks = chunk_frame_indices(12, 3)
    np.testing.assert_array_equal(chunks[0], np.arange(7))
    np.testing.assert_array_equal(chunks[1], np.arange(1, 11))
    np.testing.assert_array_equal(chunks[2], np.arange(5, 12))
    assert trim_chunk_overlap(np.ones((2, 2, 7)), 0, 3).shape[-1] == 4
    assert trim_chunk_overlap(np.ones((2, 2, 10)), 1, 3).shape[-1] == 4
    assert trim_chunk_overlap(np.ones((2, 2, 7)), 2, 3).shape[-1] == 4
    with pytest.raises(ValueError, match="divisible"):
        chunk_frame_indices(50, 3)
    np.testing.assert_array_equal(scale_frame_indices(frames_per_chunk=8), np.array([5, 6, 7]))


def test_native_xy_crop_and_separate_usc_display_transform_have_deterministic_axes():
    native = np.arange(360 * 360 * 2).reshape(360, 360, 2)
    cropped = center_crop_native_xy(native, (240, 213))
    assert cropped.shape == (240, 213, 2)
    np.testing.assert_array_equal(cropped, native[60:300, 73:286, :])

    display = usc_display_transform_native_xy(cropped)
    assert display.shape == (213, 240, 2)
    np.testing.assert_array_equal(display[0, 0], cropped[-1, 0])
    np.testing.assert_array_equal(display[-1, -1], cropped[0, -1])
