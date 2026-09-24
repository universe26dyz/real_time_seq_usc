from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from t13_offline_adapter import (
    apply_pre_discard,
    group_consecutive_arms,
    reduce_adc_oversampling,
    rotate_base_trajectory,
    validate_raw_kspace_shape,
)


def test_adc_reduction_preserves_explicit_pair_even_and_odd_choices():
    raw = np.arange(12, dtype=np.float32).reshape(1, 2, 6).astype(np.complex64)

    np.testing.assert_allclose(
        reduce_adc_oversampling(raw, "pair_mean"),
        np.array([[[0.5, 2.5, 4.5], [6.5, 8.5, 10.5]]], dtype=np.complex64),
    )
    np.testing.assert_allclose(
        reduce_adc_oversampling(raw, "even"),
        np.array([[[0, 2, 4], [6, 8, 10]]], dtype=np.complex64),
    )
    np.testing.assert_allclose(
        reduce_adc_oversampling(raw, "odd"),
        np.array([[[1, 3, 5], [7, 9, 11]]], dtype=np.complex64),
    )


def test_pre_discard_and_frame_grouping_keep_consecutive_arm_order():
    kspace = np.zeros((350, 2, 1260), dtype=np.complex64)
    trajectory = np.zeros((350, 1260, 2), dtype=np.float32)
    kspace[:, :, 10] = np.arange(350)[:, None]
    trajectory[:, 10, 0] = np.arange(350)

    kspace_use, trajectory_use = apply_pre_discard(kspace, trajectory, 10)
    kspace_frames, trajectory_frames = group_consecutive_arms(
        kspace_use, trajectory_use, arms_per_frame=7
    )

    assert kspace_use.shape == (350, 2, 1250)
    assert trajectory_use.shape == (350, 1250, 2)
    assert kspace_frames.shape == (50, 7, 2, 1250)
    assert trajectory_frames.shape == (50, 7, 1250, 2)
    np.testing.assert_array_equal(kspace_frames[:, :, 0, 0].ravel(), np.arange(350))


def test_trajectory_rotation_uses_actual_arm_angles_and_bart_kx_negative_ky_order():
    base = np.array([[1.0, 0.0], [0.0, 1.0]])
    rotated = rotate_base_trajectory(base, np.array([0.0, 90.0]))

    np.testing.assert_allclose(rotated[0], base, atol=1e-12)
    np.testing.assert_allclose(rotated[1], np.array([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-12)
    bart = np.stack((rotated[..., 0], -rotated[..., 1]), axis=-1)
    np.testing.assert_allclose(bart[1], np.array([[0.0, -1.0], [-1.0, 0.0]]), atol=1e-12)


def test_invalid_raw_shape_is_rejected_before_adc_reduction():
    with pytest.raises(ValueError, match="expected raw k-space shape"):
        validate_raw_kspace_shape(np.zeros((350, 2, 2519), dtype=np.complex64), 350, 2, 2520)
