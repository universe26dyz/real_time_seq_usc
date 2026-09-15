"""Task 1 real-time label and slice-position contracts."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uih_definitions import realtime_labels, slice_positions_m


def test_realtime_label_mapping_at_frame_boundaries():
    assert realtime_labels(0, 0, 7) == (0, 0, 0)
    assert realtime_labels(0, 6, 7) == (0, 0, 6)
    assert realtime_labels(0, 7, 7) == (0, 1, 0)
    assert realtime_labels(0, 349, 7) == (0, 49, 6)


def test_slice_positions_are_center_symmetric_in_metres():
    assert slice_positions_m(4, 2.0) == [-0.003, -0.001, 0.001, 0.003]


@pytest.mark.parametrize("num_slices, shift_mm", [(0, 2.0), (3, 0.0)])
def test_slice_positions_reject_invalid_geometry(num_slices, shift_mm):
    with pytest.raises(ValueError):
        slice_positions_m(num_slices, shift_mm)
