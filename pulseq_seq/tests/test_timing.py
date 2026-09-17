"""Task 2 generator timing and acquisition-order contracts."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_loader import load_config
from write_rtspiral_svr_uih import build_sequence


CONFIG = PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json"


@pytest.fixture(scope="module")
def one_slice_result(tmp_path_factory):
    config = copy.deepcopy(load_config(str(CONFIG)))
    config["geometry"]["num_slices"] = 1
    output_root = tmp_path_factory.mktemp("uih_rtspiral")
    return build_sequence(config, output_root)


def test_generated_sequence_passes_pypulseq_timing(one_slice_result):
    ok, errors = one_slice_result.sequence.check_timing()
    assert ok, errors
    assert one_slice_result.timing_ok is True


def test_one_slice_preserves_350_arms_and_reconstruction_only_frame_metadata(one_slice_result):
    metadata = one_slice_result.metadata

    assert metadata["played_arms_per_slice"] == 350
    assert metadata["total_played_arms"] == 350
    assert metadata["labels"][0] == {"SLC": 0, "LIN": 0}
    assert metadata["labels"][6] == {"SLC": 0, "LIN": 6}
    assert metadata["labels"][7] == {"SLC": 0, "LIN": 7}
    assert metadata["labels"][349] == {"SLC": 0, "LIN": 349}
    assert metadata["default_reconstruction_arms_per_frame"] == 7
    assert metadata["default_reconstruction_frames_per_slice"] == 50
    assert metadata["global_arm_indices"] == list(range(350))
    assert metadata["global_arm_angles_deg"][1] == pytest.approx(222.4969)


def test_real_sequence_sets_slc_once_before_rf_and_writes_no_rep_label(
    one_slice_result,
):
    sequence = one_slice_result.sequence
    blocks = [sequence.get_block(block_id) for block_id in sequence.block_events]
    acquisition_count = 0
    acquisition_blocks = []
    active_slice = None
    slc_events = []

    for block_index, block in enumerate(blocks):
        assert getattr(block, "trigger", None) is None
        label_values = {
            label_event.label: int(label_event.value)
            for label_event in (getattr(block, "label", None) or {}).values()
        }
        assert "REP" not in label_values
        if "SLC" in label_values:
            active_slice = label_values["SLC"]
            slc_events.append((block_index, active_slice))
            assert block.rf is None
            continue
        if block.rf is not None:
            assert active_slice == 0
        if block.adc is None:
            continue
        assert label_values == {"LIN": acquisition_count}
        acquisition_blocks.append(block)
        acquisition_count += 1

    assert slc_events == [(0, 0)]
    assert acquisition_count == 350
    assert acquisition_blocks[0].gx.type == "grad"
    assert [block.adc.phase_offset for block in acquisition_blocks[:3]] == pytest.approx(
        [0.0, np.pi, 0.0]
    )
    assert sequence.duration()[0] == pytest.approx(
        350 * one_slice_result.actual_tr_s
    )


def test_fov_roles_and_written_outputs_are_explicit(one_slice_result):
    assert one_slice_result.sequence.get_definition("FOV") == [0.36, 0.32, 0.006]
    assert one_slice_result.metadata["user_fov_mm"] == [360.0, 320.0]
    assert one_slice_result.metadata["trajectory_design_fov_mm"] == 360.0
    assert one_slice_result.metadata["vds_fixed_ro_fov_cm"] == [36.0]
    assert one_slice_result.metadata["adc_dead_time_s"] == pytest.approx(70e-6)
    assert one_slice_result.metadata["pre_discard_samples"] == 10
    assert one_slice_result.metadata["gradient_delay_s"] == pytest.approx(100e-6)
    assert one_slice_result.sequence_path.is_file()
    assert one_slice_result.trajectory_path.is_file()
    assert one_slice_result.signature
    trajectory = loadmat(one_slice_result.trajectory_path)
    assert trajectory["global_arm_index"].ravel().tolist() == list(range(350))


def test_target_timing_is_never_claimed_when_infeasible(one_slice_result):
    target_te_s = 0.00074
    target_tr_s = 0.00567

    assert one_slice_result.actual_te_s == pytest.approx(0.00156)
    assert one_slice_result.actual_tr_s == pytest.approx(0.00764)
    if (
        one_slice_result.actual_te_s > target_te_s
        or one_slice_result.actual_tr_s > target_tr_s
    ):
        assert one_slice_result.timing_adjustment_reason


def test_upstream_gropt_retry_is_explicit(one_slice_result):
    metadata = one_slice_result.metadata

    assert metadata["rewinder_requested_time_s"] == pytest.approx(0.003)
    assert metadata["rewinder_solver_search_time_s"] == pytest.approx(0.004)
    assert metadata["rewinder_actual_duration_s"] <= metadata[
        "rewinder_solver_search_time_s"
    ]
    assert "UnboundLocalError" in metadata["rewinder_retry_reason"]


def test_all_played_physical_axis_gradients_stay_within_uih_limit(
    one_slice_result,
):
    sequence = one_slice_result.sequence
    max_axis_mT_per_m = 0.0

    for block_id in sequence.block_events:
        block = sequence.get_block(block_id)
        for gradient in (block.gx, block.gy):
            if gradient is None:
                continue
            samples = (
                np.asarray(gradient.waveform)
                if gradient.type == "grad"
                else np.asarray([gradient.amplitude])
            )
            max_axis_mT_per_m = max(
                max_axis_mT_per_m,
                float(np.max(np.abs(samples)) / sequence.system.gamma * 1e3),
            )

    assert max_axis_mT_per_m <= 15.0 + 1e-6


def test_exported_played_base_k_matches_actual_first_arm_direction(
    one_slice_result,
):
    trajectory = loadmat(one_slice_result.trajectory_path)
    predicted = trajectory["base_k_played"]
    actual_k_adc, *_ = one_slice_result.sequence.calculate_kspace()
    actual = actual_k_adc[:2, : predicted.shape[0]].T

    predicted_direction = predicted[-1] / np.linalg.norm(predicted[-1])
    actual_direction = actual[-1] / np.linalg.norm(actual[-1])
    np.testing.assert_allclose(predicted_direction, actual_direction, atol=1e-10)

    angle_rad = np.deg2rad(
        one_slice_result.metadata["global_arm_angles_deg"][1]
    )
    predicted_second = np.asarray(
        [
            np.cos(angle_rad) * predicted[-1, 0]
            - np.sin(angle_rad) * predicted[-1, 1],
            np.sin(angle_rad) * predicted[-1, 0]
            + np.cos(angle_rad) * predicted[-1, 1],
        ]
    )
    second_endpoint = actual_k_adc[:2, 2 * predicted.shape[0] - 1]
    np.testing.assert_allclose(
        predicted_second / np.linalg.norm(predicted_second),
        second_endpoint / np.linalg.norm(second_endpoint),
        atol=1e-10,
    )
    assert one_slice_result.metadata["rewinder_base_rotation_deg"] == pytest.approx(
        float(trajectory["rewinder_base_rotation_deg"].item())
    )
