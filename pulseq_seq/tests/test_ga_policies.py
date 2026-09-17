"""Targeted T13--T15 acquisition and trajectory-policy contracts."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_loader import load_config
from uih_definitions import apply_uih_definitions, slice_positions_m
from write_rtspiral_svr_uih import build_sequence

CONFIG = PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json"


class DefinitionRecorder:
    def __init__(self):
        self.values = {}

    def set_definition(self, key, value):
        self.values[key] = value


@pytest.fixture(scope="module", params=["usc144_lut_global_wrap", "lut350_slice_reset", "continuous_global_ga"])
def policy_result(tmp_path_factory, request):
    config = copy.deepcopy(load_config(str(CONFIG)))
    config["geometry"]["num_slices"] = 2
    return request.param, build_sequence(
        config,
        tmp_path_factory.mktemp(request.param),
        ga_policy=request.param,
        sequence_filename=f"{request.param}.seq",
    )


def test_physical_support_fov_z_for_five_and_sixty_slices():
    config = load_config(str(CONFIG))
    for num_slices, expected_positions, expected_fov_z in (
        (5, [-0.004, -0.002, 0.0, 0.002, 0.004], 0.014),
        (60, [-0.059, -0.057, 0.057, 0.059], 0.124),
    ):
        config["geometry"]["num_slices"] = num_slices
        positions = slice_positions_m(num_slices, 2.0)
        recorder = DefinitionRecorder()
        apply_uih_definitions(recorder, config, 0.00156, 0.00764, positions)
        assert recorder.values["FOV"] == [0.36, 0.32, expected_fov_z]
        assert recorder.values["SliceThickness"] == 0.006
        if num_slices == 5:
            assert positions == expected_positions
        else:
            assert positions[:2] + positions[-2:] == expected_positions


def test_common_policy_contract(policy_result):
    policy, result = policy_result
    metadata = result.metadata
    assert result.timing_ok
    assert metadata["played_arms_per_slice"] == 350
    assert metadata["default_reconstruction_arms_per_frame"] == 7
    assert metadata["total_played_arms"] == 700
    assert all("REP" not in label for label in metadata["labels"])
    assert metadata["maximum_played_axis_gradient_mT_per_m"] < 15.0
    assert policy == metadata["ga_policy"]


def test_t13_global_144_wrap(policy_result):
    policy, result = policy_result
    if policy != "usc144_lut_global_wrap":
        pytest.skip("T13-only contract")
    metadata = result.metadata
    assert metadata["ga_lut_length"] == 144
    assert metadata["slice_reset"] is False
    assert metadata["trajectory_index_per_acq"][142:146] == [142, 143, 0, 1]
    assert metadata["trajectory_index_per_acq"][350] == 62


def test_t14_slice_reset_350_lut(policy_result):
    policy, result = policy_result
    if policy != "lut350_slice_reset":
        pytest.skip("T14-only contract")
    metadata = result.metadata
    assert metadata["ga_lut_length"] == 350
    assert metadata["slice_reset"] is True
    assert metadata["trajectory_index_per_acq"][:350] == list(range(350))
    assert metadata["trajectory_index_per_acq"][350:] == list(range(350))
    np.testing.assert_allclose(metadata["global_arm_angles_deg"][:350], metadata["global_arm_angles_deg"][350:])


def test_t15_continuous_global_angle_with_local_lin(policy_result):
    policy, result = policy_result
    if policy != "continuous_global_ga":
        pytest.skip("T15-only contract")
    metadata = result.metadata
    assert metadata["ga_lut_length"] is None
    assert metadata["labels"][349]["LIN"] == 349
    assert metadata["labels"][350]["LIN"] == 0
    assert metadata["global_arm_angles_deg"][350] == pytest.approx((350 * 222.4969) % 360)
