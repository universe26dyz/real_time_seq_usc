"""Task 1 configuration and UIH definition contracts."""

import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config_loader import load_config
from uih_definitions import apply_uih_definitions, slice_positions_m


CONFIG = PROJECT_ROOT / "config" / "uih790_rtspiral_realtime.json"


class DefinitionRecorder:
    def __init__(self):
        self.values = {}

    def set_definition(self, key, value):
        self.values[key] = value


def test_load_config_preserves_uih_protocol_values_and_metadata():
    config = load_config(str(CONFIG))

    assert config["geometry"]["fov_mm"] == [360.0, 320.0]
    assert config["spiral"]["trajectory_design_fov_mm"] == 360.0
    assert config["spiral"]["trajectory_design_fov_mm"] == max(
        config["geometry"]["fov_mm"]
    )
    assert config["geometry"]["matrix"] == [240, 213]
    assert config["uih_definitions"]["write_matrix"] is True
    assert config["uih_definitions"]["write_resolution"] is True
    assert config["spiral"]["parameter_metadata"]["trajectory_design_fov_mm"] == {
        "unit": "mm",
        "source": "derived from max(geometry.fov_mm)",
        "note": "Radial FOV profile for vds_fixed_ro; not a paper parameter.",
    }
    assert config["geometry"]["parameter_metadata"]["fov_mm"] == {
        "unit": "mm",
        "source": "user_specified",
        "note": "Readout and phase FOV supplied by the user.",
    }
    assert config["realtime"]["arms_per_frame"] == 7
    assert config["realtime"]["frames_per_slice"] == 50
    assert config["realtime"]["trs_per_slice"] == 350
    assert config["triggers"]["physio_trigger_enabled"] is False
    assert config["triggers"]["external_ttl_enabled"] is False


def test_load_config_rejects_inconsistent_realtime_arm_count(tmp_path):
    config_text = CONFIG.read_text(encoding="utf-8").replace(
        '"trs_per_slice": 350', '"trs_per_slice": 349'
    )
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match="trs_per_slice"):
        load_config(str(invalid_path))


def _write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("spiral", "contrast", None),
        ("spiral", "slew_ratio", "0.9"),
        ("rf", "rf_duration_s", 0),
        ("target_timing", "frame_time_ms", None),
        ("triggers", "external_ttl_enabled", "false"),
        ("labels", "arm_in_frame_label", ""),
        ("output", "seq_dir", ""),
    ],
)
def test_load_config_rejects_missing_or_malformed_nested_seq_setting(
    tmp_path, section, key, value
):
    config = load_config(str(CONFIG))
    if value is None:
        del config[section][key]
    else:
        config[section][key] = value

    with pytest.raises(ValueError, match=f"{section}.{key}"):
        load_config(str(_write_config(tmp_path, config)))


@pytest.mark.parametrize("metadata_key", ["unit", "source", "note"])
def test_load_config_rejects_incomplete_fov_metadata(tmp_path, metadata_key):
    config = load_config(str(CONFIG))
    del config["geometry"]["parameter_metadata"]["fov_mm"][metadata_key]

    with pytest.raises(ValueError, match="geometry.fov_mm metadata"):
        load_config(str(_write_config(tmp_path, config)))


def test_load_config_rejects_incomplete_field_strength_metadata(tmp_path):
    config = load_config(str(CONFIG))
    config["project"]["parameter_metadata"]["field_strength_T"] = {
        "unit": "T",
        "source": "uMR 790 capability reference",
    }

    with pytest.raises(ValueError, match="project.field_strength_T metadata"):
        load_config(str(_write_config(tmp_path, config)))


def test_load_config_rejects_trajectory_design_fov_not_derived_from_user_fov(
    tmp_path,
):
    config = load_config(str(CONFIG))
    config["spiral"]["trajectory_design_fov_mm"] = 320.0

    with pytest.raises(ValueError, match="trajectory_design_fov_mm"):
        load_config(str(_write_config(tmp_path, config)))


def test_apply_uih_definitions_writes_official_keys_in_si_units():
    config = load_config(str(CONFIG))
    sequence = DefinitionRecorder()
    positions_m = slice_positions_m(3, 2.0)

    apply_uih_definitions(sequence, config, 0.001, 0.006, positions_m)

    assert sequence.values == {
        "Dimension": 2,
        "FOV": [0.36, 0.32, 0.006],
        "SliceNumber": 60,
        "SliceThickness": 0.006,
        "Matrix": [240, 213],
        "Resolution": [240, 213],
        "Center": [120.0, 106.5],
        "SlicePositions": [-0.002, 0.0, 0.002],
        "Name": "UIH_RTSpiral_RealTime",
        "TE": 0.001,
        "TR": 0.006,
        "FA": 100.0,
    }
