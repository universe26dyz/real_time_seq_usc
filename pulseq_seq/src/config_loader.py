"""Load and validate the single UIH real-time spiral JSON configuration."""

import json
import math
from pathlib import Path


_REQUIRED_SECTIONS = (
    "project", "provenance", "scanner_system", "spiral", "rf", "target_timing",
    "realtime", "acquisition", "reconstruction_defaults", "geometry", "triggers", "labels", "uih_definitions", "output",
)
_METADATA_FIELDS = ("unit", "source", "note")
_METADATA_PARAMETERS = {
    "project": ("name", "field_strength_T"),
    "scanner_system": ("max_grad_mT_per_m", "max_slew_T_per_m_per_s", "grad_raster_time_s", "rf_raster_time_s", "rf_ringdown_time_s", "rf_dead_time_s", "adc_dead_time_s"),
    "spiral": ("contrast", "arm_ordering", "ga_angle_deg", "slew_ratio", "readout_duration_s", "adc_dwell_s", "rewinder_method", "rewinder_time_s", "m1_nulling", "rotate_grads", "trajectory_design_fov_mm"),
    "rf": ("flip_angle_deg", "rf_duration_s", "time_bandwidth_product", "fa_schedule_enabled"),
    "target_timing": ("te_ms", "tr_ms", "frame_time_ms", "slice_dwell_time_s"),
    "realtime": ("arms_per_frame", "frames_per_slice", "trs_per_slice", "expected_frame_time_ms_at_target_tr", "expected_slice_time_s_at_target_tr"),
    "acquisition": ("arms_per_slice",),
    "reconstruction_defaults": ("arms_per_frame",),
    "geometry": ("inplane_resolution_mm", "slice_thickness_mm", "slice_shift_mm", "num_slices", "typical_num_slices_range", "fov_mm", "matrix"),
    "triggers": ("physio_trigger_enabled", "external_ttl_enabled"),
    "labels": ("slice_label", "frame_label", "arm_in_frame_label"),
    "uih_definitions": ("dimension", "write_slice_positions", "write_center", "write_matrix", "write_resolution"),
    "output": ("write_seq", "write_trajectory_mat", "write_resolved_json", "write_generation_report", "seq_dir", "trajectory_dir", "report_dir"),
}


def load_config(path: str) -> dict:
    """Return a validated configuration from *path*."""
    try:
        with Path(path).open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid configuration file: {path}") from error
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    for section_name in _REQUIRED_SECTIONS:
        _require_mapping(config, section_name)

    _validate_project(config["project"])
    _validate_provenance(config["provenance"])
    _validate_scanner(config["scanner_system"])
    _validate_spiral(config["spiral"])
    _validate_rf(config["rf"])
    _validate_timing(config["target_timing"])
    _validate_realtime(config["realtime"])
    _validate_acquisition(config["acquisition"])
    _validate_reconstruction_defaults(config["reconstruction_defaults"])
    _validate_geometry(config["geometry"])
    _validate_triggers(config["triggers"])
    _validate_labels(config["labels"])
    _validate_definitions(config["uih_definitions"])
    _validate_output(config["output"])
    _validate_metadata(config)

    realtime = config["realtime"]
    # 保证每层的连续采集臂数能由 frame 标签完整表达。
    if realtime["trs_per_slice"] != realtime["arms_per_frame"] * realtime["frames_per_slice"]:
        raise ValueError("realtime.trs_per_slice must equal arms_per_frame * frames_per_slice")
    if config["geometry"]["parameter_metadata"]["fov_mm"]["unit"] != "mm":
        raise ValueError("geometry.fov_mm metadata unit must be mm")
    # vds_fixed_ro 接收径向 FOV profile；它不是二维 Cartesian FOV。
    if not math.isclose(
        config["spiral"]["trajectory_design_fov_mm"],
        max(config["geometry"]["fov_mm"]),
    ):
        raise ValueError(
            "spiral.trajectory_design_fov_mm must equal max(geometry.fov_mm)"
        )


def _validate_project(section: dict) -> None:
    for key in ("name", "purpose", "scanner_target", "config_version"):
        _require_string(section, key, "project")
    _require_positive_number(section, "field_strength_T", "project")


def _validate_provenance(section: dict) -> None:
    for key in ("primary_sequence_repo", "primary_sequence_commit", "primary_source_file", "reference_protocol_file", "uih_system_repo", "uih_system_commit", "uih_system_file", "uih_manual"):
        _require_string(section, key, "provenance")


def _validate_scanner(section: dict) -> None:
    for key in ("max_grad_mT_per_m", "max_slew_T_per_m_per_s", "grad_raster_time_s", "rf_raster_time_s", "rf_ringdown_time_s", "rf_dead_time_s", "adc_dead_time_s"):
        _require_positive_number(section, key, "scanner_system")


def _validate_spiral(section: dict) -> None:
    for key in ("contrast", "arm_ordering", "rewinder_method"):
        _require_string(section, key, "spiral")
    _require_positive_number(section, "ga_angle_deg", "spiral")
    if _require_positive_number(section, "slew_ratio", "spiral") > 1:
        raise ValueError("spiral.slew_ratio must not exceed 1")
    for key in (
        "readout_duration_s",
        "adc_dwell_s",
        "rewinder_time_s",
        "trajectory_design_fov_mm",
    ):
        _require_positive_number(section, key, "spiral")
    for key in ("m1_nulling", "rotate_grads"):
        _require_bool(section, key, "spiral")


def _validate_rf(section: dict) -> None:
    for key in ("flip_angle_deg", "rf_duration_s", "time_bandwidth_product"):
        _require_positive_number(section, key, "rf")
    _require_bool(section, "fa_schedule_enabled", "rf")


def _validate_timing(section: dict) -> None:
    for key in ("te_ms", "tr_ms", "frame_time_ms", "slice_dwell_time_s"):
        _require_positive_number(section, key, "target_timing")


def _validate_realtime(section: dict) -> None:
    for key in ("arms_per_frame", "frames_per_slice", "trs_per_slice"):
        _require_positive_int(section, key, "realtime")
    for key in ("expected_frame_time_ms_at_target_tr", "expected_slice_time_s_at_target_tr"):
        _require_positive_number(section, key, "realtime")


def _validate_acquisition(section: dict) -> None:
    _require_positive_int(section, "arms_per_slice", "acquisition")


def _validate_reconstruction_defaults(section: dict) -> None:
    _require_positive_int(section, "arms_per_frame", "reconstruction_defaults")


def _validate_geometry(section: dict) -> None:
    _require_two_positive_numbers(section, "inplane_resolution_mm", "geometry")
    for key in ("slice_thickness_mm", "slice_shift_mm"):
        _require_positive_number(section, key, "geometry")
    _require_positive_int(section, "num_slices", "geometry")
    bounds = _require_two_positive_ints(section, "typical_num_slices_range", "geometry")
    if bounds[0] > bounds[1]:
        raise ValueError("geometry.typical_num_slices_range must be ascending")
    _require_two_positive_numbers(section, "fov_mm", "geometry")
    if section.get("matrix") is not None:
        _require_two_positive_ints(section, "matrix", "geometry")


def _validate_triggers(section: dict) -> None:
    for key in ("physio_trigger_enabled", "external_ttl_enabled"):
        _require_bool(section, key, "triggers")


def _validate_labels(section: dict) -> None:
    for key in ("slice_label", "frame_label", "arm_in_frame_label"):
        _require_string(section, key, "labels")


def _validate_definitions(section: dict) -> None:
    if _require_positive_int(section, "dimension", "uih_definitions") != 2:
        raise ValueError("uih_definitions.dimension must be 2")
    for key in ("write_slice_positions", "write_center", "write_matrix", "write_resolution"):
        _require_bool(section, key, "uih_definitions")


def _validate_output(section: dict) -> None:
    for key in ("write_seq", "write_trajectory_mat", "write_resolved_json", "write_generation_report"):
        _require_bool(section, key, "output")
    for key in ("seq_dir", "trajectory_dir", "report_dir"):
        _require_string(section, key, "output")


def _validate_metadata(config: dict) -> None:
    for section_name, parameter_names in _METADATA_PARAMETERS.items():
        metadata = _require_mapping(config[section_name], "parameter_metadata", section_name)
        for parameter_name in parameter_names:
            entry = _require_mapping(metadata, parameter_name, f"{section_name}.parameter_metadata")
            for field in _METADATA_FIELDS:
                _require_string(entry, field, f"{section_name}.{parameter_name} metadata")


def _require_mapping(section: dict, key: str, path: str = "configuration") -> dict:
    if not isinstance(section, dict) or not isinstance(section.get(key), dict):
        raise ValueError(f"{path}.{key} must be an object")
    return section[key]


def _require_string(section: dict, key: str, path: str) -> str:
    value = section.get(key) if isinstance(section, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _require_bool(section: dict, key: str, path: str) -> bool:
    value = section.get(key) if isinstance(section, dict) else None
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be a boolean")
    return value


def _require_positive_number(section: dict, key: str, path: str) -> float:
    value = section.get(key) if isinstance(section, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive number")
    return value


def _require_positive_int(section: dict, key: str, path: str) -> int:
    value = section.get(key) if isinstance(section, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer")
    return value


def _require_two_positive_numbers(section: dict, key: str, path: str) -> list[float]:
    value = section.get(key) if isinstance(section, dict) else None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path}.{key} must contain exactly two values")
    if any(isinstance(value_item, bool) or not isinstance(value_item, (int, float)) or not math.isfinite(value_item) or value_item <= 0 for value_item in value):
        raise ValueError(f"{path}.{key} must contain positive numbers")
    return value


def _require_two_positive_ints(section: dict, key: str, path: str) -> list[int]:
    value = section.get(key) if isinstance(section, dict) else None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path}.{key} must contain exactly two integer values")
    if any(isinstance(value_item, bool) or not isinstance(value_item, int) or value_item <= 0 for value_item in value):
        raise ValueError(f"{path}.{key} must contain positive integers")
    return value
