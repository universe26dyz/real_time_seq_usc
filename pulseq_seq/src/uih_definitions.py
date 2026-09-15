"""Deterministic UIH labels, geometry, and Pulseq definitions."""


def realtime_labels(
    slice_idx: int, arm_idx_in_slice: int, arms_per_frame: int
) -> tuple[int, int, int]:
    """Return the UIH SLC, REP, and LIN labels for one played arm."""
    if arms_per_frame <= 0:
        raise ValueError("arms_per_frame must be positive")
    return slice_idx, arm_idx_in_slice // arms_per_frame, arm_idx_in_slice % arms_per_frame


def slice_positions_m(num_slices: int, shift_mm: float) -> list[float]:
    """Return centre-symmetric slice centres in metres."""
    if isinstance(num_slices, bool) or not isinstance(num_slices, int) or num_slices <= 0:
        raise ValueError("num_slices must be a positive integer")
    if isinstance(shift_mm, bool) or not isinstance(shift_mm, (int, float)) or shift_mm <= 0:
        raise ValueError("shift_mm must be positive")

    # 以 stack 中心为零点，保留重叠层的真实物理中心位置。
    shift_m = shift_mm * 1e-3
    centre = (num_slices - 1) / 2
    return [round((index - centre) * shift_m, 12) for index in range(num_slices)]


def apply_uih_definitions(
    seq, config: dict, actual_te_s: float, actual_tr_s: float, positions_m: list[float]
) -> None:
    """Write the official UIH 2D definitions using SI units."""
    if not hasattr(seq, "set_definition"):
        raise TypeError("seq must expose set_definition")

    geometry = config["geometry"]
    fov_mm = geometry["fov_mm"]
    thickness_m = geometry["slice_thickness_mm"] * 1e-3
    # Pulseq 的 FOV、厚度、位置和时序 Definition 一律使用 SI 单位。
    definitions = {
        "Dimension": config["uih_definitions"]["dimension"],
        "FOV": [fov_mm[0] * 1e-3, fov_mm[1] * 1e-3, thickness_m],
        "SliceNumber": geometry["num_slices"],
        "SliceThickness": thickness_m,
        "Center": [0.0, 0.0, 0.0],
        "SlicePositions": positions_m,
        "Name": config["project"]["name"],
        "TE": actual_te_s,
        "TR": actual_tr_s,
        "FA": config["rf"]["flip_angle_deg"],
    }
    for key, value in definitions.items():
        seq.set_definition(key, value)
