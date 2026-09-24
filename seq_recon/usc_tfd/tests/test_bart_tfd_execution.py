from pathlib import Path
import json
import sys

import numpy as np
import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

import reconstruct_t13_slice0_tfd_bart as runner
import check_bart_environment as environment
from bart_tfd import ScaleEstimate, pics_command, select_usc_scale


class FakeBart:
    def __init__(self, bad_pics_shape: bool = False):
        self.calls = []
        self.bad_pics_shape = bad_pics_shape

    def bart(self, nargout, command, *args):
        self.calls.append(command)
        if command.startswith("nlinv "):
            return np.zeros((32, 32, 1, 1), np.complex64), np.ones((32, 32, 1, 2), np.complex64)
        if command.startswith("resize -c 0 720"):
            return np.ones((720, 720, 1, 2), np.complex64)
        if command.startswith("resize -c 0 360"):
            return np.ones((360, 360, 1, 2), np.complex64)
        if command == "normalize 8":
            return args[0]
        if command.startswith("nufft -g -x 360:360:1 -a"):
            return np.ones((360, 360, 1, 2, 1, 1, 1, 1, 1, 1, 1), np.complex64)
        if command.startswith("pics "):
            frames = 49 if self.bad_pics_shape else 50
            return np.ones((360, 360, 1, 1, 1, 1, 1, 1, 1, 1, frames), np.complex64)
        raise AssertionError(command)


def _prepared_npz(tmp_path: Path) -> tuple[Path, Path]:
    kspace = np.ones((50, 7, 2, 1250), dtype=np.complex64)
    trajectory = np.zeros((50, 7, 1250, 2), dtype=np.float32)
    npz = tmp_path / "input.npz"
    np.savez_compressed(npz, kspace_frame_arm_coil_sample=kspace, trajectory_frame_arm_sample_kx_negative_ky=trajectory)
    summary = tmp_path / "input_summary.json"
    summary.write_text(json.dumps({"number_of_frames": 50}), encoding="utf-8")
    return npz, summary


def test_scale_statistics_follow_usc_rule_and_pics_uses_selected_value_directly():
    estimate = ScaleEstimate(median=2.0, p90=4.0, maximum=7.0, selected=7.0)
    assert estimate.selected == 7.0
    assert "-w 7.0" in pics_command(estimate.selected, 0.01, 120)
    assert select_usc_scale(1.0, 2.0, 3.0) == 2.0
    assert select_usc_scale(1.0, 2.0, 5.0) == 5.0


def test_execute_pipeline_keeps_native_complex_and_uses_source_order(tmp_path, monkeypatch):
    npz, input_summary = _prepared_npz(tmp_path)
    fake = FakeBart()
    monkeypatch.setattr(runner.np, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_save_qc_images", lambda *args, **kwargs: None)
    config = MODULE_ROOT / "configs" / "t13_5_slice0.toml"

    summary = runner.execute_reconstruction(npz, input_summary, config, tmp_path / "out", fake, "0")

    assert summary["native_squeezed_shape"] == [360, 360, 50]
    assert summary["native_crop_shape"] == [240, 213, 50]
    assert summary["usc_display_shape"] == [213, 240, 50]
    assert summary["selected_scale"] == pytest.approx(2.0)
    assert summary["selected_scale_pics_note"] == "selected scale is passed directly to pics -w; no reciprocal"
    nlinv_index = next(i for i, command in enumerate(fake.calls) if command.startswith("nlinv "))
    scale_index = next(i for i, command in enumerate(fake.calls) if command.startswith("nufft -g -x"))
    pics_index = next(i for i, command in enumerate(fake.calls) if command.startswith("pics "))
    assert nlinv_index < scale_index < pics_index
    assert fake.calls[nlinv_index] == "nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t"
    assert "-w 2.0" in fake.calls[pics_index]


def test_execute_rejects_unexpected_squeezed_pics_dimensions(tmp_path, monkeypatch):
    npz, input_summary = _prepared_npz(tmp_path)
    monkeypatch.setattr(runner.np, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_save_qc_images", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="raw BART PICS output shape"):
        runner.execute_reconstruction(
            npz, input_summary, MODULE_ROOT / "configs" / "t13_5_slice0.toml", tmp_path / "out", FakeBart(True), "0"
        )


def test_environment_checker_uses_nlinv_parser_probe_for_hidden_a_b(monkeypatch):
    def fake_run(command):
        assert command[-5:] == ["-a", "32", "-b", "16", "-h"]
        return 0, "nlinv help"

    monkeypatch.setattr(environment, "_run", fake_run)
    result = environment._nlinv_hidden_option_capabilities("/fake/bart")
    assert result["-a"]["status"] == "PASS"
    assert result["-b"]["status"] == "PASS"
