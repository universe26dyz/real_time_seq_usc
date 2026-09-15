# UIH uMR 790 Real-time Spiral bSSFP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a UIH-compatible, fixed-slice 2D multi-slice real-time golden-angle spiral bSSFP Pulseq sequence and its fully traceable artifacts.

**Architecture:** Preserve the USC `write_rtspiral_svr.py` algorithm and its local `libspiral`/`gropt` dependencies in `third_party`; place UIH-specific configuration, definitions, labels, reporting, and output logic in `src`.  The generator consumes one JSON configuration and writes a `.seq`, trajectory `.mat`, resolved JSON, and explicit timing/parameter-gap reports.

**Tech Stack:** Python 3, PyPulseq, NumPy, SciPy, USC `libspiral`, USC `gropt`, pytest.

**Spec:** `/home/universe/Documents/ChatGPT/multimap_svr/real_time_seq/Codex_Prompt_UIH_RTSpiral_Multislice_RealTime_SEQ_2026-09-15.md`

## Global Constraints

- Preserve USC trajectory generation, arbitrary gradients, M1-nulled `gropt` rewinder, golden-angle rotation, trueFISP cycling, timing check, and MATLAB trajectory export.
- Do not edit any file under `third_party/usc_rtspiral_pypulseq`.
- Use UIH Opts: 15 mT/m, 32 T/m/s, gradient raster 10 us, RF raster 1 us, RF dead time 400 us, ADC dead time 70 us.
- Use `FOV = [360, 320] mm` (readout, phase), resolution 1.5 mm, thickness 6 mm, shift 2 mm, 60 configurable slices.
- Play 350 global golden-angle arms consecutively for each slice: 7 arms/frame and 50 frames/slice.  Labels must be `SLC=slice`, `REP=frame`, `LIN=arm-within-frame`.
- Targets remain TE/TR 0.74/5.67 ms and FA 100 degrees.  Report, rather than hide, an infeasible target timing.
- ECG, respiratory triggering, and external TTL are all disabled.

---

### Task 1: Create configuration and deterministic helpers

**Files:**
- Create: `config/uih790_rtspiral_realtime.json`
- Create: `src/config_loader.py`
- Create: `src/uih_definitions.py`
- Create: `tests/test_config.py`
- Create: `tests/test_realtime_labels.py`

**Interfaces:**
- `load_config(path: str) -> dict` validates required JSON values.
- `realtime_labels(slice_idx: int, arm_idx_in_slice: int, arms_per_frame: int) -> tuple[int, int, int]` returns `(SLC, REP, LIN)`.
- `slice_positions_m(num_slices: int, shift_mm: float) -> list[float]` returns center-symmetric physical locations.
- `apply_uih_definitions(seq, config, actual_te_s, actual_tr_s, positions_m) -> None` writes UIH keys.

- [ ] **Step 1: Write failing configuration and label tests**

```python
def test_realtime_label_mapping_at_frame_boundaries():
    assert realtime_labels(0, 0, 7) == (0, 0, 0)
    assert realtime_labels(0, 6, 7) == (0, 0, 6)
    assert realtime_labels(0, 7, 7) == (0, 1, 0)
    assert realtime_labels(0, 349, 7) == (0, 49, 6)
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/test_config.py tests/test_realtime_labels.py -v`

- [ ] **Step 3: Implement the smallest JSON loader and helper API**

```python
def realtime_labels(slice_idx, arm_idx_in_slice, arms_per_frame):
    return slice_idx, arm_idx_in_slice // arms_per_frame, arm_idx_in_slice % arms_per_frame
```

- [ ] **Step 4: Run the helper tests and confirm GREEN**

Run: `python -m pytest tests/test_config.py tests/test_realtime_labels.py -v`

### Task 2: Adapt USC generator with UIH constraints

**Files:**
- Create: `src/write_rtspiral_svr_uih.py`
- Create: `scripts/generate_uih790_rtspiral.sh`
- Create: `tests/test_timing.py`

**Interfaces:**
- `build_sequence(config: dict, output_root: Path) -> GenerationResult` constructs the sequence, raises on timing failure, and returns actual timing, paths, signature, and labels metadata.
- `GenerationResult` includes `sequence`, `actual_te_s`, `actual_tr_s`, `sequence_path`, `trajectory_path`, `signature`, and `timing_ok`.

- [ ] **Step 1: Write a failing timing test**

```python
def test_generated_sequence_passes_pypulseq_timing(tmp_path):
    result = build_sequence(load_config(CONFIG), tmp_path)
    ok, errors = result.sequence.check_timing()
    assert ok, errors
```

- [ ] **Step 2: Run the test and confirm RED because generator is absent**

Run: `python -m pytest tests/test_timing.py -v`

- [ ] **Step 3: Copy the minimum needed USC execution path and adapt it**

Implement UIH `Opts`, no trigger, explicit 350-arm fixed-slice loop, 350 rotated gradients based on the unmodified base arm, trueFISP cycling, label blocks before every acquisition, UIH definitions, and no Siemens slab-thickness workaround.

- [ ] **Step 4: Run timing test and resolve only explicit feasibility errors**

Run: `python -m pytest tests/test_timing.py -v`

### Task 3: Generate traceable outputs and reports

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `third_party/usc_rtspiral_pypulseq/README_SOURCE.md`
- Create: `third_party/pulseq_systems/README_SOURCE.md`
- Create: `reports/PARAMETER_GAPS.md`
- Generated: `out_seq/*.seq`, `out_trajectory/*.mat`, `reports/resolved_config.json`, `reports/TIMING_REPORT.md`, `reports/GENERATION_REPORT.md`

**Interfaces:**
- `write_reports(result, config, output_root) -> None` records target versus actual timing, scan duration, signature, paths, and timing outcome.

- [ ] **Step 1: Write a failing report-artifact test**

```python
def test_generation_writes_resolved_config_and_reports(tmp_path):
    result = build_sequence(load_config(CONFIG), tmp_path)
    assert (tmp_path / 'reports' / 'resolved_config.json').is_file()
    assert result.sequence_path.is_file()
    assert result.trajectory_path.is_file()
```

- [ ] **Step 2: Run it and confirm RED**

Run: `python -m pytest tests/test_timing.py -v`

- [ ] **Step 3: Implement trajectory metadata and reports**

Use `scipy.io.savemat` to include base spiral samples, per-arm global index and angle, labeling relation, FOV, geometry, slices, target/actual timing, and Pulseq signature.  State FOV is user-provided (360 x 320 mm), so no FOV gap remains.

- [ ] **Step 4: Run all tests, then generate the actual deliverables**

Run: `python -m pytest tests -v && bash scripts/generate_uih790_rtspiral.sh`

- [ ] **Step 5: Verify output integrity**

Run: `python -m pytest tests -v && ls -lh out_seq out_trajectory reports`

### Task 4: Final verification

**Files:**
- Verify: all generated outputs and reports.

- [ ] **Step 1: Inspect timing report and resolved configuration**

Confirm target and actual TE/TR are separate values, labels use the required ranges, total duration equals `num_slices * 350 * actual_TR`, and `.seq` size is reported.

- [ ] **Step 2: Check source provenance**

Run: `git -C third_party/usc_rtspiral_pypulseq rev-parse HEAD && rg -n 'uMR 790' third_party/pulseq_systems/MRSystems.json`

- [ ] **Step 3: Record any unresolved scanner-side checks**

Keep SAR/PNS and AIDE Virtual Scan validation as explicit next steps; never claim them complete locally.
