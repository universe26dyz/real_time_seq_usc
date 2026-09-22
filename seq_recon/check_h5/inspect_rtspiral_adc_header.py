#!/usr/bin/env python3
"""
Inspect ADC-header timing fields in a UIH->ISMRMRD H5 file.

READ-ONLY. Intended for the USC RT-Spiral / CR-DREME project.

It reports fields that are critical for resolving UIH's mandatory 2x ADC
oversampling before NUFFT reconstruction:
  - number_of_samples
  - sample_time_us
  - discard_pre / discard_post
  - center_sample
  - scan_counter and its delta histogram
  - acquisition_time_stamp and its delta histogram
  - available/active channels
  - data shape
  - selected user_int / user_float fields when exposed

Example:
  python inspect_rtspiral_adc_header.py \
    --h5 /media/universe/DATA/lab/SVR/data/20260917_real_time_seq_usc/h5/UID_7685991886275844011_pulseq_T13_5/testdata.h5 \
    --out /home/universe/SVR/real_time_seq_usc/seq_recon/check_h5/preflight_T13_5/adc_header_detail.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import ismrmrd
import numpy as np


def scalar(x, default=None):
    try:
        if hasattr(x, "item"):
            return x.item()
        return x
    except Exception:
        return default


def hist(values):
    c = Counter(values)
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


def delta_hist(values):
    arr = np.asarray(values, dtype=np.int64)
    if len(arr) < 2:
        return {}
    d = np.diff(arr)
    return {str(int(k)): int(v) for k, v in sorted(Counter(d.tolist()).items())}


def candidate_groups(path: Path):
    out = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Group):
                score = int("data" in obj) * 3 + int("xml" in obj) * 3
                out.append((score, key))
    return [x[1] for x in sorted(out, reverse=True)]


def n_acq(ds):
    for name in ("number_of_acquisitions", "getNumberOfAcquisitions"):
        fn = getattr(ds, name, None)
        if fn:
            return int(fn())
    raise RuntimeError("Cannot find ISMRMRD acquisition-count API")


def read_acq(ds, i):
    for name in ("read_acquisition", "readAcquisition"):
        fn = getattr(ds, name, None)
        if fn:
            return fn(i)
    raise RuntimeError("Cannot find ISMRMRD acquisition-read API")


def open_ds(path: Path):
    names = candidate_groups(path)
    if "dataset" in names:
        names = ["dataset"] + [n for n in names if n != "dataset"]
    if not names:
        names = ["dataset"]
    errors = {}
    for name in names:
        try:
            ds = ismrmrd.Dataset(str(path), name, create_if_needed=False)
            _ = n_acq(ds)
            return ds, name, errors
        except Exception as e:
            errors[name] = repr(e)
    raise RuntimeError(errors)


def array_field(acq, names):
    for name in names:
        if hasattr(acq, name):
            try:
                x = np.asarray(getattr(acq, name))
                return x.tolist()
            except Exception:
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--actual-tr-ms", type=float, default=7.64)
    args = ap.parse_args()

    path = args.h5.expanduser().resolve()
    ds, group, open_errors = open_ds(path)
    n = n_acq(ds)

    fields = {
        "number_of_samples": [],
        "sample_time_us": [],
        "discard_pre": [],
        "discard_post": [],
        "center_sample": [],
        "available_channels": [],
        "active_channels": [],
        "scan_counter": [],
        "acquisition_time_stamp": [],
    }

    selected = {}
    selected_indices = sorted(set([0, 1, 2, max(0, n // 2), max(0, n - 3), max(0, n - 2), max(0, n - 1)]))

    for i in range(n):
        a = read_acq(ds, i)
        for key in fields:
            fields[key].append(scalar(getattr(a, key, None)))
        if i in selected_indices:
            selected[str(i)] = {
                "number_of_samples": scalar(getattr(a, "number_of_samples", None)),
                "sample_time_us": scalar(getattr(a, "sample_time_us", None)),
                "discard_pre": scalar(getattr(a, "discard_pre", None)),
                "discard_post": scalar(getattr(a, "discard_post", None)),
                "center_sample": scalar(getattr(a, "center_sample", None)),
                "available_channels": scalar(getattr(a, "available_channels", None)),
                "active_channels": scalar(getattr(a, "active_channels", None)),
                "scan_counter": scalar(getattr(a, "scan_counter", None)),
                "acquisition_time_stamp": scalar(getattr(a, "acquisition_time_stamp", None)),
                "slice": int(a.idx.slice),
                "lin": int(a.idx.kspace_encode_step_1),
                "repetition": int(a.idx.repetition),
                "trajectory_dimensions": scalar(getattr(a, "trajectory_dimensions", None)),
                "data_shape": list(np.asarray(a.data).shape),
                "user_int": array_field(a, ("user_int",)),
                "user_float": array_field(a, ("user_float",)),
                "physiology_time_stamp": array_field(a, ("physiology_time_stamp",)),
            }

    scan = [int(x) for x in fields["scan_counter"] if x is not None]
    ts = [int(x) for x in fields["acquisition_time_stamp"] if x is not None]
    scan_d = np.diff(np.asarray(scan, dtype=np.int64)) if len(scan) > 1 else np.array([])
    ts_d = np.diff(np.asarray(ts, dtype=np.int64)) if len(ts) > 1 else np.array([])

    result = {
        "h5": str(path),
        "ismrmrd_group": group,
        "open_errors_before_success": open_errors,
        "n_acquisitions": n,
        "field_histograms": {k: hist(v) for k, v in fields.items()},
        "scan_counter_delta_histogram": delta_hist(scan),
        "acquisition_timestamp_delta_histogram": delta_hist(ts),
        "scan_counter_delta_median": float(np.median(scan_d)) if len(scan_d) else None,
        "timestamp_delta_median": float(np.median(ts_d)) if len(ts_d) else None,
        "actual_tr_ms_given": args.actual_tr_ms,
        "inferred_ms_per_scan_counter_unit_from_median": (
            args.actual_tr_ms / float(np.median(scan_d))
            if len(scan_d) and np.median(scan_d) != 0 else None
        ),
        "inferred_ms_per_timestamp_unit_from_median": (
            args.actual_tr_ms / float(np.median(ts_d))
            if len(ts_d) and np.median(ts_d) != 0 else None
        ),
        "selected_acquisitions": selected,
        "notes": [
            "Do not assume scan_counter must equal acquisition ordinal on UIH-converted data.",
            "If sample_time_us is 1.0 while the Pulseq ADC dwell was 2.0 us and samples are 2520 vs 1260, that directly supports the documented UIH mandatory 2x ADC oversampling.",
            "discard_pre/discard_post should be used if populated; do not infer them only from the nominal pre_discard setting.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "n_acquisitions": n,
        "number_of_samples": result["field_histograms"]["number_of_samples"],
        "sample_time_us": result["field_histograms"]["sample_time_us"],
        "discard_pre": result["field_histograms"]["discard_pre"],
        "discard_post": result["field_histograms"]["discard_post"],
        "center_sample": result["field_histograms"]["center_sample"],
        "scan_counter_delta_histogram": result["scan_counter_delta_histogram"],
        "acquisition_timestamp_delta_histogram": result["acquisition_timestamp_delta_histogram"],
        "out": str(args.out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
