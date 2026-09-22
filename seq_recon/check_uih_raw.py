#!/usr/bin/env python3
"""
Read-only audit helper for UIH RTSpiral raw data.

Native UIH .raw is vendor-specific. This script can inventory native .raw files
and extract only technical-looking printable strings from small sampled regions,
but it does NOT pretend to decode the binary acquisition headers.

For exact acquisition metadata (SLC/LIN-like indices, number of samples, coil
channels, timestamps and geometry), first convert UIH raw to ISMRMRD and pass
--mrd <file.h5>. The exact-audit branch uses the Python `ismrmrd` package.

Examples
--------
python check_uih_raw.py \
  --input-dir /path/to/T13_60 \
  --expected-slices 60 \
  --arms-per-slice 350 \
  --arms-per-frame 7 \
  --out-dir ./audit_T13_60

# after UIH -> ISMRMRD conversion
pip install ismrmrd h5py
python check_uih_raw.py \
  --input-dir /path/to/T13_60 \
  --expected-slices 60 \
  --mrd /path/to/T13_60.h5 \
  --out-dir ./audit_T13_60_mrd
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

TECH = re.compile(
    r"(pulseq|slc|slice|lin|line|rep|repetition|adc|sample|channel|coil|"
    r"kspace|encode|trajectory|position|read[_ ]?dir|phase[_ ]?dir|slice[_ ]?dir|"
    r"orientation|fov|matrix|dwell|bandwidth|frequency|shim|b0|gradient|"
    r"timestamp|scan[_ ]?counter|average|contrast|segment|set|echo|tr|te)",
    re.I,
)
SENSITIVE = re.compile(
    r"(patient|name|birth|dob|address|phone|email|accession|hospital|institution|operator|physician)",
    re.I,
)


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024


def classify(name: str) -> str:
    s = name.lower()
    if "palb0shim" in s:
        return "auxiliary_candidate: PALB0Shim (likely B0-shim/prescan; confirm with UIH)"
    if "palfre" in s:
        return "auxiliary_candidate: PALFre (likely frequency/prescan; confirm with UIH)"
    if "pulseq" in s:
        return "main_candidate: Pulseq imaging raw"
    return "unknown_raw"


def quick_hash(path: Path, block=1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        offsets = [0]
        if size > block:
            offsets += [max(0, size // 2 - block // 2), max(0, size - block)]
        for off in sorted(set(offsets)):
            f.seek(off)
            h.update(off.to_bytes(8, "little"))
            h.update(f.read(min(block, size - off)))
    return h.hexdigest()


def sampled_blocks(path: Path, n=2 * 1024 * 1024):
    size = path.stat().st_size
    with path.open("rb") as f:
        offsets = [0]
        if size > n:
            offsets += [max(0, size // 2 - n // 2), max(0, size - n)]
        for off in sorted(set(offsets)):
            f.seek(off)
            yield off, f.read(min(n, size - off))


def strings_ascii(data: bytes, min_len=5):
    pat = rb"[\x20-\x7E]{" + str(min_len).encode() + rb",}"
    for m in re.finditer(pat, data):
        yield m.group().decode("ascii", errors="replace")


def strings_utf16le(data: bytes, min_len=5):
    pat = rb"(?:[\x20-\x7E]\x00){" + str(min_len).encode() + rb",}"
    for m in re.finditer(pat, data):
        yield m.group().decode("utf-16le", errors="replace")


def technical_strings(path: Path, limit=300):
    out, seen = [], set()
    for off, data in sampled_blocks(path):
        for s in list(strings_ascii(data)) + list(strings_utf16le(data)):
            s = " ".join(s.split())
            if not TECH.search(s) or SENSITIVE.search(s):
                continue
            s = s[:500]
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"sample_region_offset": off, "text": s})
            if len(out) >= limit:
                return out
    return out


def inspect_config(path: Path):
    raw = path.read_bytes()
    result = {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "first64_hex": raw[:64].hex(" "),
        "text_preview": None,
    }
    for enc in ("utf-8", "utf-16le", "gb18030", "latin1"):
        try:
            txt = raw.decode(enc)
        except Exception:
            continue
        ratio = sum(c.isprintable() or c in "\r\n\t" for c in txt) / max(len(txt), 1)
        if ratio > 0.9:
            lines = []
            for line in txt.splitlines()[:200]:
                lines.append("[REDACTED_SENSITIVE_LINE]" if SENSITIVE.search(line) else line)
            result["decoded_as"] = enc
            result["text_preview"] = "\n".join(lines)
            break
    return result


def v3(x):
    try:
        return [float(x[0]), float(x[1]), float(x[2])]
    except Exception:
        return []


def audit_mrd(path: Path, dataset_name: str, expected_slices, arms_per_slice, arms_per_frame, out_dir: Path):
    try:
        import ismrmrd
    except Exception as e:
        return {"status": "SKIPPED", "reason": f"cannot import ismrmrd: {e}", "hint": "pip install ismrmrd h5py"}

    try:
        ds = ismrmrd.Dataset(str(path), dataset_name, create_if_needed=False)
    except Exception as e:
        return {"status": "FAILED_TO_OPEN", "reason": str(e), "dataset_name": dataset_name}

    n = int(ds.number_of_acquisitions())
    rows = []
    slice_count = Counter()
    lin_count = Counter()
    rep_count = Counter()
    sample_count = Counter()
    channel_count = Counter()
    slice_changes = []
    prev_slc = None

    flag_names = [
        "ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PARALLEL_CALIBRATION",
        "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING", "ACQ_IS_NAVIGATION_DATA",
        "ACQ_IS_PHASECORR_DATA", "ACQ_FIRST_IN_SLICE", "ACQ_LAST_IN_SLICE",
        "ACQ_FIRST_IN_REPETITION", "ACQ_LAST_IN_REPETITION",
    ]

    for i in range(n):
        a = ds.read_acquisition(i)
        idx = a.idx
        slc = int(idx.slice)
        lin = int(idx.kspace_encode_step_1)
        rep = int(idx.repetition)
        if prev_slc is None or slc != prev_slc:
            slice_changes.append(i)
            prev_slc = slc

        flags = []
        for fn in flag_names:
            flag = getattr(ismrmrd, fn, None)
            if flag is not None:
                try:
                    if a.isFlagSet(flag):
                        flags.append(fn)
                except Exception:
                    pass

        row = {
            "i": i,
            "scan_counter": int(a.scan_counter),
            "acquisition_time_stamp": int(a.acquisition_time_stamp),
            "number_of_samples": int(a.number_of_samples),
            "active_channels": int(a.active_channels),
            "trajectory_dimensions": int(a.trajectory_dimensions),
            "center_sample": int(a.center_sample),
            "slice": slc,
            "lin_kspace_encode_step_1": lin,
            "kspace_encode_step_2": int(idx.kspace_encode_step_2),
            "repetition": rep,
            "average": int(idx.average),
            "contrast": int(idx.contrast),
            "phase": int(idx.phase),
            "set": int(idx.set),
            "segment": int(idx.segment),
            "position": v3(a.position),
            "read_dir": v3(a.read_dir),
            "phase_dir": v3(a.phase_dir),
            "slice_dir": v3(a.slice_dir),
            "flags": ";".join(flags),
        }
        rows.append(row)
        slice_count[slc] += 1
        lin_count[lin] += 1
        rep_count[rep] += 1
        sample_count[int(a.number_of_samples)] += 1
        channel_count[int(a.active_channels)] += 1

    csv_path = out_dir / "ismrmrd_acquisitions.csv"
    flat_fields = [
        "i", "scan_counter", "acquisition_time_stamp", "number_of_samples", "active_channels",
        "trajectory_dimensions", "center_sample", "slice", "lin_kspace_encode_step_1",
        "kspace_encode_step_2", "repetition", "average", "contrast", "phase", "set", "segment",
        "position", "read_dir", "phase_dir", "slice_dir", "flags",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for key in ("position", "read_dir", "phase_dir", "slice_dir"):
                rr[key] = json.dumps(rr[key])
            w.writerow(rr)

    # compact boundary table: four acquisitions before and after every slice change
    boundary_path = out_dir / "ismrmrd_slice_boundaries.csv"
    boundary_rows = []
    for b in slice_changes:
        for r in rows[max(0, b - 4):min(n, b + 5)]:
            boundary_rows.append({
                "boundary_at": b, "i": r["i"], "scan_counter": r["scan_counter"],
                "slice": r["slice"], "lin": r["lin_kspace_encode_step_1"],
                "repetition": r["repetition"], "samples": r["number_of_samples"],
                "channels": r["active_channels"], "timestamp": r["acquisition_time_stamp"],
            })
    with boundary_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["boundary_at", "i", "scan_counter", "slice", "lin", "repetition", "samples", "channels", "timestamp"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(boundary_rows)

    expected_acq = expected_slices * arms_per_slice if expected_slices is not None else None
    per_slice_lin = {}
    for slc in sorted(slice_count):
        seq = [r["lin_kspace_encode_step_1"] for r in rows if r["slice"] == slc]
        per_slice_lin[str(slc)] = {
            "n": len(seq), "unique": len(set(seq)), "min": min(seq), "max": max(seq),
            "first12": seq[:12], "last12": seq[-12:],
        }

    return {
        "status": "OK",
        "n_acquisitions": n,
        "expected_acquisitions": expected_acq,
        "acquisition_count_matches_expected": (n == expected_acq if expected_acq is not None else None),
        "unique_slices": sorted(slice_count),
        "slice_count": len(slice_count),
        "expected_slices": expected_slices,
        "acquisitions_per_slice": dict(sorted(slice_count.items())),
        "all_slices_have_expected_arms": all(v == arms_per_slice for v in slice_count.values()),
        "lin_unique_values": sorted(lin_count),
        "repetition_unique_values": sorted(rep_count),
        "samples_histogram": dict(sample_count),
        "channels_histogram": dict(channel_count),
        "slice_change_indices_first20": slice_changes[:20],
        "per_slice_lin_summary": per_slice_lin,
        "first_acquisition_geometry": {
            "position": rows[0]["position"] if rows else [],
            "read_dir": rows[0]["read_dir"] if rows else [],
            "phase_dir": rows[0]["phase_dir"] if rows else [],
            "slice_dir": rows[0]["slice_dir"] if rows else [],
        },
        "expected_frames_per_slice": arms_per_slice // arms_per_frame,
        "outputs": {"acquisitions_csv": str(csv_path), "slice_boundaries_csv": str(boundary_path)},
    }


def main():
    ap = argparse.ArgumentParser(description="Audit UIH RTSpiral raw export; exact audit supported after ISMRMRD conversion.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--expected-slices", type=int, default=None)
    ap.add_argument("--arms-per-slice", type=int, default=350)
    ap.add_argument("--arms-per-frame", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("uih_raw_audit"))
    ap.add_argument("--mrd", type=Path, default=None, help="optional converted ISMRMRD .h5/.mrd")
    ap.add_argument("--dataset-name", default="dataset")
    args = ap.parse_args()

    inp = args.input_dir.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not inp.is_dir():
        raise SystemExit(f"input directory not found: {inp}")

    raw_files = sorted(inp.glob("*.raw"))
    cfg_files = sorted(inp.glob("*.config"))
    raw_info = []
    tech = {}
    for p in raw_files:
        st = p.stat()
        with p.open("rb") as f:
            first16 = f.read(16).hex(" ")
        raw_info.append({
            "path": str(p.resolve()), "name": p.name, "size_bytes": st.st_size,
            "size_human": human_bytes(st.st_size),
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "classification": classify(p.name), "quick_fingerprint_sha256": quick_hash(p),
            "first16_hex": first16,
        })
        tech[p.name] = technical_strings(p)

    expected_acq = args.expected_slices * args.arms_per_slice if args.expected_slices is not None else None
    report = {
        "schema": "cr-dreme-uih-raw-audit-v1",
        "input_dir": str(inp),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expectation": {
            "expected_slices": args.expected_slices,
            "arms_per_slice": args.arms_per_slice,
            "arms_per_frame": args.arms_per_frame,
            "expected_acquisitions": expected_acq,
            "expected_frames_per_slice": args.arms_per_slice // args.arms_per_frame,
            "expected_total_frames": (expected_acq // args.arms_per_frame if expected_acq is not None else None),
        },
        "raw_files": raw_info,
        "config_files": [inspect_config(p) for p in cfg_files],
        "technical_strings_from_sampled_regions": tech,
        "native_raw_limit": "No exact binary UIH header parsing is claimed. Convert to ISMRMRD for exact metadata audit.",
    }

    if args.mrd is not None:
        mrd = args.mrd.expanduser().resolve()
        report["ismrmrd"] = (
            audit_mrd(mrd, args.dataset_name, args.expected_slices, args.arms_per_slice, args.arms_per_frame, out)
            if mrd.exists() else {"status": "MISSING", "path": str(mrd)}
        )

    json_path = out / "raw_audit.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["=== UIH RTSpiral RAW AUDIT ===", f"Input: {inp}", "", "Files:"]
    for x in raw_info:
        lines.append(f"  {x['name']:<55} {x['size_human']:>12}  {x['classification']}")
    lines += [
        "", "Expected project structure:",
        f"  slices={args.expected_slices}, arms/slice={args.arms_per_slice}, arms/frame={args.arms_per_frame}",
        f"  expected acquisitions={expected_acq}",
        f"  expected frames/slice={args.arms_per_slice // args.arms_per_frame}",
        "", "Important:",
        "  Native .raw inventory/string scan is reconnaissance only.",
        "  Exact SLC/LIN/sample/channel/geometry checks require UIH -> ISMRMRD conversion.",
    ]
    if "ismrmrd" in report:
        m = report["ismrmrd"]
        lines += ["", "ISMRMRD:", f"  status={m.get('status')}"]
        if m.get("status") == "OK":
            lines += [
                f"  n_acquisitions={m.get('n_acquisitions')}",
                f"  matches_expected={m.get('acquisition_count_matches_expected')}",
                f"  slices={m.get('unique_slices')}",
                f"  reps={m.get('repetition_unique_values')}",
                f"  samples={m.get('samples_histogram')}",
                f"  channels={m.get('channels_histogram')}",
            ]
        else:
            lines.append(f"  reason={m.get('reason', '')}")
    lines += ["", f"JSON: {json_path}"]
    txt_path = out / "raw_audit.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
