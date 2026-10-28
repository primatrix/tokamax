"""Extract device_duration_ps from the XLA Ops lines of an XPlane trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _duration(event):
  return int(dict((str(k), str(v)) for k, v in event.stats)["device_duration_ps"])


def _event_rows(trace_dir: Path):
  from jaxlib import _profile_data
  rows = []
  for path in sorted(trace_dir.rglob("*.xplane.pb")):
    profile = _profile_data.ProfileData.from_file(str(path))
    for plane in profile.planes:
      # SparseCore planes have suffixes; the bare plane is the TensorCore
      # timeline used for the CP8 kernel comparison.
      if not plane.name.startswith("/device:TPU:") or "SparseCore" in plane.name:
        continue
      for line in plane.lines:
        if line.name != "XLA Ops":
          continue
        for event in line.events:
          stats = {str(k): str(v) for k, v in event.stats}
          if "device_duration_ps" not in stats:
            continue
          rows.append({
              "plane": plane.name,
              "name": event.name,
              "offset_ps": int(stats.get("device_offset_ps", 0)),
              "duration_ps": int(stats["device_duration_ps"]),
          })
  return rows


def extract_baseline(trace_dir: Path):
  rows = _event_rows(trace_dir)
  per_plane = {}
  for row in rows:
    per_plane.setdefault(row["plane"], []).append(row)
  result = {"kind": "baseline", "planes": {}, "raw_xla_ops": rows}
  for plane, events in per_plane.items():
    splash = [x for x in events if "splash_mha_fwd_segmented_residuals" in x["name"]]
    if len(splash) != 1:
      raise RuntimeError(f"{plane}: expected one Splash event, got {len(splash)}")
    splash_row = splash[0]
    splash_end = splash_row["offset_ps"] + splash_row["duration_ps"]
    collectives = [x for x in events if any(token in x["name"].lower() for token in (
        "all-to-all", "all_to_all", "allgather", "all-gather",
        "collective-permute", "collective_permute"))]
    pre = [x for x in collectives if x["offset_ps"] < splash_row["offset_ps"]]
    post = [x for x in collectives if x["offset_ps"] >= splash_end]
    result["planes"][plane] = {
        "pre_a2a_ms": sum(x["duration_ps"] for x in pre) / 1e9,
        "splash_ms": splash_row["duration_ps"] / 1e9,
        "post_a2a_ms": sum(x["duration_ps"] for x in post) / 1e9,
        "pre_a2a_events": pre,
        "post_a2a_events": post,
        "splash_event": splash_row,
    }
  if not result["planes"]:
    raise RuntimeError("no TensorCore XLA Ops planes found")
  keys = ("pre_a2a_ms", "splash_ms", "post_a2a_ms")
  result["max_ms"] = {key: max(x[key] for x in result["planes"].values())
                       for key in keys}
  result["max_ms"]["baseline_sum_ms"] = sum(result["max_ms"][key] for key in keys)
  return result


def extract_optimized(trace_dir: Path, name_hint: str | None = None):
  rows = _event_rows(trace_dir)
  per_plane = {}
  for row in rows:
    per_plane.setdefault(row["plane"], []).append(row)
  result = {"kind": "optimized", "planes": {}, "raw_xla_ops": rows}
  hints = tuple(x for x in (name_hint, "v7a_step_c_head_overlap_a2a_splash512",
                            "splash_mha_a2a") if x)
  for plane, events in per_plane.items():
    matches = [x for x in events if any(h in x["name"] for h in hints)]
    if len(matches) != 1:
      raise RuntimeError(f"{plane}: expected one fused pallas call, got {len(matches)}")
    result["planes"][plane] = {
        "pallas_call_ms": matches[0]["duration_ps"] / 1e9,
        "pallas_call_event": matches[0],
    }
  if not result["planes"]:
    raise RuntimeError("no TensorCore XLA Ops planes found")
  result["max_pallas_call_ms"] = max(
      x["pallas_call_ms"] for x in result["planes"].values()
  )
  return result


def main(kind: str):
  parser = argparse.ArgumentParser()
  parser.add_argument("--trace-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--name-hint")
  args = parser.parse_args()
  result = (extract_baseline(args.trace_dir) if kind == "baseline"
            else extract_optimized(args.trace_dir, args.name_hint))
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({k: v for k, v in result.items() if k not in ("raw_xla_ops", "planes")}, indent=2))
