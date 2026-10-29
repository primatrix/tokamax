"""Run the direct-body probe and persist a traceback on failure."""

from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path


def main() -> None:
  if "--out" not in sys.argv:
    raise SystemExit("--out is required")
  out = Path(sys.argv[sys.argv.index("--out") + 1])
  try:
    runpy.run_path(
        str(Path(__file__).with_name("run_direct_body_compute_only.py")),
        run_name="__main__",
    )
  except BaseException:
    out.mkdir(parents=True, exist_ok=True)
    (out / "error.txt").write_text(traceback.format_exc())
    raise


if __name__ == "__main__":
  main()
