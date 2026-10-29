"""Extract the compute-only fused Splash custom-call duration."""

import sys

from extract_events import main


if __name__ == "__main__":
  sys.argv.extend(("--name-hint", "splash_mha_compute_only"))
  main("optimized")
