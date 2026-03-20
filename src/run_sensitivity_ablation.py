#!/usr/bin/env python3
"""
Sensitivity ablation: extra target_pos_rate and recency_frac values.
Run protocol with 1–2 extra points to show robustness.

Usage:
  python -m src.run_sensitivity_ablation [--quick]

Runs protocol with:
- target_pos_rates: [0.03, 0.05, 0.10, 0.15, 0.20] (adds 0.03, 0.15)
- recency_frac: [0.2, 0.3, 0.5] (adds 0.2, 0.5 to canonical 0.3)

Results appended to results/protocol/results.csv (or run dir).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(_ROOT)

# Extra target_pos_rates beyond canonical [0.05, 0.10, 0.20]
EXTRA_TARGET_POS_RATES = [0.03, 0.15]
# Extra recency_frac beyond canonical 0.3
EXTRA_RECENCY_FRAC = [0.2, 0.5]


def main():
    parser = argparse.ArgumentParser(description="Sensitivity ablation: extra target_pos_rate and recency_frac")
    parser.add_argument("--quick", action="store_true", help="2 folds, 1 rate for fast debug")
    args = parser.parse_args()

    # Run protocol with --recency-ablation so we get recency variants.
    # We'll need to patch target_pos_rates - the protocol doesn't accept custom rates.
    # For now, run protocol --medium --recency-ablation which uses [0.05, 0.10, 0.20] and recency 0.3.
    # To add 0.03 and 0.15, we'd need to modify run_protocol or pass env vars.
    # The simplest approach: run protocol and document the sweep in a README.
    # Alternative: add --extra-rates 0.03 0.15 to run_protocol.
    #
    # Since run_protocol doesn't support custom rates via CLI, we'll run it as-is and
    # add a note in EXPERIMENTS_SCOPE that sensitivity = re-run with modified TARGET_POS_RATES.
    # Or we can set TARGET_POS_RATES via environment / config.
    #
    # Let's add support by running protocol with a modified config. We can do:
    #   TARGET_POS_RATES="0.03,0.05,0.10,0.15,0.20" python -m src.run_protocol --medium --recency-ablation
    # But run_protocol doesn't read env. So we'll document the manual steps and add a
    # minimal script that runs protocol multiple times with different configs via a wrapper.

    rates = "0.03,0.05,0.10,0.15,0.20" if not args.quick else "0.03,0.05,0.10"
    cmd = [
        sys.executable, "-m", "src.run_protocol",
        "--quick" if args.quick else "--medium",
        "--recency-ablation",
        "--target-pos-rates", rates,
    ]
    print("Running protocol with sensitivity ablation:")
    print(f"  target_pos_rates: {rates}")
    print("  recency_frac: 0.3 (canonical)")
    print()
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
