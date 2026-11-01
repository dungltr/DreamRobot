#!/usr/bin/env python3
"""
sweep_r2_threshold.py

Offline sweep of the DREAM R^2 threshold over the already-recorded Gazebo
odometry (bench_data/odom_{A,B,C}.csv). For each threshold we report the number
of DREAM landmarks, the data reduction ratio (vs the baseline sampler) and the
recovery position error. This is the DR<->PE operating-point table (NOT a Pareto
frontier); the baseline is a single fixed point for reference.

Usage:  python3 sweep_r2_threshold.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402

WORLDS = ["A", "B", "C"]
THRESHOLDS = [0.85, 0.90, 0.95, 0.98, 1.00]


def load(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    return rb.reconstruct(raw)


def main():
    print("=" * 72)
    print("DREAM R^2_th sweep on recorded Gazebo odometry "
          "(closed-loop pure-pursuit replay)")
    print("=" * 72)
    for name in WORLDS:
        recon = load(name)
        b_idx = rb.baseline_sampler(recon)
        pe_b = rb.recovery_pe_closed_loop(recon, b_idx)

        print(f"\n=== World {name} ===  "
              f"N_total={len(recon)}  N_baseline={len(b_idx)}  "
              f"PE_baseline_cl={pe_b:.3f} m")
        print("-" * 72)
        print(f"{'R2_th':>7} | {'N_dream':>7} | {'DR':>7} | {'PE_dream_cl':>11} | "
              f"{'PE vs base':>11}")
        print("-" * 72)
        for th in THRESHOLDS:
            d_idx = rb.dream_sampler(recon, r2_threshold=th)
            dr = 1.0 - len(d_idx) / len(b_idx) if b_idx else 0.0
            pe = rb.recovery_pe_closed_loop(recon, d_idx)
            ratio = pe / pe_b if pe_b > 1e-9 else float("inf")
            print(f"{th:>7.2f} | {len(d_idx):>7} | {dr:>6.1%} | {pe:>10.3f} m | "
                  f"{ratio:>10.2f}x")
        print("-" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
