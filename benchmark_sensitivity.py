#!/usr/bin/env python3
"""
benchmark_sensitivity.py

Sensitivity and reliability analyses for the two-state (forward-drift-retained)
closed-loop recovery study, responding to the reliability/sensitivity asks:

  (S1) success fraction vs odometry noise level (sigma_v and sigma_omega);
  (S2) a constant gyro-bias (heading) perturbation;
  (S3) camera-loss fraction sweep (70/80/90/100% of the forward distance);
  (S4) ECDF of the true endpoint error at the nominal noise level.

All experiments use the same seed convention as run_gazebo_benchmark.py:
forward-phase seed Random(5000+r), return-phase seed Random(9000+r), 50 runs.

Usage:  python3 benchmark_sensitivity.py
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402

DATA = os.path.join("bench_data")
RUNS = 50
SW_LIST = [0.005, 0.01, 0.02, 0.03, 0.05]
SV_LIST = [0.005, 0.01, 0.02, 0.03]


def load(name):
    raw = []
    with open(os.path.join(DATA, f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return raw


def two_state_pe(name, sigma_v, sigma_w, frac=1.0, gyro_bias=0.0, runs=RUNS):
    raw = load(name)
    true_full = rb.reconstruct(raw)
    pes = []
    for run in range(runs):
        rng = random.Random(5000 + run)
        est_full = _reconstruct_noisy_bias(raw, sigma_v, sigma_w, gyro_bias, rng)
        k = int(frac * len(est_full))
        true_recon = true_full[:k]
        est = est_full[:k]
        d = rb.dream_sampler(est, r2_threshold=0.98)
        if len(d) < 2:
            continue
        r1 = random.Random(9000 + run)
        pe_true, _, _, _ = rb.recovery_pe_closed_loop_noisy_2state(
            true_recon, est, d, sigma_v, sigma_w, r1)
        pes.append(pe_true)
    return pes


def _reconstruct_noisy_bias(raw, sigma_v, sigma_w, gyro_bias, rng):
    out = []
    x = y = psi = s = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw:
        vn = max(0.0, v + sigma_v * rng.gauss(0.0, 1.0))
        wn = w + gyro_bias + sigma_w * rng.gauss(0.0, 1.0)
        if prev is not None:
            dt = t - prev
            if 0.0 < dt < 0.5:
                psi += wn * dt
                x += vn * math.cos(psi) * dt
                y += vn * math.sin(psi) * dt
                s += vn * dt
        out.append((t, x, y, psi, s, vn))
        prev = t
    return out


def succ(pes):
    return sum(1 for x in pes if x < 0.06) / len(pes)


def main():
    print("=" * 78)
    print(f"Sensitivity and reliability (two-state study, {RUNS} runs)")
    print("=" * 78)

    print("\n(S1a) success fraction vs sigma_omega (sigma_v=0.01 m/s)")
    print("  " + " ".join(f"w={sw}" for sw in SW_LIST))
    for name in "ABC":
        row = []
        for sw in SW_LIST:
            pes = two_state_pe(name, 0.01, sw)
            row.append(f"{succ(pes)*100:.0f}%")
        print(f"  {name}: " + " ".join(row))

    print("\n(S1b) success fraction vs sigma_v (sigma_omega=0.02 rad/s)")
    print("  " + " ".join(f"v={sv}" for sv in SV_LIST))
    for name in "ABC":
        row = []
        for sv in SV_LIST:
            pes = two_state_pe(name, sv, 0.02)
            row.append(f"{succ(pes)*100:.0f}%")
        print(f"  {name}: " + " ".join(row))

    print("\n(S2) constant gyro bias (sigma_v=0.01, sigma_omega=0.02)")
    for name in "ABC":
        for bias in (0.005, 0.01, 0.02):
            pes = two_state_pe(name, 0.01, 0.02, gyro_bias=bias)
            m = sum(pes) / len(pes)
            print(f"  {name} bias={bias}: mean={m:.4f} m, success={succ(pes)*100:.0f}%")

    print("\n(S3) camera-loss fraction (nominal noise)")
    for name in "ABC":
        row = []
        for frac in (0.7, 0.8, 0.9, 1.0):
            pes = two_state_pe(name, 0.01, 0.02, frac=frac)
            row.append(f"f={frac}:{succ(pes)*100:.0f}%")
        print(f"  {name}: " + "  ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
