#!/usr/bin/env python3
"""
benchmark_m10_longtimeout.py

Long-trajectory study with a length-scaled timeout (reviewer N10). The M6
trace study used a fixed 180 s return timeout, which at the return speed of
0.12 m/s caps the return at ~21 m, so the 40 m traces timed out for a reason
unrelated to the sampler. This script re-runs 20/40/100 m traces with the
timeout scaled to the path length (timeout = path_length / 0.12 * 1.5 + 60),
so the timeout no longer dominates, and reports the true endpoint error and
completion for the baseline and DREAM samplers under the same paired seeds.

Usage:  python3 benchmark_m10_longtimeout.py
"""

import argparse
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_m5_noise as m5            # noqa: E402

RUNS = 50
SIGMA_V, SIGMA_W = 0.01, 0.02
RETURN_SPEED = 0.12

MAX_LIN_ACCEL = 0.20
MAX_ANG_ACCEL = 0.50
DT = 0.02


def straight_then_turn(length_m):
    """A straight segment of `length_m`, a 90 deg turn, then a short return
    leg; keeps the profile mostly straight so the distance--time and drift
    behaviour is what is exercised."""
    straight_s = (length_m - 1.0) / 0.20
    return [(0.20, 0.00, straight_s), (0.20, 0.60, 2.6), (0.20, 0.00, 5.0)]


def curved_trace(length_m):
    """An alternating-curvature trace of total length ~ `length_m`."""
    # one S-cycle ~ 2 * (0.20 * 12 s) = 4.8 m; number of cycles to reach length
    n = max(1, int(round(length_m / 4.8)))
    return [(0.20, 0.30, 12.0), (0.20, -0.30, 12.0)] * n


def synthesize_raw(profile):
    out = []
    t = 0.0
    x = y = yaw = v = w = 0.0
    for (v_cmd, w_cmd, dur) in profile:
        n = max(1, int(round(dur / DT)))
        for _ in range(n):
            dv = v_cmd - v
            v = v_cmd if abs(dv) <= MAX_LIN_ACCEL * DT else \
                v + (MAX_LIN_ACCEL * DT if dv > 0 else -MAX_LIN_ACCEL * DT)
            dw = w_cmd - w
            w = w_cmd if abs(dw) <= MAX_ANG_ACCEL * DT else \
                w + (MAX_ANG_ACCEL * DT if dw > 0 else -MAX_ANG_ACCEL * DT)
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            t += DT
            out.append((t, x, y, v, w))
    return out


def path_length(recon):
    return sum(math.hypot(recon[i][1] - recon[i - 1][1],
                          recon[i][2] - recon[i - 1][2])
               for i in range(1, len(recon)))


def run(name, profile, runs=RUNS):
    raw = synthesize_raw(profile)
    true_full = rb.reconstruct(raw)
    L = path_length(true_full)
    timeout = L / RETURN_SPEED * 1.5 + 60.0
    base_pe, dream_pe = [], []
    base_complete = dream_complete = 0
    for run_i in range(runs):
        rng = random.Random(5000 + run_i)
        est_full = m5.forward_estimate(raw, SIGMA_V, SIGMA_W, rng)
        b = rb.baseline_sampler(est_full)
        d = rb.dream_sampler(est_full, r2_threshold=0.98)
        r1 = random.Random(9000 + run_i)
        pb, _po, _c, b_done, _t = m5.two_state_return(
            true_full, est_full, b, SIGMA_V, SIGMA_W, r1, timeout=timeout)
        r1 = random.Random(9000 + run_i)
        pd, _po2, _c2, d_done, _t2 = m5.two_state_return(
            true_full, est_full, d, SIGMA_V, SIGMA_W, r1, timeout=timeout)
        base_pe.append(pb)
        dream_pe.append(pd)
        base_complete += 1 if b_done else 0
        dream_complete += 1 if d_done else 0

    diffs = [d - b for b, d in zip(base_pe, dream_pe)]

    def m(v):
        return sum(v) / len(v)

    def ci(v):
        if len(v) < 2:
            return 0.0
        var = sum((x - m(v)) ** 2 for x in v) / (len(v) - 1)
        return 1.96 * math.sqrt(var / len(v))

    return {
        "name": name, "L": L, "timeout": timeout,
        "base_pe": m(base_pe), "dream_pe": m(dream_pe),
        "diff_pe": m(diffs), "diff_ci": ci(diffs),
        "base_succ": sum(1 for x in base_pe if x < 0.06) / runs,
        "dream_succ": sum(1 for x in dream_pe if x < 0.06) / runs,
        "base_complete": base_complete / runs,
        "dream_complete": dream_complete / runs,
        "N_base": len(rb.baseline_sampler(true_full)),
        "N_dream": len(rb.dream_sampler(true_full, r2_threshold=0.98)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=RUNS)
    args = ap.parse_args()
    cases = [
        ("straight_20m", straight_then_turn(20.0)),
        ("curved_20m", curved_trace(20.0)),
        ("straight_40m", straight_then_turn(40.0)),
        ("curved_40m", curved_trace(40.0)),
        ("straight_100m", straight_then_turn(100.0)),
        ("curved_100m", curved_trace(100.0)),
    ]
    print("=" * 104)
    print(f"Long-trace study with length-scaled timeout ({args.runs} runs, "
          f"sigma_v={SIGMA_V}, sigma_w={SIGMA_W})")
    print("=" * 104)
    print(f"{'trace':<16} {'L(m)':>6} {'timeout(s)':>11} {'N_b':>4} {'N_d':>4} "
          f"{'PE_base':>9} {'PE_dream':>9} {'diff(d-b)':>11} {'succ_b':>7} "
          f"{'succ_d':>7} {'comp_b':>7} {'comp_d':>7}")
    for name, prof in cases:
        r = run(name, prof, runs=args.runs)
        print(f"{r['name']:<16} {r['L']:>6.1f} {r['timeout']:>11.1f} "
              f"{r['N_base']:>4} {r['N_dream']:>4} "
              f"{r['base_pe']:>9.3f} {r['dream_pe']:>9.3f} "
              f"{r['diff_pe']:>+7.3f}\u00b1{r['diff_ci']:.3f} "
              f"{r['base_succ']*100:>6.0f}% {r['dream_succ']*100:>6.0f}% "
              f"{r['base_complete']*100:>6.0f}% {r['dream_complete']*100:>6.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
