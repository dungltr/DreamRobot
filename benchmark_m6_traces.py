#!/usr/bin/env python3
"""
benchmark_m6_traces.py

Independent-trajectory generalization study (reviewer M6): the three recorded
~4 m Gazebo worlds leave several failure modes unexercised. This script
synthesizes six additional trajectory *geometries* (long straight -> sharp
turn, low-curvature long arc, alternating S, stop-turn-go, hairpin/U-turn, and
a self-crossing figure-eight), and for each runs the two-state
(forward-drift-retained) closed-loop return with the SAME per-run noise seeds
for the baseline and DREAM samplers, so the endpoint error difference is a
*paired* comparison.

Reported per trace:
  * mean true endpoint error (baseline vs DREAM);
  * the paired per-run difference (DREAM - baseline), mean +/- 95% CI;
  * ECDF summary of DREAM's true endpoint error: max, 90th and 95th percentiles
    (an empirical percentile interpolation, not a tail-stability claim).

The return replay is progress-constrained (nearest-point search only advances
forward), so it cannot jump across the near branches of the self-crossing
trace.

Usage:  python3 benchmark_m6_traces.py
"""

import csv
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

# Synthetic (v, w, T) profiles; each segment is a constant-command block that
# is integrated kinematically at dt=0.02 s with slew limits (mirrors the
# recorded-world model). Units: v (m/s), w (rad/s), T (s).
PROFILES = {
    "long_straight_sharp": [(0.20, 0.00, 4.0), (0.20, 0.60, 2.6),   # ~90 deg turn
                            (0.20, 0.00, 3.0)],
    "low_curv_long_arc":   [(0.20, 0.15, 10.0), (0.20, 0.00, 2.0)],
    "alternating_S":       [(0.20, 0.50, 2.5), (0.20, -0.50, 2.5),
                            (0.20, 0.50, 2.5), (0.20, -0.50, 2.5)],
    "stop_turn_go":        [(0.20, 0.00, 2.0), (0.00, 0.00, 0.5),
                            (0.00, 0.80, 1.0), (0.20, 0.00, 3.0)],
    "hairpin_uturn":       [(0.20, 0.00, 2.0), (0.20, 0.70, 4.5),   # ~180 deg turn
                            (0.20, 0.00, 2.0)],
    # figure-eight: one CCW then one CW full circle sharing the start point
    "self_crossing":       [(0.20, 0.60, 10.5), (0.20, -0.60, 10.5),
                            (0.20, 0.00, 0.5)],
}

MAX_LIN_ACCEL = 0.20
MAX_ANG_ACCEL = 0.50
DT = 0.02


def synthesize_raw(profile):
    """Integrate a (v, w, T) profile with slew limits -> raw (t, x, y, v, w)."""
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


def mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, z * math.sqrt(var / len(values))


def pct(values, q):
    """Empirical percentile by linear interpolation."""
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_trace(name):
    raw = synthesize_raw(PROFILES[name])
    true_full = rb.reconstruct(raw)
    n = len(true_full)
    base_pe, dream_pe, diffs = [], [], []
    for run_i in range(RUNS):
        rng = random.Random(5000 + run_i)
        est_full = m5.forward_estimate(raw, SIGMA_V, SIGMA_W, rng)
        b = rb.baseline_sampler(est_full)
        d = rb.dream_sampler(est_full, r2_threshold=0.98)
        r1 = random.Random(9000 + run_i)
        pb, _, _, _, _ = m5.two_state_return(true_full, est_full, b,
                                             SIGMA_V, SIGMA_W, r1)
        r1 = random.Random(9000 + run_i)
        pd, _, _, _, _ = m5.two_state_return(true_full, est_full, d,
                                             SIGMA_V, SIGMA_W, r1)
        base_pe.append(pb)
        dream_pe.append(pd)
        diffs.append(pd - pb)
    mb, cb = mean_ci(base_pe)
    md, cd = mean_ci(dream_pe)
    mdiff, cdiff = mean_ci(diffs)
    return {
        "name": name,
        "N_total": n,
        "N_base": len(rb.baseline_sampler(true_full)),
        "N_dream": len(rb.dream_sampler(true_full, r2_threshold=0.98)),
        "base_mean": mb, "base_ci": cb,
        "dream_mean": md, "dream_ci": cd,
        "diff_mean": mdiff, "diff_ci": cdiff,
        "max": max(dream_pe), "p90": pct(dream_pe, 0.90),
        "p95": pct(dream_pe, 0.95),
        "success": sum(1 for x in dream_pe if x < 0.06) / RUNS,
    }


def main():
    print("=" * 90)
    print(f"M6 independent-trace generalization (paired seeds, {RUNS} runs, "
          f"sigma_v={SIGMA_V}, sigma_w={SIGMA_W})")
    print("=" * 90)
    print(f"{'trace':<20} {'N':>4} {'N_b':>4} {'N_d':>4} "
          f"{'PE_base':>12} {'PE_dream':>12} {'diff (d-b)':>14} "
          f"{'p90':>6} {'p95':>6} {'max':>6} {'succ':>5}")
    for name in PROFILES:
        r = run_trace(name)
        print(f"{r['name']:<20} {r['N_total']:>4} {r['N_base']:>4} "
              f"{r['N_dream']:>4} "
              f"{r['base_mean']:>6.3f}\u00b1{r['base_ci']:.3f} "
              f"{r['dream_mean']:>6.3f}\u00b1{r['dream_ci']:.3f} "
              f"{r['diff_mean']:>+7.4f}\u00b1{r['diff_ci']:.4f} "
              f"{r['p90']:>6.3f} {r['p95']:>6.3f} {r['max']:>6.3f} "
              f"{r['success']*100:>4.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
