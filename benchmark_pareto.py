#!/usr/bin/env python3
"""
benchmark_pareto.py

Fair online-sampler comparison at matched landmark budgets: sweep DREAM's
threshold, the sweep-angle trigger, and a "sweep-angle + speed channel" hybrid
over the recorded Gazebo odometry, and report the (landmark count, max
path-reconstruction error) pairs with the SAME start+end boundary policy.

Also evaluates the speed channel independently of the N>=2 gate by measuring
the s(t) (distance-time) reconstruction error, and runs a clean straight-line
unit test.

Usage:  python3 benchmark_pareto.py
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))
import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_baselines_gazebo as bb    # noqa: E402
import dream_core                          # noqa: E402


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def sweep_angle_speed(recon, angle=0.5, r2_th=0.98):
    """Hybrid: emit a landmark when EITHER the net wrapped heading change since
    the previous landmark exceeds `angle`, OR the distance-time linearity R^2
    falls below `r2_th` (a speed change). Resets both accumulators on trigger."""
    speed = dream_core.LinFitAccumulator()
    idx = [0]
    anchor_t = recon[0][0]
    cum = 0.0
    last_psi = recon[0][3]
    n_samples = 1
    for i, (t, _x, _y, psi, s, _v) in enumerate(recon):
        if i == 0:
            continue
        d = psi - last_psi
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        cum += d
        last_psi = psi
        speed.add(t - anchor_t, s)
        n_samples += 1
        if n_samples < 5:
            # need a minimum window before judging the speed R^2
            continue
        trig_geo = abs(cum) >= angle
        trig_speed = speed.r2() < r2_th
        if trig_geo or trig_speed:
            idx.append(i)
            cum = 0.0
            anchor_t = t
            speed = dream_core.LinFitAccumulator()
            speed.add(0.0, s)
            n_samples = 1
    return idx


def fair_n_err(recon, idx):
    """Return (N, max path error) with the start+end boundary policy."""
    n_total = len(recon)
    idx = sorted(set([0] + idx + [n_total - 1]))
    pts = [(recon[i][1], recon[i][2]) for i in idx]
    mx = 0.0
    for (_t, x, y, *_r) in recon:
        mx = max(mx, bb.dist_polyline(x, y, pts))
    return len(idx), mx


def s_recon_error(recon, idx):
    """Max |reconstructed s - true s| over samples, interpolated by index."""
    idx = sorted(set([0] + idx + [len(recon) - 1]))
    if len(idx) < 2:
        return float("inf")
    j = 0
    mx = 0.0
    for i, (t, _x, _y, _psi, s, _v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        sh = recon[i0][4] + frac * (recon[i1][4] - recon[i0][4])
        mx = max(mx, abs(sh - s))
    return mx


def main():
    print("=" * 78)
    print("Fair online comparison: (landmark count N, max path error m)")
    print("=" * 78)
    for name in "ABC":
        recon = load_recon(name)
        print(f"\n--- World {name} ---")
        print(f"{'method':<24} | {'N':>5} | {'path_err':>9}")

        # DREAM sweep over R^2_th
        for th in (0.85, 0.90, 0.95, 0.98, 1.00):
            idx = rb.dream_sampler(recon, r2_threshold=th)
            n, e = fair_n_err(recon, idx)
            print(f"  dream R2={th:<5}       | {n:>5} | {e:>9.4f}")

        # sweep-angle sweep
        for ang in (0.2, 0.35, 0.5, 0.8, 1.2):
            idx = bb.sweep_angle(recon, angle=ang)
            n, e = fair_n_err(recon, idx)
            print(f"  sweep-angle {ang:<7} | {n:>5} | {e:>9.4f}")

        # hybrid sweep-angle + speed channel
        for ang, th in ((0.35, 0.98), (0.5, 0.98), (0.5, 0.95), (0.8, 0.98)):
            idx = sweep_angle_speed(recon, angle=ang, r2_th=th)
            n, e = fair_n_err(recon, idx)
            print(f"  hybrid a={ang} R2={th:<5}| {n:>5} | {e:>9.4f}")

    print("\n" + "=" * 78)
    print("Speed-channel evaluation (s(t) reconstruction, independent of N>=2)")
    print("=" * 78)
    for name in "ABC":
        recon = load_recon(name)
        # DREAM with vs without speed channel (regress_speed on/off)
        d_on = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
        d_off = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
        # sweep-angle + speed hybrid vs pure sweep-angle
        h = sweep_angle_speed(recon, angle=0.5, r2_th=0.98)
        s = bb.sweep_angle(recon, angle=0.5)
        print(f"[{name}] s(t) max err: dream(on)={s_recon_error(recon, d_on):.4f} "
              f"dream(off)={s_recon_error(recon, d_off):.4f} | "
              f"sweep+speed={s_recon_error(recon, h):.4f} "
              f"sweep={s_recon_error(recon, s):.4f}")

    print("\n" + "=" * 78)
    print("Clean straight-line unit test")
    print("=" * 78)
    # synthetic straight line, constant speed, uniform samples
    t = [i * 0.05 for i in range(201)]
    sline = [(ti, 0.2 * ti, 0.0, 0.0, 0.2 * ti, 0.2) for ti in t]
    # two-rate on the clean line (yaw window never curves)
    idx_tw = rb.baseline_sampler(sline)
    n_tw, e_tw = fair_n_err(sline, idx_tw)
    idx_dp = bb.douglas_peucker(sline, 0.05)
    n_dp, e_dp = fair_n_err(sline, idx_dp)
    print(f"  clean straight line: two-rate N={n_tw} err={e_tw:.6f} m | "
          f"DP N={n_dp} err={e_dp:.6f} m")
    ok = (e_tw < 1e-6 and e_dp < 1e-6)
    print("  RESULT:", "PASS (geometric error ~0, independent of sampling)" if ok
          else "FAIL")


if __name__ == "__main__":
    main()
