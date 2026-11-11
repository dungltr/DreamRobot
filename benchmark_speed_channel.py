#!/usr/bin/env python3
"""
benchmark_speed_channel.py

Reproduces Table tab:speed (the distance--time channel ablation) on the three
recorded Gazebo worlds. For each world it reports, for DREAM path-only,
DREAM path+speed, sweep-angle, and sweep-angle+speed: the landmark count N,
the maximum path-reconstruction error, the maximum distance--time
reconstruction error e_s with the recommended arc-increment field d=Δs_k
(the accumulated arc length increment between consecutive landmarks), and the
maximum speed-reconstruction error e_v = max_i |v_hat_i - v_i| with
v_hat = Δs_k / T_k (the segment-average speed between consecutive landmarks).

e_s and e_v are both computed from the *arc* increment Δs_k (recon[i][4] is the
true accumulated travelled distance), NOT from the backward-compatible chord
field. This matches the primary representation recommended in the manuscript;
the chord-payload decoder (rb.chord_payload_s_error) is a separate,
backward-compatibility metric.

Usage:  python3 benchmark_speed_channel.py
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
import benchmark_pareto as bp              # noqa: E402
import dream_core                          # noqa: E402


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def arc_s_error(recon, idx):
    """e_s from the arc-increment field: reconstruct s(t) by interpolating the
    accumulated arc length s (recon[i][4]) between landmarks; e_s = max |s_tilde
    - s_true|. This is the sampling component only (no chord-vs-arc error)."""
    idx = sorted(set([0] + idx + [len(recon) - 1]))
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


def arc_v_error(recon, idx):
    """e_v from the arc-increment field: v_hat = Δs_k / T_k (segment-average
    speed), e_v = max_i |v_hat_i - v_i| against the recorded odometry speed."""
    idx = sorted(set([0] + idx + [len(recon) - 1]))
    mx = 0.0
    j = 0
    for i, (t, _x, _y, _psi, _s, v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        if i1 == i0:
            vh = 0.0
        else:
            t0, s0 = recon[i0][0], recon[i0][4]
            t1, s1 = recon[i1][0], recon[i1][4]
            vh = (s1 - s0) / (t1 - t0)
        mx = max(mx, abs(vh - v))
    return mx


def path_error(recon, idx):
    idx = sorted(set([0] + idx + [len(recon) - 1]))
    pts = [(recon[i][1], recon[i][2]) for i in idx]
    mx = 0.0
    for (_t, x, y, *_r) in recon:
        mx = max(mx, bb.dist_polyline(x, y, pts))
    return mx


def main():
    print("=" * 92)
    print("Distance--time channel ablation, arc-increment field d=Δs_k "
          "(Table tab:speed)")
    print("=" * 92)
    print(f"{'World':<6} {'method':<20} {'N':>3} {'path err':>9} "
          f"{'e_s (arc)':>10} {'e_v (arc)':>10}")
    for name in "ABC":
        recon = load_recon(name)
        methods = {
            "path-only": rb.dream_sampler(recon, r2_threshold=0.98,
                                          regress_speed=False),
            "path+speed": rb.dream_sampler(recon, r2_threshold=0.98,
                                           regress_speed=True),
            "sweep-angle": bb.sweep_angle(recon, angle=0.5),
            "sweep-angle+speed": bp.sweep_angle_speed(recon, angle=0.5,
                                                      r2_th=0.98),
        }
        for mname, idx in methods.items():
            n = len(sorted(set([0] + idx + [len(recon) - 1])))
            pe = path_error(recon, idx)
            es = arc_s_error(recon, idx)
            ev = arc_v_error(recon, idx)
            print(f"{name:<6} {mname:<20} {n:>3} {pe:>9.4f} {es:>10.4f} "
                  f"{ev:>10.4f}")
        print()


if __name__ == "__main__":
    main()
