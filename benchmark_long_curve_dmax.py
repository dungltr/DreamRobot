#!/usr/bin/env python3
"""
benchmark_long_curve_dmax.py

Long gentle-curve / segment-length-flush (D_max) study for reviewer E1
(DREAM_Revision4_Section_by_Section_Revision_Plan.md). Quantifies the
finite-window missed-event failure: on a long, low-curvature segment the
capped fitting window can stay collinear while the accumulated curvature (or
length) grows, so the geometric channel never fires and the waypoint set
becomes sparse. The optional segment-length flush D_max emits a landmark
every D_max metres, but it only bounds the inter-landmark interval -- it is
NOT a geometric-error bound or a collision-avoidance guarantee.

Trajectories are generated at CONSTANT speed (V) so the distance--time
channel is silent (R^2_s = 1) and only the geometric channel is exercised.
For each trajectory and each D_max in {off, 1, 2, 4} m the script reports:
  * N_sampler : raw sampler output (start + regression + flush events)
  * N_stored  : complete waypoint set {start} + sampler output + {end}
  * max path error : maximum perpendicular distance from recorded samples to
                     the decoded landmark polyline (one-sided Hausdorff)
  * mean path error
  * e_s (arc) : distance--time reconstruction error from the arc-increment
                payload (accumulated travelled distance interpolated in time)

Offline, no ROS/Gazebo needed:  python3 benchmark_long_curve_dmax.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_baselines_gazebo as bb    # noqa: E402
import benchmark_speed_channel as sc       # noqa: E402

V = 0.2        # constant speed (m/s)
DT = 0.02      # odometry time step (s)  -> 50 Hz


def raw_constant_v(length_m, w_of_f):
    """Constant-speed raw odometry stream (t, x, y, v, w); w = w_of_f(f) with
    f in [0,1] the fraction of the travelled distance."""
    out = []
    x = y = psi = 0.0
    n = max(1, int(round(length_m / V / DT)))
    for j in range(n):
        f = j / max(1, n - 1)
        w = w_of_f(f) if callable(w_of_f) else w_of_f
        x += V * math.cos(psi) * DT
        y += V * math.sin(psi) * DT
        psi += w * DT
        out.append((out[-1][0] + DT if out else DT, x, y, V, w))
    return out


def straight(length_m):
    return raw_constant_v(length_m, 0.0)


def arc(kappa, length_m):
    return raw_constant_v(length_m, kappa * V)


def scurve(kappa, length_m):
    return raw_constant_v(length_m, lambda f: (kappa * V) if (int(f * 6) % 2 == 0) else (-kappa * V))


def var_curvature(kappa_max, length_m):
    # triangular curvature profile 0 -> kappa_max -> 0
    return raw_constant_v(length_m, lambda f: kappa_max * V * (1.0 - abs(2.0 * f - 1.0)))


def analyze(name, raw, dmax_values):
    recon = rb.reconstruct(raw)
    rows = []
    for dmax in dmax_values:
        idx = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True,
                               max_window_size=500, max_window_arc=None,
                               max_segment_length=dmax)
        n_sampler = len(idx)
        full = sorted(set([0] + idx + [len(recon) - 1]))
        n_stored = len(full)
        worst, mean = bb.path_error(recon, full)
        es = sc.arc_s_error(recon, idx)
        rows.append((name, dmax, n_sampler, n_stored, worst, mean, es))
    return rows


def main():
    dmax_values = [None, 1.0, 2.0, 4.0]
    trajectories = [
        ("straight 20 m", straight(20.0)),
        ("straight 40 m", straight(40.0)),
        ("straight 100 m", straight(100.0)),
        ("arc k=0.1 20 m", arc(0.1, 20.0)),
        ("arc k=0.3 20 m", arc(0.3, 20.0)),
        ("arc k=0.5 20 m", arc(0.5, 20.0)),
        ("S-curve k=0.3 40 m", scurve(0.3, 40.0)),
        ("var-curv 40 m", var_curvature(0.5, 40.0)),
    ]
    header = ["trajectory", "D_max (m)", "N_sampler", "N_stored",
              "max path err (m)", "mean path err (m)", "e_s arc (m)"]
    print("=" * 104)
    print("Long gentle-curve / D_max sweep (R2_th=0.98, m_max=500, no arc cap, const speed)")
    print("=" * 104)
    print(f"{header[0]:20s} {header[1]:>8s} {header[2]:>9s} {header[3]:>9s} "
          f"{header[4]:>15s} {header[5]:>16s} {header[6]:>11s}")
    all_rows = []
    for name, raw in trajectories:
        rows = analyze(name, raw, dmax_values)
        all_rows.extend(rows)
        for r in rows:
            dlab = "off" if r[1] is None else f"{r[1]:.0f}"
            print(f"{r[0]:20s} {dlab:>8s} {r[2]:>9d} {r[3]:>9d} "
                  f"{r[4]:>15.4f} {r[5]:>16.4f} {r[6]:>11.4f}")
        print("-" * 104)

    out = os.path.join(_HERE, "bench_data", "long_curve_dmax.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in all_rows:
            w.writerow([r[0], "" if r[1] is None else r[1], r[2], r[3],
                        round(r[4], 6), round(r[5], 6), round(r[6], 6)])
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
