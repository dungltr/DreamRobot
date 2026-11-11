#!/usr/bin/env python3
"""
benchmark_mc1_speedprofiles.py

Matched-budget online comparison of the temporal channel (reviewer MC1):
hold the geometric trigger fixed and toggle the temporal trigger, across
several independently generated speed profiles and trajectory geometries,
with uncertainty (Monte-Carlo noise runs).

For each profile we compare four online configurations:
  * DREAM path-only   (geometric trigger only, regress_speed=False)
  * DREAM path+speed  (geometric + distance--time trigger)
  * sweep-angle       (scalar geometric trigger only)
  * sweep-angle+speed (scalar geometric + distance--time trigger)
and report, at each configuration's own landmark count, the arc-increment
distance--time reconstruction error e_s and the speed-profile error e_v, plus
the incremental landmark cost of turning the temporal trigger on.

A matched-budget view is then formed by lowering R^2_th of path-only until it
reaches (or exceeds) the path+speed count, so the temporal channel is compared
against "just more geometric landmarks" rather than against an unconstrained
baseline.

Usage:  python3 benchmark_mc1_speedprofiles.py
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
import benchmark_baselines_gazebo as bb    # noqa: E402
import benchmark_pareto as bp              # noqa: E402
import benchmark_speed_channel as sc       # noqa: E402

DT = 0.05
RUNS = 30
SIGMA_V, SIGMA_W = 0.01, 0.02


def synth_profile(segs):
    """Integrate a piecewise (v_end, w, dur) profile with linear velocity ramps
    into a recon list (t, x, y, psi, s, v)."""
    x = y = psi = s = t = 0.0
    out = []
    for (v_end, w, dur) in segs:
        n = int(round(dur / DT))
        v_prev = out[-1][5] if out else 0.0
        for j in range(n):
            f = (j + 1) / n
            v = v_prev + (v_end - v_prev) * f
            x += v * math.cos(psi) * DT
            y += v * math.sin(psi) * DT
            psi += w * DT
            s += v * DT
            t += DT
            out.append((t, x, y, psi, s, v))
    return out


# Independent speed profiles: different speed-change sequences, some on a
# straight line, some on curves, some with stops and reversals of curvature.
PROFILES = {
    "accel-decel": [(0.0, 0.0, 1.0), (0.20, 0.0, 2.0), (0.20, 0.0, 1.0),
                    (0.05, 0.0, 1.5), (0.05, 0.3, 2.0), (0.15, 0.0, 1.0),
                    (0.15, -0.3, 2.0), (0.0, 0.0, 1.0)],
    "cruise-stops": [(0.20, 0.0, 2.0), (0.0, 0.0, 1.0), (0.15, 0.0, 2.0),
                     (0.0, 0.0, 1.0), (0.20, 0.0, 2.0), (0.0, 0.0, 1.0)],
    "ramp-down-turn": [(0.20, 0.0, 1.5), (0.12, 0.4, 2.0), (0.12, -0.4, 2.0),
                       (0.20, 0.0, 1.5), (0.06, 0.0, 2.0)],
    "sine-speed-slalom": [(0.20, 0.35, 2.0), (0.08, 0.35, 2.0),
                          (0.16, -0.35, 2.0), (0.10, -0.35, 2.0),
                          (0.20, 0.35, 2.0)],
    "double-accel": [(0.05, 0.0, 2.0), (0.20, 0.0, 2.0), (0.20, 0.0, 1.0),
                     (0.10, 0.0, 2.0), (0.20, 0.0, 2.0), (0.05, 0.0, 2.0)],
}


def recon_from_clean(profile):
    return synth_profile(PROFILES[profile])


def noisy_recon(clean, rng):
    """Re-integrate the clean profile under odometry noise (same two-state
    forward model as the main study)."""
    raw = [(t, x, y, v, 0.0) for (t, x, y, psi, s, v) in clean]
    # rebuild raw with angular rate reconstructed from psi increments
    raw2 = []
    prev_psi = 0.0
    prev_t = None
    for (t, x, y, psi, s, v) in clean:
        w = 0.0
        if prev_t is not None and (t - prev_t) > 0:
            d = psi - prev_psi
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            w = d / (t - prev_t)
        raw2.append((t, x, y, v, w))
        prev_psi = psi
        prev_t = t
    est = []
    x = y = psi = s = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw2:
        dt = t - prev if prev is not None else 0.0
        vn = max(0.0, v + SIGMA_V * rng.gauss(0.0, 1.0))
        wn = w + SIGMA_W * rng.gauss(0.0, 1.0)
        if prev is not None and 0.0 < dt < 0.5:
            psi += wn * dt
            x += vn * math.cos(psi) * dt
            y += vn * math.sin(psi) * dt
            s += vn * dt
        est.append((t, x, y, psi, s, vn))
        prev = t
    return est


def n_es_ev(recon, idx):
    n = len(sorted(set([0] + idx + [len(recon) - 1])))
    return n, sc.arc_s_error(recon, idx), sc.arc_v_error(recon, idx)


def matched_budget_path_only(recon, target_n, r2_hi=0.98, r2_lo=0.80):
    """Lower R^2_th of path-only until its count reaches >= target_n (the
    path+speed count), so the temporal channel is compared against 'just more
    geometric landmarks' at matched budget."""
    best = None
    th = r2_hi
    while th >= r2_lo:
        idx = rb.dream_sampler(recon, r2_threshold=th, regress_speed=False)
        n = len(sorted(set([0] + idx + [len(recon) - 1])))
        if n >= target_n:
            best = (th, n, sc.arc_s_error(recon, idx))
            break
        th -= 0.01
    return best


def main():
    print("=" * 100)
    print(f"MC1 matched-budget temporal-channel comparison "
          f"({RUNS} noisy runs, sigma_v={SIGMA_V}, sigma_w={SIGMA_W})")
    print("=" * 100)
    print(f"{'profile':<18} {'method':<22} {'N':>4} {'e_s arc':>9} {'e_v':>8} "
          f"{'dN(+speed)':>10}")
    for pname in PROFILES:
        clean = recon_from_clean(pname)
        # aggregate over noisy runs
        agg = {"path-only": [], "path+speed": [], "sweep": [], "sweep+speed": [],
               "mb-path-only": []}
        for run in range(RUNS):
            rng = random.Random(7000 + run)
            recon = noisy_recon(clean, rng)
            po = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
            ps = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
            sw = bb.sweep_angle(recon, angle=0.5)
            sws = bp.sweep_angle_speed(recon, angle=0.5, r2_th=0.98)
            n_po, es_po, ev_po = n_es_ev(recon, po)
            n_ps, es_ps, ev_ps = n_es_ev(recon, ps)
            n_sw, es_sw, ev_sw = n_es_ev(recon, sw)
            n_sws, es_sws, ev_sws = n_es_ev(recon, sws)
            agg["path-only"].append((n_po, es_po, ev_po))
            agg["path+speed"].append((n_ps, es_ps, ev_ps))
            agg["sweep"].append((n_sw, es_sw, ev_sw))
            agg["sweep+speed"].append((n_sws, es_sws, ev_sws))
            mb = matched_budget_path_only(recon, n_ps)
            if mb:
                agg["mb-path-only"].append((mb[1], mb[2]))

        def summ(key):
            rows = agg[key]
            if not rows:
                return None
            n = sum(r[0] for r in rows) / len(rows)
            es = sum(r[1] for r in rows) / len(rows)
            ev = sum(r[2] for r in rows) / len(rows)
            return n, es, ev

        po = summ("path-only")
        ps = summ("path+speed")
        sw = summ("sweep")
        sws = summ("sweep+speed")
        mb = summ("mb-path-only")
        dN = (ps[0] - po[0]) if po and ps else float("nan")
        es_ratio = po[1] / ps[1] if po and ps else float("nan")
        print(f"{pname:<18} {'path-only':<22} {po[0]:>4.1f} {po[1]:>9.4f} "
              f"{po[2]:>8.4f} {0:>10.1f}")
        print(f"{pname:<18} {'path+speed':<22} {ps[0]:>4.1f} {ps[1]:>9.4f} "
              f"{ps[2]:>8.4f} {dN:>10.1f}")
        print(f"{pname:<18} {'sweep-angle':<22} {sw[0]:>4.1f} {sw[1]:>9.4f} "
              f"{sw[2]:>8.4f} {0:>10.1f}")
        print(f"{pname:<18} {'sweep-angle+speed':<22} {sws[0]:>4.1f} "
              f"{sws[1]:>9.4f} {sws[2]:>8.4f} {sws[0]-sw[0]:>10.1f}")
        if mb:
            print(f"{pname:<18} {'path-only@mbudget':<22} {mb[0]:>4.1f} "
                  f"{mb[1]:>9.4f} {'-':>8} {'(>=N_ps)':>10}")
        print(f"  -> e_s ratio path/+speed = {es_ratio:.3f}x")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
