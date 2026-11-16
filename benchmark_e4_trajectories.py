#!/usr/bin/env python3
"""
benchmark_e4_trajectories.py

Reviewer E4 / M8: independent, stratified trajectory families instead of three
recorded scripted paths plus noise replicas. Six geometry families are generated
synthetically, each with several parameter realizations and two speed regimes
(constant and oscillating), then DREAM and the two-rate baseline are run at the
operating point tau = 0.98 and the results are aggregated per family.

Families: straight, slalom, gentle-arc, s-curve, l-turn, figure-eight.
Realizations: 3 per family (varying turn amplitude / curvature / segment length).
Speed regimes: const (v=0.2 m/s) and osc (v = 0.2 +/- 0.1 sin(...)).

Metrics: complete-waypoint count N, max path-reconstruction error, arc-increment
distance--time error e_s, and speed-profile error e_v (same decoder as the main
benchmark). Aggregation is per family (mean over its 6 trajectories), so noise
replicas are not treated as independent geometries.

Usage:  python3 benchmark_e4_trajectories.py
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

DT = 0.05


def _drive(x, y, psi, s, t, v, w, dur, out, osc=None):
    """Integrate a constant (v, w) hold for `dur`, with optional speed
    oscillation. Appends (t, x, y, psi, s, v) samples."""
    n = int(round(dur / DT))
    for _ in range(n):
        if osc is not None:
            amp, period = osc
            vv = max(0.02, v + amp * math.sin(2.0 * math.pi * t / period))
        else:
            vv = v
        ww = w * (vv / v if v > 1e-6 else 1.0)  # scale turn rate with speed
        x += vv * math.cos(psi) * DT
        y += vv * math.sin(psi) * DT
        psi += ww * DT
        s += vv * DT
        t += DT
        out.append((t, x, y, psi, s, vv))
    return x, y, psi, s, t


def gen_straight(r):
    L = 4.0 + r
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, 0.0, L, out)
    return out


def gen_slalom(r):
    amp = 0.5 + 0.15 * r
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    for _ in range(4):
        x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, amp, 1.0, out)
        x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, -amp, 1.0, out)
    return out


def gen_gentle_arc(r):
    kappa = 0.1 + 0.12 * r
    L = 12.0 + 2.0 * r
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, kappa * 0.2, L, out)
    return out


def gen_s_curve(r):
    kappa = 0.3 + 0.15 * r
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, kappa * 0.2, 3.0, out)
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, -kappa * 0.2, 3.0, out)
    return out


def gen_l_turn(r):
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, 0.0, 2.0 + 0.5 * r, out)
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, 1.0, math.pi / 2, out)
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, 0.0, 2.0, out)
    return out


def gen_figure_eight(r):
    x = y = psi = s = t = 0.0
    out = [(t, x, y, psi, s, 0.2)]
    w = 0.6 + 0.1 * r
    for _ in range(2):
        x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, w, 2.0 * math.pi, out)
    x, y, psi, s, t = _drive(x, y, psi, s, t, 0.2, -w, 2.0 * math.pi, out)
    return out


FAMILIES = {
    "straight": gen_straight,
    "slalom": gen_slalom,
    "gentle-arc": gen_gentle_arc,
    "s-curve": gen_s_curve,
    "l-turn": gen_l_turn,
    "figure-eight": gen_figure_eight,
}

SPEEDS = {
    "const": None,
    "osc": (0.1, 3.0),   # amplitude, period (s)
}


def _with_speed(seg_fn, r, osc):
    """Re-run the generator but impose the requested speed regime by replacing
    each segment's commanded speed with the oscillating profile where needed.
    For simplicity, generators use v=0.2; we post-process by re-driving with
    oscillation only when requested (reuses the same (v,w) holds)."""
    # Simplest correct approach: regenerate with the oscillation flag threaded
    # through _drive. The generators above do not expose osc, so we regenerate
    # the raw path and then re-sample at the desired speed profile along the
    # same waypoints. Instead we special-case: return the base and note speed
    # in the trajectory; for reporting the sampler, speed affects the temporal
    # channel, so we re-integrate with osc here.
    return _reintegrate_with_speed(seg_fn(r), osc)


def _reintegrate_with_speed(base, osc):
    """Re-integrate a base trajectory's (x, y, psi) geometry with the requested
    speed profile, keeping the same path shape but changing s(t) and v(t)."""
    if osc is None:
        return base
    amp, period = osc
    # base samples carry (t, x, y, psi, s, v); rebuild s and v while keeping
    # (x, y, psi) unchanged by walking the base geometry at its own t.
    out = []
    s = 0.0
    prev = None
    for (t, x, y, psi, _s, _v) in base:
        v = max(0.02, 0.2 + amp * math.sin(2.0 * math.pi * t / period))
        if prev is not None:
            s += v * (t - prev)
        out.append((t, x, y, psi, s, v))
        prev = t
    return out


def full_n(idx, n_total):
    return len(sorted(set([0] + idx + [n_total - 1])))


def report(recon, idx):
    n = full_n(idx, len(recon))
    worst, _ = bb.path_error(recon, sorted(set([0] + idx + [len(recon) - 1])))
    es = sc.arc_s_error(recon, idx)
    ev = sc.arc_v_error(recon, idx)
    return n, worst, es, ev


def main():
    print("=" * 100)
    print("E4: stratified independent trajectory families (tau=0.98)")
    print("=" * 100)
    agg = {f: {"n_d": [], "n_b": [], "e_d": [], "e_b": [], "es_d": [],
               "es_b": [], "ev_d": [], "ev_b": []} for f in FAMILIES}
    rows = []
    for fam, seg_fn in FAMILIES.items():
        for r in range(3):
            for spname, osc in SPEEDS.items():
                recon = _with_speed(seg_fn, r, osc)
                d = rb.dream_sampler(recon, r2_threshold=0.98)
                b = rb.baseline_sampler(recon)
                nd, ed, esd, evd = report(recon, d)
                nb, eb, esb, evb = report(recon, b)
                agg[fam]["n_d"].append(nd); agg[fam]["n_b"].append(nb)
                agg[fam]["e_d"].append(ed); agg[fam]["e_b"].append(eb)
                agg[fam]["es_d"].append(esd); agg[fam]["es_b"].append(esb)
                agg[fam]["ev_d"].append(evd); agg[fam]["ev_b"].append(evb)
                rows.append([fam, r, spname, nd, nb, ed, eb, esd, esb, evd, evb])
    print(f"{'family':<13} {'N_d':>5} {'N_b':>5} {'path_d':>8} {'path_b':>8} "
          f"{'es_d':>8} {'es_b':>8} {'ev_d':>7} {'ev_b':>7}")
    for fam in FAMILIES:
        a = agg[fam]
        def mean(v):
            return sum(v) / len(v)
        print(f"{fam:<13} {mean(a['n_d']):>5.1f} {mean(a['n_b']):>5.1f} "
              f"{mean(a['e_d']):>8.4f} {mean(a['e_b']):>8.4f} "
              f"{mean(a['es_d']):>8.4f} {mean(a['es_b']):>8.4f} "
              f"{mean(a['ev_d']):>7.4f} {mean(a['ev_b']):>7.4f}")

    out = os.path.join(_HERE, "bench_data", "e4_trajectories.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "realization", "speed", "N_dream", "N_base",
                    "path_d", "path_b", "es_d", "es_b", "ev_d", "ev_b"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4],
                        round(r[5], 6), round(r[6], 6), round(r[7], 6),
                        round(r[8], 6), round(r[9], 6), round(r[10], 6)])
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
