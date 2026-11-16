#!/usr/bin/env python3
"""
benchmark_e3_timeaware.py

Reviewer E3: put the dual-linearity trigger against time-aware spatiotemporal
baselines under one payload/decoder. For each world (B, C) and the synthetic
speed-varying stress profile we run, with the SAME arc-increment decoder:

  * DREAM path-only          (tau = 0.98, geometry only)
  * DREAM path+speed         (tau = 0.98, dual-linearity)
  * sweep-abs-angle+speed    (cumulative |dpsi| + distance--time R^2)
  * segment-error+speed      (causal perpendicular-distance + distance--time R^2)
  * OPW-TR                   (online time-ratio distance, eps = 0.05)
  * OPW-TR+speed             (time-ratio distance + distance--time R^2)
  * SQUISH-E                 (budget-based synchronized-distance, budget = DREAM N)

and report the complete-waypoint count N, the max path-reconstruction error,
the arc-increment distance--time error e_s, and the speed-profile error e_v.
This answers whether the temporal channel's benefit survives a time-aware
geometric competitor rather than only a pure-geometry trigger.

Usage:  python3 benchmark_e3_timeaware.py
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
import dream_core                          # noqa: E402


def _speed_hybrid(recon, geom_fn, r2_th):
    """Combine a geometric trigger `geom_fn` (which itself returns landmark
    indices and resets on each emission) with the distance--time R^2 channel.
    For simplicity the geometric state is rebuilt from the segment since the
    last landmark, matching the online reset semantics of DREAM's speed channel.
    `geom_fn(recon, start, i)` returns True when a landmark must be emitted at
    index i, given the segment since `start`."""
    speed = dream_core.LinFitAccumulator()
    idx = [0]
    anchor_t = recon[0][0]
    anchor_i = 0
    n_samples = 1
    for i, (t, _x, _y, _psi, s, _v) in enumerate(recon):
        if i == 0:
            continue
        speed.add(t - anchor_t, s)
        n_samples += 1
        if n_samples < 5:
            continue
        trig_geo = geom_fn(recon, anchor_i, i)
        trig_speed = speed.r2() < r2_th
        if trig_geo or trig_speed:
            idx.append(i)
            anchor_t = t
            anchor_i = i
            speed = dream_core.LinFitAccumulator()
            speed.add(0.0, s)
            n_samples = 1
    return idx


def _abs_turn_geo(recon, start, i):
    """Cumulative absolute turning since index `start` exceeds 1.5 rad (B) /
    3.0 rad (C); a fixed representative threshold is chosen per world below."""
    angle = 1.5 if recon[-1][1] > 2.0 else 3.0  # crude world selector
    cum = 0.0
    last = recon[start][3]
    for k in range(start + 1, i + 1):
        d = recon[k][3] - last
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        cum += abs(d)
        last = recon[k][3]
    return cum >= angle


def _seg_err_geo(recon, start, i):
    """Causal perpendicular segment-error trigger: max perpendicular distance of
    the buffered points to the anchor->current segment exceeds 0.05 m."""
    eps = 0.05
    ax, ay = recon[start][1], recon[start][2]
    x, y = recon[i][1], recon[i][2]
    dx, dy = x - ax, y - ay
    den = dx * dx + dy * dy
    worst = 0.0
    for k in range(start + 1, i + 1):
        bx, by = recon[k][1], recon[k][2]
        if den < 1e-12:
            d = math.hypot(bx - ax, by - ay)
        else:
            u = ((bx - ax) * dx + (by - ay) * dy) / den
            u = min(max(u, 0.0), 1.0)
            d = math.hypot(bx - (ax + u * dx), by - (ay + u * dy))
        worst = max(worst, d)
    return worst > eps


def _opw_geo(recon, start, i):
    """Time-ratio distance since `start` exceeds 0.05 m (OPW-TR criterion)."""
    eps = 0.05
    ts, te = recon[start][0], recon[i][0]
    xs, ys = recon[start][1], recon[start][2]
    xe, ye = recon[i][1], recon[i][2]
    dte = te - ts
    worst = 0.0
    for k in range(start + 1, i):
        ti = recon[k][0]
        f = (ti - ts) / dte if dte > 1e-12 else 0.0
        xp = xs + f * (xe - xs)
        yp = ys + f * (ye - ys)
        d = math.hypot(recon[k][1] - xp, recon[k][2] - yp)
        worst = max(worst, d)
    return worst > eps


def full_n(idx, n_total):
    return len(sorted(set([0] + idx + [n_total - 1])))


def report(recon, idx):
    n = full_n(idx, len(recon))
    worst, _ = bb.path_error(recon, sorted(set([0] + idx + [len(recon) - 1])))
    es = sc.arc_s_error(recon, idx)
    ev = sc.arc_v_error(recon, idx)
    return n, worst, es, ev


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def stress_recon():
    segs = [
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
        (0.2, 0.0, 2.0), (0.2, 0.0, 1.0),
        (0.05, 0.0, 1.5), (0.05, 0.3, 2.0), (0.05, 0.0, 1.0),
        (0.15, 0.0, 1.0), (0.15, -0.3, 2.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0),
    ]
    dt = 0.05
    raw = []
    x = y = psi = s = t = 0.0
    v_prev = 0.0
    for v_end, w, dur in segs:
        n = int(round(dur / dt))
        for j in range(n):
            f = (j + 1) / n
            v = v_prev + (v_end - v_prev) * f
            x += v * math.cos(psi) * dt
            y += v * math.sin(psi) * dt
            psi += w * dt
            t += dt
            raw.append((t, x, y, v, w))
        v_prev = v_end
    return rb.reconstruct(raw)


def main():
    worlds = [("B", load_recon("B")), ("C", load_recon("C")),
              ("stress", stress_recon())]
    print("=" * 96)
    print("E3: dual-linearity vs time-aware baselines (arc decoder, boundary in N)")
    print("=" * 96)
    print(f"{'world':<8} {'method':<24} {'N':>4} {'path err':>9} "
          f"{'e_s arc':>8} {'e_v':>8}")
    all_rows = []
    for name, recon in worlds:
        po = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
        ps = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
        abs_spd = _speed_hybrid(recon, _abs_turn_geo, 0.98)
        seg_spd = _speed_hybrid(recon, _seg_err_geo, 0.98)
        opw = bb.opw_tr(recon, eps=0.05)
        opw_spd = _speed_hybrid(recon, _opw_geo, 0.98)
        squish = bb.squish(recon, full_n(ps, len(recon)))
        for meth, idx in [("DREAM path-only", po), ("DREAM path+speed", ps),
                          ("sweep-abs+speed", abs_spd),
                          ("segment-error+speed", seg_spd),
                          ("OPW-TR", opw), ("OPW-TR+speed", opw_spd),
                          ("SQUISH-E (mbudget)", squish)]:
            n, worst, es, ev = report(recon, idx)
            all_rows.append((name, meth, n, worst, es, ev))
            print(f"{name:<8} {meth:<24} {n:>4d} {worst:>9.4f} {es:>8.4f} "
                  f"{ev:>8.4f}")
        print("-" * 96)

    out = os.path.join(_HERE, "bench_data", "e3_timeaware.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "method", "N", "path_err", "e_s_arc", "e_v"])
        for r in all_rows:
            w.writerow([r[0], r[1], r[2], round(r[3], 6), round(r[4], 6),
                        round(r[5], 6)])
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
