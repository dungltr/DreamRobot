#!/usr/bin/env python3
"""
benchmark_e3_frontier.py

Reviewer E3: joint N--path and N--e_s frontier sweeps over the 36 independent
trajectory configurations of benchmark_e4_trajectories.py. Each online method is
swept over its own parameter grid (a per-method threshold/budget sweep, not an
exact matched-count comparison); the per-config (N, path error, e_s, e_v) are
aggregated as means over the configurations (realizations are the independent
unit). The geometry frontier uses all 36 configurations; the temporal frontier
uses the 18 oscillating-speed configurations (e_s is identically zero at
constant speed and does not discriminate).

Usage:  python3 benchmark_e3_frontier.py
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
import benchmark_e4_trajectories as e4     # noqa: E402
import dream_core                          # noqa: E402


# ---------------------------------------------------------------------------
# Hybrid baselines (geometric trigger + distance--time R^2 channel)
# ---------------------------------------------------------------------------
def sweep_abs_angle_speed(recon, angle=0.5, r2_th=0.98):
    """Cumulative-absolute-turning trigger augmented with the distance--time
    R^2 channel: emit when EITHER the total |heading change| since the previous
    landmark exceeds `angle`, OR s(t) becomes non-linear (R^2 < r2_th)."""
    speed = dream_core.LinFitAccumulator()
    idx = [0]
    anchor_t = recon[0][0]
    cum = 0.0
    last_psi = recon[0][3]
    n = 1
    for i, (t, _x, _y, psi, s, _v) in enumerate(recon):
        if i == 0:
            continue
        d = psi - last_psi
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        cum += abs(d)
        last_psi = psi
        speed.add(t - anchor_t, s)
        n += 1
        if n < 5:
            continue
        if cum >= angle or speed.r2() < r2_th:
            idx.append(i)
            cum = 0.0
            anchor_t = t
            speed = dream_core.LinFitAccumulator()
            speed.add(0.0, s)
            n = 1
    return idx


def segment_error_speed(recon, eps=0.05, r2_th=0.98):
    """Perpendicular segment-error trigger augmented with the distance--time
    R^2 channel: emit when EITHER the max perpendicular distance to the current
    segment exceeds `eps`, OR s(t) becomes non-linear (R^2 < r2_th)."""
    speed = dream_core.LinFitAccumulator()
    idx = [0]
    anchor = 0
    anchor_t = recon[0][0]
    n = 1
    for e in range(1, len(recon)):
        xs, ys = recon[anchor][1], recon[anchor][2]
        xe, ye = recon[e][1], recon[e][2]
        seg_len2 = (xe - xs) ** 2 + (ye - ys) ** 2
        worst = 0.0
        for i in range(anchor + 1, e):
            px, py = recon[i][1], recon[i][2]
            if seg_len2 < 1e-12:
                d = math.hypot(px - xs, py - ys)
            else:
                u = ((px - xs) * (xe - xs) + (py - ys) * (ye - ys)) / seg_len2
                u = max(0.0, min(1.0, u))
                d = math.hypot(px - (xs + u * (xe - xs)),
                               py - (ys + u * (ye - ys)))
            worst = max(worst, d)
        speed.add(recon[e][0] - anchor_t, recon[e][4])
        n += 1
        if n < 5:
            continue
        if worst > eps or speed.r2() < r2_th:
            idx.append(e - 1)
            anchor = e - 1
            anchor_t = recon[e - 1][0]
            speed = dream_core.LinFitAccumulator()
            speed.add(0.0, recon[e - 1][4])
            n = 1
    return idx


def full_n(idx, n_total):
    return len(sorted(set([0] + idx + [n_total - 1])))


def metrics(recon, idx):
    i = sorted(set([0] + idx + [len(recon) - 1]))
    return full_n(idx, len(recon)), sc.path_error(recon, idx), \
        sc.arc_s_error(recon, idx), sc.arc_v_error(recon, idx)


def gen_configs():
    out = []
    for fam, seg_fn in e4.FAMILIES.items():
        for r in range(3):
            for spname, osc in e4.SPEEDS.items():
                recon = e4._with_speed(seg_fn, r, osc)
                out.append((fam, r, spname, recon))
    return out


# Method definitions: name -> list of (label, callable(recon)->idx)
def method_sweeps():
    m = []
    for tau in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995):
        m.append((f"DREAM tau={tau}",
                  lambda r, tau=tau: rb.dream_sampler(r, r2_threshold=tau,
                                                     regress_speed=True)))
        m.append((f"DREAM-path tau={tau}",
                  lambda r, tau=tau: rb.dream_sampler(r, r2_threshold=tau,
                                                     regress_speed=False)))
    for a in (0.3, 0.5, 0.8, 1.2, 1.6):
        m.append((f"abs+speed a={a}", lambda r, a=a: sweep_abs_angle_speed(r, a)))
    for eps in (0.02, 0.05, 0.08, 0.12, 0.16):
        m.append((f"seg+speed e={eps}",
                  lambda r, eps=eps: segment_error_speed(r, eps)))
        m.append((f"OPW-TR e={eps}", lambda r, eps=eps: bb.opw_tr(r, eps)))
    for b in (3, 6, 9, 12, 18):
        m.append((f"SQUISH-E k={b}", lambda r, b=b: bb.squish(r, b)))
    return m


def pareto(points):
    """Return the set of nondominated indices of (x, y) with both minimized."""
    keep = []
    for i, (x, y) in enumerate(points):
        dom = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if xj <= x and yj <= y and (xj < x or yj < y):
                dom = True
                break
        if not dom:
            keep.append(i)
    return keep


def main():
    configs = gen_configs()
    osc_configs = [c for c in configs if c[2] == "osc"]
    sweeps = method_sweeps()

    rows = []          # per (method, config) -> (N, path, e_s, e_v)
    for (fam, r, sp, recon) in configs:
        for label, fn in sweeps:
            idx = fn(recon)
            n, p, es, ev = metrics(recon, idx)
            rows.append((label, fam, r, sp, n, p, es, ev))

    # aggregate means
    from collections import defaultdict
    geo = defaultdict(lambda: [0.0, 0.0])       # label -> [sumN, sumPath]
    tmp = defaultdict(lambda: [0.0, 0.0])       # label -> [sumN, sumEs] over osc
    cnt_all = len(configs)
    cnt_osc = len(osc_configs)
    for (label, fam, r, sp, n, p, es, ev) in rows:
        geo[label][0] += n
        geo[label][1] += p
        if sp == "osc":
            tmp[label][0] += n
            tmp[label][1] += es

    print("=" * 100)
    print("E3: joint frontier sweeps over 36 configurations")
    print("=" * 100)
    print(f"{'method':<22} | {'N(geo)':>7} {'path(geo)':>9} | "
          f"{'N(osc)':>7} {'e_s(osc)':>9} | frontier")
    geo_pts, geo_labels = [], []
    tmp_pts, tmp_labels = [], []
    for label in sorted(geo):
        gN = geo[label][0] / cnt_all
        gP = geo[label][1] / cnt_all
        tN = tmp[label][0] / cnt_osc
        tEs = tmp[label][1] / cnt_osc
        geo_pts.append((gN, gP)); geo_labels.append(label)
        tmp_pts.append((tN, tEs)); tmp_labels.append(label)
    gfront = set(pareto(geo_pts))
    tfront = set(pareto(tmp_pts))
    for i, label in enumerate(sorted(geo)):
        gN = geo[label][0] / cnt_all
        gP = geo[label][1] / cnt_all
        tN = tmp[label][0] / cnt_osc
        tEs = tmp[label][1] / cnt_osc
        f = ("G" if i in gfront else "-") + ("T" if i in tfront else "-")
        print(f"{label:<22} | {gN:>7.2f} {gP:>9.4f} | "
              f"{tN:>7.2f} {tEs:>9.4f} | {f}")

    out = os.path.join(_HERE, "bench_data", "e3_frontier.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "family", "realization", "speed",
                    "N", "path_err", "e_s", "e_v"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4],
                        round(r[5], 6), round(r[6], 6), round(r[7], 6)])
    print(f"\n[OK] wrote {out} ({len(rows)} method-config rows)")
    print("frontier legend: G = on N--path frontier, T = on N--e_s frontier")


if __name__ == "__main__":
    main()
