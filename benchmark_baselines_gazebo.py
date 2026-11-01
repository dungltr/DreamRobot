#!/usr/bin/env python3
"""
benchmark_baselines_gazebo.py

Offline comparison of DREAM against four baselines (uniform, Douglas--Peucker,
send-on-delta, Shannon--Nyquist) on the *recorded Gazebo odometry*
(bench_data/odom_{A,B,C}.csv) rather than on a synthetic trajectory.

For every world it reports, per method, the landmark count N and the path /
heading reconstruction errors used in the "offline baselines" table of the
manuscript.

  --mc            Monte-Carlo noise re-integration (mean +/- 95% CI)
  --runs N        number of Monte-Carlo runs (default 20)
  --noise SIGMA   angular-rate noise sigma in rad/s (default 0.02)

Usage:
  python3 benchmark_baselines_gazebo.py           # clean single-run comparison
  python3 benchmark_baselines_gazebo.py --mc      # Monte-Carlo comparison
"""

import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402


def uniform(recon, period=1.0):
    idx = []
    last = -1e9
    for i, (t, *_rest) in enumerate(recon):
        if t - last >= period:
            idx.append(i)
            last = t
    return idx


def douglas_peucker(recon, eps=0.05):
    pts = [(p[1], p[2]) for p in recon]
    n = len(pts)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True

    def dist(i, a, b):
        ax, ay = pts[a]
        bx, by = pts[b]
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        if den < 1e-12:
            return math.hypot(pts[i][0] - ax, pts[i][1] - ay)
        u = ((pts[i][0] - ax) * dx + (pts[i][1] - ay) * dy) / den
        u = min(max(u, 0.0), 1.0)
        return math.hypot(pts[i][0] - (ax + u * dx), pts[i][1] - (ay + u * dy))

    def rec(lo, hi):
        if hi <= lo + 1:
            return
        m = max(range(lo + 1, hi), key=lambda i: dist(i, lo, hi))
        if dist(m, lo, hi) > eps:
            keep[m] = True
            rec(lo, m)
            rec(m, hi)

    rec(0, n - 1)
    return [i for i in range(n) if keep[i]]


def send_on_delta(recon, delta=0.05):
    idx = [0]
    lx, ly = recon[0][1], recon[0][2]
    for i, p in enumerate(recon):
        if math.hypot(p[1] - lx, p[2] - ly) >= delta:
            idx.append(i)
            lx, ly = p[1], p[2]
    return idx


def heading_rmse(recon, idx):
    if len(idx) < 2:
        return float("inf")
    err = 0.0
    n = 0
    j = 0
    for i, (_t, _x, _y, psi, _s, _v) in enumerate(recon):
        while j + 1 < len(idx) and idx[j + 1] <= i:
            j += 1
        if j + 1 >= len(idx):
            break
        i0, i1 = idx[j], idx[j + 1]
        frac = (i - i0) / (i1 - i0) if i1 > i0 else 0.0
        ph = recon[i0][3] + frac * (recon[i1][3] - recon[i0][3])
        err += (ph - psi) ** 2
        n += 1
    return math.sqrt(err / n) if n else float("inf")


def dist_polyline(x, y, seg):
    best = float("inf")
    for (ax, ay), (bx, by) in zip(seg, seg[1:]):
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        u = 0.0 if den < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / den
        u = min(max(u, 0.0), 1.0)
        best = min(best, math.hypot(x - (ax + u * dx), y - (ay + u * dy)))
    return best


def path_error(recon, idx):
    if len(idx) < 2:
        return float("inf"), float("inf")
    seg = [(recon[i][1], recon[i][2]) for i in idx]
    worst = 0.0
    total = 0.0
    n = 0
    for p in recon:
        d = dist_polyline(p[1], p[2], seg)
        worst = max(worst, d)
        total += d
        n += 1
    return worst, (total / n if n else 0.0)


def load_raw(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    return raw


def mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, z * math.sqrt(var / len(values))


METHODS = {
    "uniform (1 Hz)": uniform,
    "douglas-peucker (0.05 m)": douglas_peucker,
    "send-on-delta (0.05 m)": send_on_delta,
    "nyquist (0.5/1.0 Hz)": rb.baseline_sampler,
    "dream (R2_th=0.98)": rb.dream_sampler,
}


def evaluate(recon, fn):
    n_total = len(recon)
    idx = fn(recon)
    # Every method must include the trajectory endpoint for a fair full-path
    # reconstruction comparison.
    if not idx or idx[-1] != n_total - 1:
        idx = idx + [n_total - 1]
    pmax, pmean = path_error(recon, idx)
    pr = heading_rmse(recon, idx)
    return len(idx), pmax, pmean, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", action="store_true")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.02)
    args = ap.parse_args()

    print("=" * 78)
    if args.mc:
        print(f"Offline baselines on recorded Gazebo odometry, Monte-Carlo "
              f"({args.runs} runs, sigma_v=0.01, sigma_w={args.noise})")
    else:
        print("Offline baselines on recorded Gazebo odometry (single run)")
    print("=" * 78)

    for name in "ABC":
        raw = load_raw(name)
        print(f"--- World {name} ---")
        print(f"{'method':<26} | {'N':>8} | {'path_max':>16} | {'psi_RMSE':>16}")
        for mname, fn in METHODS.items():
            if args.mc:
                ns, pms, prs = [], [], []
                for run in range(args.runs):
                    rng = random.Random(9000 + run)
                    recon = rb.reconstruct_noisy(raw, 0.01, args.noise, rng)
                    n, pm, _pmn, pr = evaluate(recon, fn)
                    ns.append(n)
                    pms.append(pm)
                    prs.append(pr)
                nm, nc = mean_ci(ns)
                pmm, pmc = mean_ci(pms)
                prm, prc = mean_ci(prs)
                print(f"{mname:<26} | {nm:>6.1f}\u00b1{nc:.1f} | "
                      f"{pmm:>7.3f}\u00b1{pmc:.3f} | {prm:>7.3f}\u00b1{prc:.3f}")
            else:
                recon = rb.reconstruct(raw)
                n, pm, _pmn, pr = evaluate(recon, fn)
                print(f"{mname:<26} | {n:>8} | {pm:>16.4f} | {pr:>16.4f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
