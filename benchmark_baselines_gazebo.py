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


def sweep_angle(recon, angle=0.5):
    """Online geometric baseline: record a landmark whenever the cumulative
    heading change (sweep angle) since the last landmark exceeds `angle` (rad)."""
    idx = [0]
    cum = 0.0
    last_psi = recon[0][3]
    for i, (_t, _x, _y, psi, _s, _v) in enumerate(recon):
        if i == 0:
            continue
        d = psi - last_psi
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        cum += d
        last_psi = psi
        if abs(cum) >= angle:
            idx.append(i)
            cum = 0.0
    return idx


def sweep_abs_angle(recon, angle=0.5):
    """Online geometric baseline: record a landmark whenever the cumulative
    *absolute* heading change (total turning) since the last landmark exceeds
    `angle` (rad). Unlike the net sweep-angle trigger, |dpsi| never cancels on
    alternating-curvature paths."""
    idx = [0]
    cum = 0.0
    last_psi = recon[0][3]
    for i, (_t, _x, _y, psi, _s, _v) in enumerate(recon):
        if i == 0:
            continue
        d = psi - last_psi
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        cum += abs(d)
        last_psi = psi
        if cum >= angle:
            idx.append(i)
            cum = 0.0
    return idx


def segment_error(recon, eps=0.05):
    """Online (causal) polyline-simplifier trigger: emit a landmark when the
    maximum perpendicular distance of the buffered points since the last
    landmark to the current anchor->point segment exceeds `eps` (metres). This
    is the streaming analogue of the batch Douglas--Peucker criterion, so it
    needs no future information."""
    idx = [0]
    ax, ay = recon[0][1], recon[0][2]
    buf = [(ax, ay)]
    for i in range(1, len(recon)):
        x, y = recon[i][1], recon[i][2]
        buf.append((x, y))
        dx, dy = x - ax, y - ay
        den = dx * dx + dy * dy
        worst = 0.0
        for (bx, by) in buf:
            if den < 1e-12:
                d = math.hypot(bx - ax, by - ay)
            else:
                u = ((bx - ax) * dx + (by - ay) * dy) / den
                u = min(max(u, 0.0), 1.0)
                d = math.hypot(bx - (ax + u * dx), by - (ay + u * dy))
            worst = max(worst, d)
        if worst > eps:
            idx.append(i)
            ax, ay = x, y
            buf = [(x, y)]
    return idx


def online_pla(recon, eps=0.05):
    """Online piecewise-linear approximation (sliding window with a bounded
    L-infinity error): when the buffered point of maximum perpendicular
    deviation from the anchor->current segment exceeds `eps`, emit that point
    (not the current sample) and restart there. This is the online analogue of
    the batch Douglas--Peucker vertex choice, with the same bounded-error
    guarantee and no future information."""
    idx = [0]
    anchor = 0
    buf = [0]
    for i in range(1, len(recon)):
        buf.append(i)
        ax, ay = recon[anchor][1], recon[anchor][2]
        bx, by = recon[i][1], recon[i][2]
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        worst = 0.0
        worst_j = anchor
        for j in buf:
            px, py = recon[j][1], recon[j][2]
            if den < 1e-12:
                d = math.hypot(px - ax, py - ay)
            else:
                u = ((px - ax) * dx + (py - ay) * dy) / den
                u = min(max(u, 0.0), 1.0)
                d = math.hypot(px - (ax + u * dx), py - (ay + u * dy))
            if d > worst:
                worst = d
                worst_j = j
        if worst > eps:
            idx.append(worst_j)
            anchor = worst_j
            buf = [worst_j]
    return idx


def _seg_error(recon, a, b):
    """Maximum perpendicular distance of points [a..b] to the segment a->b."""
    ax, ay = recon[a][1], recon[a][2]
    bx, by = recon[b][1], recon[b][2]
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    worst = 0.0
    for j in range(a, b + 1):
        px, py = recon[j][1], recon[j][2]
        if den < 1e-12:
            d = math.hypot(px - ax, py - ay)
        else:
            u = ((px - ax) * dx + (py - ay) * dy) / den
            u = min(max(u, 0.0), 1.0)
            d = math.hypot(px - (ax + u * dx), py - (ay + u * dy))
        worst = max(worst, d)
    return worst


def swab(recon, eps=0.05, window=200):
    """SWAB (sliding window and bottom-up) online piecewise-linear
    approximation: keep a bounded window of samples; when the window's fit
    error exceeds `eps`, run bottom-up segmentation on the window (repeatedly
    merge the adjacent pair with the smallest resulting error until every
    remaining segment has error <= eps) and emit the first resulting segment.
    Bounded L-infinity error, causal (no future samples)."""
    idx = [0]
    pos = 0
    n = len(recon)
    while pos < n - 1:
        end = min(pos + window, n)
        # bottom-up: start with per-sample segments, then greedily merge
        segs = [[i] for i in range(pos, end)]
        while len(segs) > 1:
            best_cost = float("inf")
            best_k = -1
            for k in range(len(segs) - 1):
                a = segs[k][0]
                b = segs[k + 1][-1]
                e = _seg_error(recon, a, b)
                if e < best_cost:
                    best_cost = e
                    best_k = k
            if best_cost > eps:
                break
            segs[best_k] = segs[best_k] + segs[best_k + 1]
            segs.pop(best_k + 1)
        # emit the endpoint of the first segment and advance
        first_end = segs[0][-1]
        if first_end <= pos:
            first_end = min(pos + 1, n - 1)
        idx.append(first_end)
        pos = first_end
    return idx


def opw_tr(recon, eps=0.05):
    """OPW-TR (opening window + time-ratio distance, Meratnia & de By 2004):
    online spatiotemporal compression. Keep growing the segment from the anchor
    while the maximum synchronized (time-ratio) distance of intermediate points
    to the anchor->end segment is <= `eps`; when it exceeds `eps`, emit the
    previous point and restart there. The time-ratio distance is the spatial
    distance at the temporally-synchronized point (the time-linear interpolation
    of the segment), so it is the time-aware analogue of the perpendicular
    distance used by the segment-error trigger."""
    idx = [0]
    anchor = 0
    for e in range(1, len(recon)):
        ts = recon[anchor][0]
        te = recon[e][0]
        xs, ys = recon[anchor][1], recon[anchor][2]
        xe, ye = recon[e][1], recon[e][2]
        dte = te - ts
        worst = 0.0
        for i in range(anchor + 1, e):
            ti = recon[i][0]
            f = (ti - ts) / dte if dte > 1e-12 else 0.0
            xp = xs + f * (xe - xs)
            yp = ys + f * (ye - ys)
            d = math.hypot(recon[i][1] - xp, recon[i][2] - yp)
            worst = max(worst, d)
        if worst > eps:
            idx.append(e - 1)
            anchor = e - 1
    return idx


def _seg_sed(recon, a, b):
    """Maximum synchronized (time-ratio) distance of points [a..b] to the
    segment a->b (the spatiotemporal error used by SED/SQUISH-E)."""
    ts, te = recon[a][0], recon[b][0]
    xs, ys = recon[a][1], recon[a][2]
    xe, ye = recon[b][1], recon[b][2]
    dte = te - ts
    worst = 0.0
    for i in range(a + 1, b):
        f = (recon[i][0] - ts) / dte if dte > 1e-12 else 0.0
        xp = xs + f * (xe - xs)
        yp = ys + f * (ye - ys)
        worst = max(worst, math.hypot(recon[i][1] - xp, recon[i][2] - yp))
    return worst


def swing_slide(recon, eps=0.05):
    """Swing filter (Elmeleegy et al. 2009): online d-dimensional piecewise-
    linear approximation with per-dimension precision guarantees. For each
    dimension (here x(t) and y(t) as functions of time) the filter maintains
    the feasible slope interval of a line anchored at the last landmark that
    stays within +/- `eps` of every buffered sample; when the interval of any
    dimension becomes empty, the previous sample is emitted as a landmark and
    a new segment starts there. This is a *time-synchronized per-axis L-inf*
    bound, distinct from the perpendicular distance of the segment-error
    trigger and from the synchronized Euclidean distance (SED) of OPW-TR."""
    idx = [0]
    anchor = 0
    for e in range(1, len(recon)):
        t0 = recon[anchor][0]
        feasible = True
        for dim in (1, 2):  # x(t) and y(t)
            v0 = recon[anchor][dim]
            m_lo, m_hi = -math.inf, math.inf
            for i in range(anchor + 1, e + 1):
                dt = recon[i][0] - t0
                if dt <= 1e-12:
                    if abs(recon[i][dim] - v0) > eps:
                        m_lo, m_hi = math.inf, -math.inf
                        break
                    continue
                m_lo = max(m_lo, (recon[i][dim] - v0 - eps) / dt)
                m_hi = min(m_hi, (recon[i][dim] - v0 + eps) / dt)
                if m_lo > m_hi:
                    break
            if m_lo > m_hi:
                feasible = False
                break
        if not feasible:
            idx.append(e - 1)
            anchor = e - 1
    return idx


def squish(recon, n_keep):
    """SQUISH (budget-based spatiotemporal simplification, Muckell 2013): keep
    exactly `n_keep` points (the two endpoints are always kept) by iteratively
    removing the interior point whose removal causes the smallest synchronized
    (time-ratio) error, updating its two neighbours after each removal. This is
    the matched-budget (fixed-N) analogue of the threshold-based online PLA
    baselines."""
    import heapq
    n = len(recon)
    if n <= n_keep:
        return list(range(n))
    prev = [i - 1 for i in range(n)]
    nxt = [i + 1 for i in range(n)]
    nxt[n - 1] = -1
    prev[0] = -1
    removed = [False] * n

    def err_of(i):
        a, b = prev[i], nxt[i]
        if a < 0 or b < 0:
            return float("inf")
        return _seg_sed(recon, a, b)

    heap = [(err_of(i), i) for i in range(1, n - 1)]
    heapq.heapify(heap)
    kept = n
    while kept > n_keep and heap:
        e, i = heapq.heappop(heap)
        if removed[i] or e != err_of(i):
            continue
        removed[i] = True
        a, b = prev[i], nxt[i]
        nxt[a] = b
        if b >= 0:
            prev[b] = a
        kept -= 1
        for j in (a, b):
            if 0 < j < n - 1 and not removed[j]:
                heapq.heappush(heap, (err_of(j), j))
    return [i for i in range(n) if not removed[i]]


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
    "swing/slide (0.05 m)": swing_slide,
    "sweep-angle (0.5 rad)": sweep_angle,
    "sweep-abs-angle (0.5 rad)": sweep_abs_angle,
    "segment-error (0.05 m)": segment_error,
    "online-PLA (0.05 m)": online_pla,
    "SWAB (0.05 m)": swab,
    "OPW-TR (0.05 m)": opw_tr,
    "two-rate periodic (0.5/1.0 Hz)": rb.baseline_sampler,
    "dream (R2_th=0.98)": rb.dream_sampler,
}


def evaluate(recon, fn):
    n_total = len(recon)
    idx = fn(recon)
    # Every method must include the trajectory start and endpoint for a fair
    # full-path reconstruction comparison (otherwise a sampler whose first
    # emission is delayed would be penalised for the uncovered head segment).
    idx = sorted(set([0] + idx + [n_total - 1]))
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
                    rng = random.Random(5000 + run)
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
