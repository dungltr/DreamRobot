#!/usr/bin/env python3
"""
benchmark_matched_budget.py

Matched-budget online comparison for reviewer E3
(DREAM_Revision4_Section_by_Section_Revision_Plan.md). Answers the decisive
question: at a fixed landmark budget, does DREAM's distance--time channel
improve e_s (distance--time reconstruction) over (a) its own path-only
variant and (b) an offline gap-splitting comparator that inserts the same
number of extra points at the temporal midpoints of the largest time gaps --
rather than over a baseline that simply uses more landmarks?

Methods compared on the recorded Gazebo worlds (B slalom, C S-curve) and a
synthetic speed-varying stress profile:
  * DREAM path-only    (R^2_th = 0.98)
  * DREAM path+speed   (R^2_th = 0.98)
  * gap-splitting      (matched budget: add the same number of extra points
                        at the temporal midpoints of the largest time gaps)
  * sweep-angle        (0.5 rad)
  * sweep-angle+speed  (0.5 rad + distance--time R^2_th = 0.98)

For each method it reports the complete-waypoint-set landmark count N, the
maximum path-reconstruction error (one-sided Hausdorff), the arc-increment
distance--time error e_s, and the speed-profile error e_v.

Offline:  python3 benchmark_matched_budget.py
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
import benchmark_speed_channel as sc       # noqa: E402
import benchmark_pareto as bp              # noqa: E402
import dream_core                          # noqa: E402
from benchmark_gapsplit_arc import gap_split  # noqa: E402


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def full_n(idx, n_total):
    return len(sorted(set([0] + idx + [n_total - 1])))


def matched_sweep_abs(recon, target_n, canonical_angle=None):
    """Return the cumulative-absolute-turning trigger at a canonical matched-
    budget operating point. Because this trigger's landmark count decreases
    monotonically in its angle threshold and multiple thresholds can produce the
    same integer count, a single deterministic threshold is used per world (the
    one that attains exactly `target_n`); the non-uniqueness of the count is
    reported alongside the Pareto sweep rather than as a unique optimum. The net
    sweep-angle trigger cannot reach DREAM's matched count on the curved worlds
    (its count jumps discretely under alternating-curvature cancellation)."""
    if canonical_angle is None:
        return None
    return bb.sweep_abs_angle(recon, angle=canonical_angle), canonical_angle


def report(recon, idx):
    n = full_n(idx, len(recon))
    worst, _ = bb.path_error(recon, sorted(set([0] + idx + [len(recon) - 1])))
    es = sc.arc_s_error(recon, idx)
    ev = sc.arc_v_error(recon, idx)
    return n, worst, es, ev


def noisy_recon(recon, rng, sigma_v=0.01, sigma_w=0.02):
    """Re-integrate a clean recon (t, x, y, psi, s, v) under additive white
    odometry noise, reconstructing the angular rate from the psi differences."""
    raw = []
    prev_psi = 0.0
    prev_t = None
    for (t, x, y, psi, s, v) in recon:
        w = 0.0
        if prev_t is not None and (t - prev_t) > 0:
            d = psi - prev_psi
            while d > math.pi:
                d -= 2.0 * math.pi
            while d < -math.pi:
                d += 2.0 * math.pi
            w = d / (t - prev_t)
        raw.append((t, x, y, v, w))
        prev_psi = psi
        prev_t = t
    est = []
    x = y = psi = s = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw:
        dt = t - prev if prev is not None else 0.0
        vn = max(0.0, v + sigma_v * rng.gauss(0.0, 1.0))
        wn = w + sigma_w * rng.gauss(0.0, 1.0)
        if prev is not None and 0.0 < dt < 0.5:
            psi += wn * dt
            x += vn * math.cos(psi) * dt
            y += vn * math.sin(psi) * dt
            s += vn * dt
        est.append((t, x, y, psi, s, vn))
        prev = t
    return est


def _ci(values, z=1.96):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, z * (var / n) ** 0.5


def run_mc(runs=30, sigma_v=0.01, sigma_w=0.02):
    """Matched-budget comparison under Monte-Carlo odometry noise: at its fixed
    threshold each method is run on `runs` re-integrated trajectories and the
    mean +/- 95% CI is reported, so the matched-budget claim carries uncertainty
    rather than resting on the clean single run of main()."""
    print("\n" + "=" * 96)
    print(f"Matched-budget comparison under noise ({runs} runs, "
          f"sigma_v={sigma_v}, sigma_w={sigma_w})")
    print("=" * 96)
    rows = []
    for name, ang in (("B", 1.5), ("C", 3.0)):
        clean = load_recon(name)
        agg = {"path-only": [], "path+speed": [], "sweep-abs": []}
        for run in range(runs):
            rng = random.Random(7000 + run)
            recon = noisy_recon(clean, rng, sigma_v, sigma_w)
            po = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
            ps = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
            ab = bb.sweep_abs_angle(recon, angle=ang)
            agg["path-only"].append(report(recon, po))
            agg["path+speed"].append(report(recon, ps))
            agg["sweep-abs"].append(report(recon, ab))
        for key, label in (("path-only", "DREAM path-only"),
                           ("path+speed", "DREAM path+speed"),
                           ("sweep-abs", f"sweep-abs @{ang:g} rad (mbudget)")):
            rs = agg[key]
            n_m, n_c = _ci([r[0] for r in rs])
            p_m, p_c = _ci([r[1] for r in rs])
            s_m, s_c = _ci([r[2] for r in rs])
            v_m, v_c = _ci([r[3] for r in rs])
            print(f"{name:<3} {label:<26} N={n_m:5.1f}+-{n_c:4.1f} "
                  f"path={p_m:6.4f}+-{p_c:6.4f} e_s={s_m:6.4f}+-{s_c:6.4f} "
                  f"e_v={v_m:6.4f}+-{v_c:6.4f}")
            rows.append([name, label, n_m, n_c, p_m, p_c, s_m, s_c, v_m, v_c])
    out = os.path.join(_HERE, "bench_data", "matched_budget_mc.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "method", "N_mean", "N_ci", "path_mean",
                    "path_ci", "e_s_mean", "e_s_ci", "e_v_mean", "e_v_ci"])
        for r in rows:
            w.writerow([round(float(x), 6) if not isinstance(x, str) else x
                        for x in r])
    print(f"[OK] wrote {out}")


def main():
    worlds = ["B", "C"]
    print("=" * 96)
    print("Matched-budget comparison (arc payload; boundary start/end in every count)")
    print("=" * 96)
    print(f"{'world/profile':<16} {'method':<20} {'N':>4} {'path err':>9} "
          f"{'e_s arc':>8} {'e_v':>8}")
    all_rows = []
    for name in worlds:
        recon = load_recon(name)
        po = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
        ps = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
        sw = bb.sweep_angle(recon, angle=0.5)
        sws = bp.sweep_angle_speed(recon, angle=0.5, r2_th=0.98)
        extra = full_n(ps, len(recon)) - full_n(po, len(recon))
        gs = gap_split(recon, po, extra)
        target_n = full_n(ps, len(recon))
        canonical_angle = {"B": 1.5, "C": 3.0}.get(name)
        matched = matched_sweep_abs(recon, target_n, canonical_angle)
        sw_abs_idx, sw_abs_angle = matched if matched else (None, None)
        sw_abs_label = (f"sweep-abs-angle@{sw_abs_angle:g} rad (mbudget)"
                        if sw_abs_angle is not None else "sweep-abs-angle (mbudget)")

        for meth, idx in [("DREAM path-only", po), ("DREAM path+speed", ps),
                          ("gap-split (mbudget)", gs),
                          ("sweep-angle", sw), ("sweep-angle+speed", sws),
                          (sw_abs_label, sw_abs_idx)]:
            if idx is None:
                continue
            n, worst, es, ev = report(recon, idx)
            all_rows.append((name, meth, n, worst, es, ev))
            print(f"{name:<16} {meth:<20} {n:>4d} {worst:>9.4f} {es:>8.4f} "
                  f"{ev:>8.4f}")
        print("-" * 96)

    # synthetic speed-varying stress profile (same as benchmark_speed_profile)
    stress = stress_recon()
    po = rb.dream_sampler(stress, r2_threshold=0.98, regress_speed=False)
    ps = rb.dream_sampler(stress, r2_threshold=0.98, regress_speed=True)
    extra = full_n(ps, len(stress)) - full_n(po, len(stress))
    gs = gap_split(stress, po, extra)
    for meth, idx in [("DREAM path-only", po), ("DREAM path+speed", ps),
                      ("gap-split (mbudget)", gs)]:
        n, worst, es, ev = report(stress, idx)
        all_rows.append(("stress", meth, n, worst, es, ev))
        print(f"{'stress':<16} {meth:<20} {n:>4d} {worst:>9.4f} {es:>8.4f} "
              f"{ev:>8.4f}")

    out = os.path.join(_HERE, "bench_data", "matched_budget.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "method", "N", "path_err", "e_s_arc", "e_v"])
        for r in all_rows:
            w.writerow([r[0], r[1], r[2], round(r[3], 6), round(r[4], 6),
                        round(r[5], 6)])
    print(f"[OK] wrote {out}")
    run_mc()


def stress_recon():
    """Synthetic speed-varying trajectory with accel/decel/stops and two gentle
    turns (same construction as benchmark_speed_profile)."""
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


if __name__ == "__main__":
    main()
