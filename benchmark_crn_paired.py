#!/usr/bin/env python3
"""
benchmark_crn_paired.py

Common-random-number (CRN) paired two-state comparison of the baseline and
DREAM samplers. Unlike the headline Monte-Carlo study (which draws the
return-phase noise from a per-sampler Random(9000+r) stream and therefore gives
the two samplers *different* realized noise sequences once their landmark sets
diverge), here a single pre-generated noise array per run is indexed by the
return-phase step counter, so at step k both samplers see exactly the same
(v, w) noise. This makes the paired per-run difference a contrast under
identical realized per-step noise, answering the reviewer's reproducibility
concern that a shared seed base does not imply common random numbers.

Reported: mean +/- 95% CI of PE_true for each sampler, and the paired
per-run difference (DREAM - baseline) with its 95% CI, median and 90th
percentile.

Usage:  python3 benchmark_crn_paired.py [--runs 50]
"""

import argparse
import csv
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb  # noqa: E402


def _load(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def _crn_two_state(true_recon, est_recon, landmarks_idx, sigma_v, sigma_w,
                   noise_v, noise_w):
    """Two-state reverse replay with common random numbers: noise is drawn from
    the pre-generated arrays `noise_v`/`noise_w` indexed by a step counter, so
    two calls with the same arrays consume identical noise at each step."""
    if len(landmarks_idx) < 1:
        return float("inf"), float("inf"), False, True

    forward = [(est_recon[i][1], est_recon[i][2]) for i in landmarks_idx]

    x = true_recon[-1][1]
    y = true_recon[-1][2]
    yaw = true_recon[-1][3]
    x_est = est_recon[-1][1]
    y_est = est_recon[-1][2]
    yaw_est = est_recon[-1][3]

    dt = 1.0 / 50.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    k = 0  # common noise step counter

    target = rb._norm_angle(yaw_est + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = rb._norm_angle(target - yaw_est)
        if abs(err) < 0.03:
            break
        w = rb._clamp(2.0 * err, -0.8, 0.8)
        yaw = rb._norm_angle(yaw + w * dt)
        wn = w + sigma_w * noise_w[k]
        yaw_est = rb._norm_angle(yaw_est + wn * dt)
        k += 1
        rt += dt

    path = [(x_est, y_est)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = rb._polyline_samples(path)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart_est = math.hypot(x_est, y_est)
        if dstart_est < kFinalTolerance:
            break
        gx, gy = rb._lookahead_goal(x_est, y_est, samples, kLookahead)
        alpha = rb._norm_angle(math.atan2(gy - y_est, gx - x_est) - yaw_est)

        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = rb._clamp(dstart_est / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))

        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = rb._clamp(w_des, -0.25, 0.25)

        v = rb._slew(last_v, v_des, 0.20 * dt)
        w = rb._slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = rb._clamp(w, -0.25, 0.25)

        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = rb._norm_angle(yaw + w * dt)

        vn = v + sigma_v * noise_v[k]
        wn = w + sigma_w * noise_w[k]
        x_est += vn * math.cos(yaw_est) * dt
        y_est += vn * math.sin(yaw_est) * dt
        yaw_est = rb._norm_angle(yaw_est + wn * dt)
        k += 1
        t += dt

    return math.hypot(x, y), math.hypot(x_est, y_est), \
        math.hypot(x_est, y_est) < 0.03, math.hypot(x_est, y_est) >= 0.03


def _percentile(a, p):
    s = sorted(a)
    k = (len(s) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()
    runs = args.runs
    sigma_v, sigma_w = 0.01, 0.02
    max_steps = int(180.0 * 50.0) + 1000  # pi-turn + translation margin

    print("=" * 96)
    print(f"CRN paired two-state comparison (runs={runs}, "
          f"sigma_v={sigma_v}, sigma_w={sigma_w})")
    print("=" * 96)
    rows = []
    for name in "ABC":
        true_recon = _load(name)
        pe_b, pe_d, diff = [], [], []
        for run in range(runs):
            # forward-phase estimate is shared (both samplers consume it).
            fwd_rng = random.Random(5000 + run)
            est = rb.reconstruct_noisy(
                [tuple(r) for r in _raw(name)], sigma_v, sigma_w, fwd_rng)
            b = rb.baseline_sampler(est)
            d = rb.dream_sampler(est)
            # common return-phase noise arrays for both samplers.
            nrng = random.Random(9000 + run)
            noise_v = [nrng.gauss(0.0, 1.0) for _ in range(max_steps)]
            noise_w = [nrng.gauss(0.0, 1.0) for _ in range(max_steps)]
            pb = _crn_two_state(true_recon, est, b, sigma_v, sigma_w,
                                noise_v, noise_w)
            pd = _crn_two_state(true_recon, est, d, sigma_v, sigma_w,
                                noise_v, noise_w)
            pe_b.append(pb[0])
            pe_d.append(pd[0])
            diff.append(pd[0] - pb[0])

        mb, cb = rb._mean_ci(pe_b)
        md, cd = rb._mean_ci(pe_d)
        dm, dc = rb._mean_ci(diff)
        print(f"[{name}] PE_true: base={mb:.4f}+-{cb:.4f}  "
              f"dream={md:.4f}+-{cd:.4f} | paired diff={dm:+.4f}+-{dc:.4f} "
              f"(med {_percentile(diff, 0.5):+.4f}, p90 "
              f"{_percentile(diff, 0.90):+.4f}, max |d| "
              f"{max(abs(x) for x in diff):.4f})")
        rows.append([name, mb, cb, md, cd, dm, dc,
                     _percentile(diff, 0.5), _percentile(diff, 0.90)])

    out = os.path.join(_HERE, "bench_data", "crn_paired.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["world", "base_mean", "base_ci", "dream_mean", "dream_ci",
                    "diff_mean", "diff_ci", "diff_median", "diff_p90"])
        for r in rows:
            w.writerow([r[0]] + [round(float(x), 6) for x in r[1:]])
    print(f"[OK] wrote {out}")
    return 0


def _raw(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return raw


if __name__ == "__main__":
    sys.exit(main())
