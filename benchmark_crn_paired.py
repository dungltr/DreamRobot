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
    """CRN two-state reverse replay: delegates to the single canonical engine
    (benchmark_m5_noise.two_state_return) with pre-generated measurement-noise
    arrays, so two calls with the same arrays consume identical per-step noise.
    A dummy RNG is passed for the (zero) process-noise terms."""
    import benchmark_m5_noise as m5  # local import avoids circular dependency
    dummy = random.Random(12345)
    pe_true, pe_odom, _clearance, completed, timed_out = \
        m5.two_state_return(true_recon, est_recon, landmarks_idx,
                            sigma_v=sigma_v, sigma_w=sigma_w, rng=dummy,
                            gyro_bias=0.0, wheel_scale=0.0, gyro_scale=0.0,
                            process_v=0.0, process_w=0.0,
                            timeout=180.0, turn_timeout=12.0,
                            noise_v=noise_v, noise_w=noise_w)
    return pe_true, pe_odom, completed, timed_out
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
