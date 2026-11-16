#!/usr/bin/env python3
"""
benchmark_e4_runtime.py

Reviewer E4: comparative CPU / memory contract for the online samplers, measured
on the same host, same Python runtime, and the same input (recorded Gazebo
odometry of World C, 1695 samples, reconstructed to the (t,x,y,psi,s,v) layout
used throughout the paper). Each method is timed as a whole over the input with
a warm-up pass and repeated timing; the reported value is the minimum wall-clock
over the repeats (to reduce scheduler noise), converted to a mean per-sample
cost. Peak incremental allocation is measured separately with tracemalloc.

Usage:
  python3 benchmark_e4_runtime.py [--world C] [--repeats 5]
"""

import argparse
import csv
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "src", "vslam_recovery_core", "src"))
import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_baselines_gazebo as bb    # noqa: E402


def load_recon(name):
    return rb.reconstruct(bb.load_raw(name))


def _dream(recon):
    return rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)


def _squish(recon, n_keep):
    return bb.squish(recon, n_keep)


def timed_min(fn, recon, repeats):
    """Return (best wall-clock seconds, n_samples, n_landmarks)."""
    idx = fn(recon)                       # warm-up + get landmark count
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(recon)
        best = min(best, time.perf_counter() - t0)
    return best, len(recon), len(idx)


def peak_alloc(fn, recon):
    tracemalloc.start()
    fn(recon)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="C")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    recon = load_recon(args.world)
    n = len(recon)
    dream_count = len(_dream(recon))

    methods = {
        "DREAM (tau=0.98)": _dream,
        "two-rate (0.5/1.0 Hz)": rb.baseline_sampler,
        "sweep-angle (0.5 rad)": lambda r: bb.sweep_angle(r),
        "sweep-abs-angle (0.5 rad)": lambda r: bb.sweep_abs_angle(r),
        "segment-error (0.05 m)": lambda r: bb.segment_error(r),
        "online-PLA (0.05 m)": lambda r: bb.online_pla(r),
        "SWAB (0.05 m)": lambda r: bb.swab(r),
        "OPW-TR (0.05 m)": lambda r: bb.opw_tr(r),
        "swing/slide (0.05 m)": lambda r: bb.swing_slide(r),
        f"SQUISH-E (budget={dream_count})": lambda r: _squish(r, dream_count),
    }

    print(f"Comparative runtime/memory contract on World {args.world} "
          f"({n} samples, {args.repeats} timing repeats, min over repeats):")
    print(f"  {'method':<26} | {'N':>5} | {'total (s)':>10} | "
          f"{'us/sample':>10} | {'peak alloc':>12}")
    rows = []
    for name, fn in methods.items():
        total, _ns, count = timed_min(fn, recon, args.repeats)
        us = total / n * 1e6
        peak = peak_alloc(fn, recon)
        rows.append((name, count, round(total, 6), round(us, 3), peak))
        print(f"  {name:<26} | {count:>5} | {total:>10.5f} | "
              f"{us:>10.3f} | {peak:>10.0f} B")

    out = os.path.join("bench_data", "e4_runtime.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "N_landmarks", "total_s", "us_per_sample",
                    "peak_alloc_B"])
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
