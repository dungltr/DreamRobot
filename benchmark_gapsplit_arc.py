#!/usr/bin/env python3
"""
benchmark_gapsplit_arc.py

Re-derive the gap-splitting table (manuscript Table tab:gapsplit) under the
*recommended arc-increment* distance field d = Δs_k instead of the
backward-compatible chord payload.

For each trajectory (the three recorded worlds plus the synthetic speed-profile
stress test) we:
  1. compute the DREAM path-only base landmark set and the path+speed set;
  2. build the offline gap-splitting comparator: starting from the path-only
     base, repeatedly add a landmark at the midpoint of the largest remaining
     time gap until the total count matches the path+speed set;
  3. report e_s = max |s_reconstructed - s_true| with BOTH decoders:
        - chord payload  (deployed, d = Euclidean chord between landmarks), and
        - arc increment  (recommended, d = Δs_k = accumulated arc-length diff),
     so the effect of the representation is visible at matched budget.

The chord column must reproduce the manuscript's existing tab:gapsplit values;
the arc column is the new result.

Usage:  python3 benchmark_gapsplit_arc.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_speed_profile as sp        # noqa: E402
import benchmark_speed_channel as sc        # noqa: E402


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def with_boundary(idx, n):
    return sorted(set([0] + list(idx) + [n - 1]))


def gap_split(recon, base_idx, n_add):
    """Add `n_add` landmarks at the midpoints of the largest time gaps of the
    (boundary-included) base set. Offline diagnostic: needs the whole path."""
    idx = with_boundary(base_idx, len(recon))
    for _ in range(n_add):
        best_gap, best_k, best_i = -1.0, -1, -1
        for k in range(len(idx) - 1):
            a, b = idx[k], idx[k + 1]
            if b - a < 2:
                continue
            gap = recon[b][0] - recon[a][0]
            if gap > best_gap:
                t_mid = 0.5 * (recon[a][0] + recon[b][0])
                # nearest sample index to the temporal midpoint
                lo, hi = a, b
                while hi - lo > 1:
                    m = (lo + hi) // 2
                    if recon[m][0] < t_mid:
                        lo = m
                    else:
                        hi = m
                mi = lo if abs(recon[lo][0] - t_mid) <= abs(recon[hi][0] - t_mid) else hi
                if mi == a or mi == b:
                    mi = (a + b) // 2
                best_gap, best_k, best_i = gap, k, mi
        if best_k < 0:
            break
        idx.insert(best_k + 1, best_i)
    return idx


def main():
    print("=" * 96)
    print("Gap-splitting re-derived under arc-increment d=Δs_k (tab:gapsplit)")
    print("=" * 96)
    print(f"{'traj':<7} {'method':<16} {'base':>4} {'added':>5} {'total':>5} "
          f"{'e_s chord':>10} {'e_s arc':>10}")

    cases = []
    for name in "ABC":
        cases.append((name, load_recon(name)))
    cases.append(("stress", sp.speed_profile_recon()))

    for name, recon in cases:
        n = len(recon)
        off = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=False)
        on = rb.dream_sampler(recon, r2_threshold=0.98, regress_speed=True)
        base = with_boundary(off, n)
        full = with_boundary(on, n)
        added = len(full) - len(base)
        gs = gap_split(recon, off, added)

        es_chord = rb.chord_payload_s_error(recon, [i for i in off])
        es_arc = sc.arc_s_error(recon, [i for i in off])
        es_chord_on = rb.chord_payload_s_error(recon, [i for i in on])
        es_arc_on = sc.arc_s_error(recon, [i for i in on])
        es_chord_gs = rb.chord_payload_s_error(recon, [i for i in gs])
        es_arc_gs = sc.arc_s_error(recon, [i for i in gs])

        print(f"{name:<7} {'path-only':<16} {len(base):>4} {'0':>5} "
              f"{len(base):>5} {es_chord:>10.4f} {es_arc:>10.4f}")
        print(f"{name:<7} {'path+speed':<16} {len(base):>4} {added:>5} "
              f"{len(full):>5} {es_chord_on:>10.4f} {es_arc_on:>10.4f}")
        print(f"{name:<7} {'gap-splitting':<16} {len(base):>4} {added:>5} "
              f"{len(gs):>5} {es_chord_gs:>10.4f} {es_arc_gs:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
