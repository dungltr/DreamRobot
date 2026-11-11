#!/usr/bin/env python3
"""
benchmark_m11_vslam_frame.py

vSLAM-pose landmark-frame variant (reviewer N7/M9). The nominal two-state
study stores waypoints in the *odometry* frame: during forward motion the
dead-reckoned estimate drifts away from the true pose, and at camera loss that
forward-phase drift is retained, so the stored waypoints and the controller's
feedback share the same drifted frame.

Before camera loss, however, vSLAM is still operating and its map-frame pose
is drift-corrected. Storing the waypoints in the *vSLAM map frame* and freezing
the map->odom transform at camera loss removes the forward-phase drift: the
waypoints are then the (near-)true positions, and the odometry estimate at
camera loss is re-anchored to that corrected pose.

This script compares the two frames on the three benchmark worlds under the
same noise model (50 runs, sigma_v=0.01, sigma_w=0.02, seed Random(5000+r)
forward / Random(9000+r) return):

  * odometry frame  : waypoints and estimate from the drifted forward estimate
                      (forward drift retained) -- the nominal study;
  * vSLAM map frame : waypoints and estimate from the true (map) pose during
                      forward motion (forward drift removed), with the return
                      phase still integrating noisy odometry.

The vSLAM variant is implemented by running the sampler on the *true* forward
trajectory and starting the return estimate at the true camera-loss pose; the
return-phase noise is unchanged. This isolates the effect of removing the
forward-phase drift on the true endpoint error.

Usage:  python3 benchmark_m11_vslam_frame.py
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
import benchmark_m5_noise as m5            # noqa: E402

DATA = os.path.join(_HERE, "bench_data")
RUNS = 50
SIGMA_V, SIGMA_W = 0.01, 0.02


def load_raw(name):
    raw = []
    with open(os.path.join(DATA, f"odom_{name}.csv")) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return raw


def mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, z * math.sqrt(var / len(values))


def main():
    print("=" * 96)
    print(f"vSLAM-map vs odometry landmark frame ({RUNS} runs, "
          f"sigma_v={SIGMA_V}, sigma_w={SIGMA_W})")
    print("=" * 96)
    print(f"{'world':<7} {'frame':<12} {'N':>4} {'mean PE_true':>13} "
          f"{'succ(<0.06)':>12} {'mean drift@loss':>16}")
    for name in "ABC":
        raw = load_raw(name)
        true_full = rb.reconstruct(raw)
        for frame in ("odometry", "vSLAM-map"):
            pes = []
            drifts = []
            n_dream = []
            for run_i in range(RUNS):
                rng = random.Random(5000 + run_i)
                est_full = m5.forward_estimate(raw, SIGMA_V, SIGMA_W, rng)
                if frame == "odometry":
                    d = rb.dream_sampler(est_full, r2_threshold=0.98)
                    # return estimate starts at the drifted estimate pose
                    r1 = random.Random(9000 + run_i)
                    pe, _po, _c, _done, _to = m5.two_state_return(
                        true_full, est_full, d, SIGMA_V, SIGMA_W, r1)
                else:
                    # vSLAM map frame: sampler sees the corrected (true) pose;
                    # the return estimate is re-anchored to the true pose at
                    # camera loss, then still integrates noisy odometry.
                    d = rb.dream_sampler(true_full, r2_threshold=0.98)
                    # re-anchored estimate == true pose during forward:
                    # (t, x, y, psi, s, v) from the true trajectory
                    est_anchored = [(t, x, y, psi, s, v)
                                    for (t, x, y, psi, s, v) in true_full]
                    r1 = random.Random(9000 + run_i)
                    pe, _po, _c, _done, _to = m5.two_state_return(
                        true_full, est_anchored, d, SIGMA_V, SIGMA_W, r1)
                pes.append(pe)
                drifts.append(math.hypot(est_full[-1][1] - true_full[-1][1],
                                        est_full[-1][2] - true_full[-1][2]))
                n_dream.append(len(d))
            m, c = mean_ci(pes)
            drift_m, _ = mean_ci(drifts)
            succ = sum(1 for x in pes if x < 0.06) / RUNS
            print(f"{name:<7} {frame:<12} {sum(n_dream)/len(n_dream):>4.1f} "
                  f"{m:>8.4f}\u00b1{c:.4f} {succ*100:>10.0f}% "
                  f"{drift_m:>12.4f}\u00b1{0:>5.4f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
