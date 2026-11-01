#!/usr/bin/env python3
"""
friction_ablation.py

Record odometry from the three benchmark worlds under reduced wheel friction
(mu=0.2, mu2=0.1 instead of the default mu=0.7, mu2=0.5) and compare the
closed-loop recovery error against the default-friction recordings in
bench_data/. The reduced friction makes the wheels slip more, so the recorded
forward path drifts; the point of the ablation is to show that this mild slip
does not degrade the closed-loop return error.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 friction_ablation.py

Note: recording needs a virtual display (the worlds carry a camera sensor,
which forces OGRE rendering), so run it under Xvfb when no physical display is
available:

    xvfb-run -a python3 friction_ablation.py
"""

import csv
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402

LOW_MU = 0.2
LOW_MU2 = 0.1


def make_low_friction(src, dst):
    """Copy a world SDF with the wheel contact friction reduced."""
    with open(src) as fh:
        s = fh.read()
    s = s.replace("<mu>0.7</mu>", f"<mu>{LOW_MU}</mu>")
    s = s.replace("<mu2>0.5</mu2>", f"<mu2>{LOW_MU2}</mu2>")
    with open(dst, "w") as fh:
        fh.write(s)


def load_recon(csv_path):
    raw = []
    with open(csv_path) as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return rb.reconstruct(raw)


def main():
    tmp = tempfile.mkdtemp(prefix="friction_ablation_")
    node, rclpy = rb._ensure_ros()
    rows = []
    print("=" * 78)
    print(f"Friction ablation: default mu=0.7/mu2=0.5 vs "
          f"low mu={LOW_MU}/mu2={LOW_MU2}")
    print("=" * 78)
    try:
        for name, world in rb.WORLDS:
            src = os.path.join(rb._WORLDS_DIR, world)
            dst = os.path.join(tmp, world.replace(".sdf", "_lowfric.sdf"))
            make_low_friction(src, dst)

            print(f"\n=== World {name} (low friction) ===", flush=True)
            odom_low = rb.record_world(dst, f"{name}_lowfric",
                                       rb.PROFILES[name], node)
            if not odom_low:
                print(f"  [{name}] no odometry recorded; skipping")
                continue
            csv_low = os.path.join(rb._DATA_DIR, f"odom_{name}_lowfric.csv")
            with open(csv_low, "w", newline="") as fh:
                csv.writer(fh).writerows(odom_low)
            print(f"  [{name}] saved {csv_low} ({len(odom_low)} samples)")

            recon_def = load_recon(os.path.join(rb._DATA_DIR,
                                                f"odom_{name}.csv"))
            recon_low = rb.reconstruct(odom_low)
            d_def = rb.dream_sampler(recon_def)
            d_low = rb.dream_sampler(recon_low)
            pe_def = rb.recovery_pe_closed_loop(recon_def, d_def)
            pe_low = rb.recovery_pe_closed_loop(recon_low, d_low)

            end_def = (recon_def[-1][1], recon_def[-1][2])
            end_low = (recon_low[-1][1], recon_low[-1][2])
            drift = math.hypot(end_def[0] - end_low[0],
                               end_def[1] - end_low[1])

            print(f"  [{name}] N_dream default={len(d_def)} low={len(d_low)} "
                  f"| PE_closed default={pe_def:.3f} m low={pe_low:.3f} m "
                  f"| forward-end drift={drift:.3f} m", flush=True)
            rows.append((name, len(d_def), len(d_low), pe_def, pe_low, drift))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("\n" + "=" * 78)
    print(f"{'world':>6} | {'N_def':>5} | {'N_low':>5} | {'PE_def':>8} | "
          f"{'PE_low':>8} | {'end_drift':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r[0]:>6} | {r[1]:>5} | {r[2]:>5} | {r[3]:>8.3f} | "
              f"{r[4]:>8.3f} | {r[5]:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
