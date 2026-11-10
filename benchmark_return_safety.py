#!/usr/bin/env python3
"""
benchmark_return_safety.py

Independent safety/quality metrics for the closed-loop reverse return on the
three obstacle-free benchmark worlds (A/B/C), mirroring the wall-margin metric
reported for the corridor demonstration in the manuscript.

For each world and each sampler (baseline Shannon-Nyquist and DREAM), this
script simulates the full closed-loop pure-pursuit return (the same controller
as recovery_pe_closed_loop in run_gazebo_benchmark.py) and reports:

  * PE           : final distance to the origin (the stopping tolerance);
  * min wall     : minimum clearance to the corridor walls (inner face y=+-1.10);
  * max lateral  : maximum cross-track deviation of the return path from the
                   recorded forward path (independent of the stopping rule).

A negative wall clearance means a collision. The worlds are obstacle-free (the
task is the blind return after camera loss, not obstacle avoidance):

  A  straight hallway, walls y=+-1.15, no obstacles.
  B  15 m hallway, no obstacles, driven in a tight symmetric slalom.
  C  15 m hallway, no obstacles, driven in a gentle symmetric S-curve.

Usage:  python3 benchmark_return_safety.py
        python3 benchmark_return_safety.py --sweep-controller
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import run_gazebo_benchmark as rb  # noqa: E402

WALL_HALF = 1.10  # inner face of the corridor wall (box centre 1.15, half-thickness 0.05)
# The benchmark worlds are obstacle-free (the task is the blind return, not
# obstacle avoidance); the dict is kept so the wall-only metrics below work
# unchanged.
OBSTACLES = {
    "A": [],
    "B": [],
    "C": [],
}
OBST_R = 0.2

# Robot footprint (mirrors test_auto_recovery_sim.py so the corridor demo and
# the Worlds A/B/C safety metrics use the *same* rectangle model).
ROBOT_HW = 0.15   # half length along the chassis x axis (m)
ROBOT_HD = 0.10   # half width along the chassis y axis (m)


def robot_corners(x, y, yaw):
    """Return the four corners of the robot rectangle at pose (x, y, yaw)."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    pts = []
    for sx in (-ROBOT_HW, ROBOT_HW):
        for sy in (-ROBOT_HD, ROBOT_HD):
            pts.append((x + sx * c - sy * s, y + sx * s + sy * c))
    return pts


def dist_point_rect(px, py, cx, cy, yaw):
    """Distance from point (px, py) to the robot rectangle (0 = inside/on edge).

    Mirrors test_auto_recovery_sim.py: the closest point on the rotated
    rectangle is found by transforming the query point into the robot frame.
    """
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    dx = px - cx
    dy = py - cy
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    qx = abs(lx) - ROBOT_HW
    qy = abs(ly) - ROBOT_HD
    if qx <= 0.0 and qy <= 0.0:
        return 0.0
    if qx <= 0.0:
        return qy
    if qy <= 0.0:
        return qx
    return math.hypot(qx, qy)


def load_recon(name):
    raw = []
    with open(os.path.join("bench_data", f"odom_{name}.csv")) as fh:
        for r in csv.reader(fh):
            if r:
                raw.append(tuple(float(v) for v in r))
    return rb.reconstruct(raw)


def simulate_return(recon, idx, lookahead=0.45, speed=0.12):
    """Simulate the closed-loop pure-pursuit return and record the (x, y, yaw)
    path.

    Mirrors recovery_pe_closed_loop() but returns the full trajectory (with
    heading, so footprint-aware clearance can be measured) plus the return
    duration, which is independent of the stopping tolerance.

    Returns (track, pe, t_return); track is None and pe is inf when the path
    is not reconstructable.
    """
    # A single landmark (the start) still yields a camera-loss-pose -> origin
    # return polyline; only an empty landmark set is unevaluable.
    if len(idx) < 1:
        return None, float("inf"), float("inf")

    forward = [(recon[i][1], recon[i][2]) for i in idx]
    x = recon[-1][1]
    y = recon[-1][2]
    yaw = recon[-1][3]
    dt = 1.0 / 50.0
    kFinalTolerance = 0.03
    track = [(x, y, yaw)]

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def slew(cur, tgt, step):
        d = tgt - cur
        return tgt if abs(d) <= step else cur + (step if d > 0 else -step)

    def norm(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    # pi-turn
    target = norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = norm(target - yaw)
        if abs(err) < 0.03:
            break
        w = clamp(2.0 * err, -0.8, 0.8)
        yaw = norm(yaw + w * dt)
        rt += dt

    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = rb._polyline_samples(path, spacing=0.05)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < kFinalTolerance:
            break
        gx, gy = rb._lookahead_goal(x, y, samples, lookahead)
        alpha = norm(math.atan2(gy - y, gx - x) - yaw)
        v_des = speed * max(0.0, math.cos(alpha))
        approach = clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, speed * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / max(lookahead, 1e-3)
        w_des = clamp(w_des, -0.25, 0.25)
        v = slew(last_v, v_des, 0.20 * dt)
        w = slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = clamp(w, -0.25, 0.25)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = norm(yaw + w * dt)
        t += dt
        track.append((x, y, yaw))
    return track, math.hypot(x, y), (rt + t)


def dist_seg(x, y, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    u = 0.0 if den < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / den
    u = min(max(u, 0.0), 1.0)
    return math.hypot(x - (ax + u * dx), y - (ay + u * dy))


def metrics(track, obstacles):
    """Footprint-aware clearance along the return trajectory.

    Wall clearance is the minimum distance from the corridor inner face to the
    robot body; obstacle clearance is the minimum distance from the obstacle
    surface (radius already subtracted) to the robot body. Both use the same
    rectangle footprint as the corridor demo (test_auto_recovery_sim.py).
    """
    wall = float("inf")
    obs = float("inf")
    for (px, py, yaw) in track:
        for (cx, cy) in robot_corners(px, py, yaw):
            wall = min(wall, WALL_HALF - abs(cy))
        for (ox, oy) in obstacles:
            obs = min(obs, dist_point_rect(ox, oy, px, py, yaw) - OBST_R)
    return wall, obs


def max_lateral(track, recon):
    fwd = [(r[1], r[2]) for r in recon]
    worst = 0.0
    for (px, py, _yaw) in track:
        best = float("inf")
        for i in range(len(fwd) - 1):
            best = min(best, dist_seg(px, py, fwd[i][0], fwd[i][1],
                                      fwd[i + 1][0], fwd[i + 1][1]))
        worst = max(worst, best)
    return worst


def main():
    do_sweep = "--sweep-controller" in sys.argv
    print("=" * 84)
    print("Closed-loop return safety metrics (independent of the stopping rule)")
    print("=" * 84)
    for name in "ABC":
        recon = load_recon(name)
        obstacles = OBSTACLES[name]
        b_idx = rb.baseline_sampler(recon)
        d_idx = rb.dream_sampler(recon, r2_threshold=0.98)
        print(f"\n--- World {name} ---")
        print(f"{'sampler':<10} {'N':>4} {'PE(m)':>7} {'wall(m)':>9} "
              f"{'obstacle(m)':>12} {'lateral(m)':>11} {'time(s)':>8}")
        for label, idx in (("baseline", b_idx), ("DREAM", d_idx)):
            track, pe, t_ret = simulate_return(recon, idx)
            if track is None:
                print(f"{label:<10} {len(idx):>4} {'inf':>7}")
                continue
            wall, obs = metrics(track, obstacles)
            lat = max_lateral(track, recon)
            print(f"{label:<10} {len(idx):>4} {pe:>7.3f} {wall:>9.3f} "
                  f"{obs:>12.3f} {lat:>11.3f} {t_ret:>8.1f}")

    if do_sweep:
        print("\n" + "=" * 84)
        print("Controller sensitivity (DREAM landmarks): PE / min obstacle per world")
        print("=" * 84)
        for name in "ABC":
            recon = load_recon(name)
            d_idx = rb.dream_sampler(recon, r2_threshold=0.98)
            obstacles = OBSTACLES[name]
            print(f"\n--- World {name} ---")
            print(f"{'lookahead':>10} {'speed':>7} {'PE(m)':>8} {'wall(m)':>9} "
                  f"{'obstacle(m)':>12}")
            for lookahead in (0.30, 0.45, 0.60):
                for speed in (0.10, 0.12, 0.15):
                    track, pe, _t_ret = simulate_return(recon, d_idx, lookahead, speed)
                    wall, obs = metrics(track, obstacles)
                    print(f"{lookahead:>10.2f} {speed:>7.2f} {pe:>8.3f} {wall:>9.3f} "
                          f"{obs:>12.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
