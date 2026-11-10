#!/usr/bin/env python3
"""
benchmark_m8_corner.py

Corner-cutting benchmark (reviewer R3-M8 / R2-Major-4/10): the windowed
linearity trigger emits a landmark only after a turn has accumulated enough
sweep, so the reconstructed return polyline can cut across the inside of a 90
degree corridor corner. This script quantifies, for each sampler and at matched
landmark budgets, the minimum footprint clearance of the closed-loop return
path to the inner corner of an L-shaped corridor.

Geometry: a 1.7 m wide L-corridor whose centreline runs (0,0) -> (3,0) -> (3,3)
with a 90 deg left turn at (3,0). The concave inner corner is at (2.15, 0.85).
A return polyline that chords the corner approaches this point; a clearance
below the robot half-width (0.10 m) is counted as a corner contact.

Samplers: DREAM (R^2_th 0.98 and 0.90), sweep-angle (0.5 rad), segment-error
(0.05 m), Douglas--Peucker (0.05 m, offline), and the two-rate baseline.

Usage:  python3 benchmark_m8_corner.py
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_baselines_gazebo as bb    # noqa: E402

ROBOT_HW, ROBOT_HD = 0.15, 0.10
INNER_CORNER = (2.15, 0.85)   # concave corner of the L
DT = 0.02


def l_corner_recon():
    """Integrate the L-centreline (straight 3 m, 90 deg left turn, straight 3 m)
    at v=0.2 m/s into a recon list (t, x, y, psi, s, v)."""
    # straight 3 m, then a 90 deg turn (w = v/R with a tight radius via a fast
    # constant-rate turn), then straight 3 m.
    segs = [(0.20, 0.00, 3.0 / 0.20),     # straight +x, 3 m (15 s)
            (0.20, 0.60, (math.pi / 2) / 0.60),  # 90 deg turn at 0.6 rad/s
            (0.20, 0.00, 3.0 / 0.20)]     # straight +y, 3 m
    out = []
    t = 0.0
    x = y = psi = s = 0.0
    for (v, w, dur) in segs:
        n = max(1, int(round(dur / DT)))
        for _ in range(n):
            x += v * math.cos(psi) * DT
            y += v * math.sin(psi) * DT
            psi += w * DT
            s += v * DT
            t += DT
            out.append((t, x, y, psi, s, v))
    return out


def dist_to_corner(x, y):
    return math.hypot(x - INNER_CORNER[0], y - INNER_CORNER[1])


def simulate_return(recon, idx):
    """Closed-loop pure-pursuit return (same controller as the manuscript);
    returns (track, final_err)."""
    if len(idx) < 1:
        return None, float("inf")
    forward = [(recon[i][1], recon[i][2]) for i in idx]
    x = recon[-1][1]
    y = recon[-1][2]
    yaw = recon[-1][3]
    dt = 1.0 / 20.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def slew(cur, tgt, step):
        d = tgt - cur
        return tgt if abs(d) <= step else cur + (step if d > 0 else -step)

    def norm(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    target = norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = norm(target - yaw)
        if abs(err) < 0.03:
            break
        yaw = norm(yaw + clamp(2.0 * err, -0.8, 0.8) * dt)
        rt += dt

    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = []
    for i in range(len(path) - 1):
        dx, dy = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        seg = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg / 0.05)))
        for j in range(n):
            f = j / n
            samples.append((path[i][0] + dx * f, path[i][1] + dy * f))
    samples.append(path[-1])

    track = [(x, y, yaw)]
    last_v = last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < kFinalTolerance:
            break
        best = 0
        best_d = float("inf")
        for i, (px, py) in enumerate(samples):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_d = d
                best = i
        acc = 0.0
        gx, gy = samples[-1]
        for i in range(best, len(samples) - 1):
            acc += math.hypot(samples[i + 1][0] - samples[i][0],
                              samples[i + 1][1] - samples[i][1])
            if acc >= kLookahead:
                gx, gy = samples[i + 1]
                break
        alpha = norm(math.atan2(gy - y, gx - x) - yaw)
        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = clamp(w_des, -0.25, 0.25)
        v = slew(last_v, v_des, 0.20 * dt)
        w = slew(last_w, w_des, 0.50 * dt)
        last_v, last_w = v, w
        v = max(0.0, v)
        w = clamp(w, -0.25, 0.25)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = norm(yaw + w * dt)
        t += dt
        track.append((x, y, yaw))
    return track, math.hypot(x, y)


def corner_clearance(track):
    """Minimum distance from the robot footprint to the inner corner point."""
    best = float("inf")
    for (x, y, yaw) in track:
        c = math.cos(yaw)
        s = math.sin(yaw)
        for sx in (-ROBOT_HW, ROBOT_HW):
            for sy in (-ROBOT_HD, ROBOT_HD):
                cx = x + sx * c - sy * s
                cy = y + sx * s + sy * c
                best = min(best, math.hypot(cx - INNER_CORNER[0],
                                            cy - INNER_CORNER[1]))
    return best


def main():
    recon = l_corner_recon()
    n = len(recon)
    samplers = [
        ("DREAM 0.98", rb.dream_sampler(recon, r2_threshold=0.98)),
        ("DREAM 0.90", rb.dream_sampler(recon, r2_threshold=0.90)),
        ("sweep-angle 0.5", bb.sweep_angle(recon, angle=0.5)),
        ("segment-error 0.05", bb.segment_error(recon, eps=0.05)),
        ("Douglas-Peucker 0.05", bb.douglas_peucker(recon, eps=0.05)),
        ("two-rate", rb.baseline_sampler(recon)),
    ]
    print("=" * 84)
    print("L-corridor 90 deg corner: footprint clearance to the inner corner")
    print("=" * 84)
    print(f"{'sampler':<22} {'N':>4} {'final(m)':>9} {'corner(m)':>10} {'contact'}")
    for name, idx in samplers:
        track, pe = simulate_return(recon, idx)
        if track is None:
            print(f"{name:<22} {len(idx):>4}  {'inf':>9}")
            continue
        clr = corner_clearance(track)
        contact = "YES" if clr < ROBOT_HD else "no"
        print(f"{name:<22} {len(idx):>4} {pe:>9.3f} {clr:>10.3f} {contact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
