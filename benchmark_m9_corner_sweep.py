#!/usr/bin/env python3
"""
benchmark_m9_corner_sweep.py

Generalized corner-cutting study (reviewer N3): sweep the corner angle
(45/90/135 deg), the corridor half-width (0.60/0.85/1.10 m), and DREAM's
R^2_th, and add an explicit sagitta (perpendicular-deviation) mitigation that
caps the maximum perpendicular deviation of the return polyline from the
forward path, then re-evaluate the footprint clearance to the inner corner.

Geometry (left turn of angle theta at the vertex (L, 0)): the centreline runs
(0,0) -> (L,0) then turns left by theta and runs L more metres along
(cos theta, sin theta). For a corridor of half-width hw, the concave inner
corner (the point a chord-cutting return polyline approaches) is

    inner_corner = (L - hw * tan(theta/2), +hw).

For theta = 90 deg this reduces to (L - hw, +hw), the value used in
benchmark_m8_corner.py.

Samplers: DREAM at R^2_th in {0.90, 0.95, 0.98}, the sweep-angle trigger,
the segment-error trigger, Douglas--Peucker (offline), the two-rate baseline,
and the DREAM + sagitta-cap mitigation (cap in {0.10, 0.05} m): starting from
DREAM's landmark set, iteratively insert the sample of maximum perpendicular
deviation to the current polyline whenever it exceeds the cap, so the chord
error is bounded and the polyline cannot cut the corner by more than the cap.

Usage:  python3 benchmark_m9_corner_sweep.py
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src", "vslam_recovery_core", "src"))

import run_gazebo_benchmark as rb          # noqa: E402
import benchmark_baselines_gazebo as bb    # noqa: E402

ROBOT_HW, ROBOT_HD = 0.15, 0.10
DT = 0.02
L = 3.0


def l_corner_recon(theta_deg):
    """Left-turn L-corner: straight L, turn theta, straight L. Returns
    (recon, inner_corner) with recon in (t, x, y, psi, s, v) form."""
    theta = math.radians(theta_deg)
    segs = [(0.20, 0.00, L / 0.20),
            (0.20, 0.60, theta / 0.60),
            (0.20, 0.00, L / 0.20)]
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
    return out, theta


def inner_corner(theta, hw):
    return (L - hw * math.tan(theta / 2.0), hw)


def simulate_return(recon, idx, corner, theta=None, hw=None):
    """Closed-loop pure-pursuit return (matches benchmark_m8_corner.py);
    returns the final error and the minimum footprint clearance to the inner
    corner."""
    if len(idx) < 1:
        return float("inf"), (float("inf"), False)
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
        if math.hypot(x, y) < kFinalTolerance:
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
        approach = clamp(math.hypot(x, y) / 0.4, 0.0, 1.0)
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
    return math.hypot(x, y), corner_clearance(track, corner, theta, hw)


def inner_wall_segments(theta, hw):
    """Inner (concave) wall polyline of the left-turn L-corner: two segments
    meeting at the inner corner. Segment 1 is the first arm's inner edge; segment
    2 is the second arm's inner edge. Returns a list of (A, B) line segments."""
    c = inner_corner(theta, hw)
    # first arm inner wall runs along +x at y = hw up to the corner
    a1 = (0.0, hw)
    # second arm inner wall extends from the corner along the turn direction
    a2 = (c[0] + L * math.cos(theta), c[1] + L * math.sin(theta))
    return [(a1, c), (c, a2)]


def _point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    u = 0.0 if den < 1e-12 else ((px - ax) * dx + (py - ay) * dy) / den
    u = min(max(u, 0.0), 1.0)
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def corner_clearance(track, corner, theta=None, hw=None):
    """Minimum distance from the inner wall polyline (the corner point plus the
    two wall segments) to the robot footprint rectangle over the whole return
    track. The distance is computed per footprint pose in the body frame: a
    wall point inside the rectangle gives penetration (signed <= 0), otherwise
    the gap is the positive point-to-rectangle distance. Returns
    (min_separation, contact); contact is True iff any wall point entered the
    rectangle at any pose. If theta/hw are given, the full wall polyline is
    used instead of only the corner point."""
    walls = [corner]
    if theta is not None and hw is not None:
        for (a, b) in inner_wall_segments(theta, hw):
            seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(math.ceil(seg_len / 0.02)))
            for j in range(n + 1):
                f = j / n
                walls.append((a[0] + (b[0] - a[0]) * f,
                              a[1] + (b[1] - a[1]) * f))
    min_sep = float("inf")
    contact = False
    for (x, y, yaw) in track:
        c = math.cos(yaw)
        s = math.sin(yaw)
        for (wx, wy) in walls:
            dx = wx - x
            dy = wy - y
            lx = dx * c + dy * s
            ly = -dx * s + dy * c
            qx = abs(lx) - ROBOT_HW
            qy = abs(ly) - ROBOT_HD
            if qx <= 0.0 and qy <= 0.0:
                min_sep = min(min_sep, 0.0)
                contact = True
            else:
                sep = math.hypot(max(qx, 0.0), max(qy, 0.0))
                min_sep = min(min_sep, sep)
    return min_sep, contact


def sagitta_cap(recon, base_idx, cap):
    """Iteratively insert the sample of maximum perpendicular deviation from
    the current polyline whenever it exceeds `cap`, bounding the chord error."""
    idx = sorted(set([0] + list(base_idx) + [len(recon) - 1]))
    changed = True
    while changed:
        changed = False
        worst, worst_i, worst_k = 0.0, -1, -1
        for k in range(len(idx) - 1):
            a, b = idx[k], idx[k + 1]
            if b - a < 2:
                continue
            for i in range(a + 1, b):
                d = bb.dist_polyline(recon[i][1], recon[i][2],
                                     [(recon[a][1], recon[a][2]),
                                      (recon[b][1], recon[b][2])])
                if d > worst:
                    worst, worst_i, worst_k = d, i, k
        if worst > cap and worst_k >= 0:
            idx.insert(worst_k + 1, worst_i)
            changed = True
    return idx


def main():
    print("=" * 92)
    print("Corner sweep: angle x half-width x R^2_th, + sagitta-cap mitigation")
    print("=" * 92)
    print(f"{'angle':>6} {'hw':>5} {'sampler':<22} {'N':>4} {'final':>8} "
          f"{'corner':>9} {'contact'}")
    for theta_deg in (45, 90, 135):
        recon, theta = l_corner_recon(theta_deg)
        for hw in (0.60, 0.85, 1.10):
            corner = inner_corner(theta, hw)
            for name, idx in [
                ("DREAM 0.98", rb.dream_sampler(recon, r2_threshold=0.98)),
                ("DREAM 0.95", rb.dream_sampler(recon, r2_threshold=0.95)),
                ("DREAM 0.90", rb.dream_sampler(recon, r2_threshold=0.90)),
                ("DREAM 0.98 + sagitta .10",
                 sagitta_cap(recon, rb.dream_sampler(recon, r2_threshold=0.98), 0.10)),
                ("DREAM 0.90 + sagitta .10",
                 sagitta_cap(recon, rb.dream_sampler(recon, r2_threshold=0.90), 0.10)),
                ("sweep-angle 0.5", bb.sweep_angle(recon, angle=0.5)),
                ("segment-error 0.05", bb.segment_error(recon, eps=0.05)),
                ("segment-error 0.10", bb.segment_error(recon, eps=0.10)),
                ("Douglas-Peucker 0.05", bb.douglas_peucker(recon, eps=0.05)),
            ]:
                fe, (clr, contact) = simulate_return(recon, idx, corner, theta, hw)
                print(f"{theta_deg:>5} {hw:>5.2f} {name:<24} {len(idx):>4} "
                      f"{fe:>8.3f} {clr:>9.3f} {'YES' if contact else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
