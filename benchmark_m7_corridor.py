#!/usr/bin/env python3
"""
benchmark_m7_corridor.py

Offline M7 study for the corridor demonstration: (a) compare the two-rate
baseline, the sweep-angle trigger, the sweep-angle+speed hybrid, and DREAM on
the SAME corridor, SAME controller, SAME camera-failure trigger; (b) sweep the
camera-loss fraction (70/80/90/100% of the forward distance); (c) report the
return distance to 5 decimals together with a completion flag (whether the
controller reached its 0.03 m stopping tolerance before the 180 s timeout) and
the footprint clearances.

The forward motion uses the deployed dashboard control methods and the return
uses the deployed 20 Hz pure-pursuit controller (recovery_controller.cpp), so
this is the deterministic kinematic reproduction, not the Gazebo physics run.

Usage:  source /opt/ros/humble/setup.bash && source install/setup.bash
        python3 benchmark_m7_corridor.py
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(_HERE, "src", "vslam_recovery_core", "src")
sys.path.insert(0, _HERE)
sys.path.insert(0, SRC)

import optimal_trajectory as ot          # noqa: E402
import test_auto_recovery_sim as sim     # noqa: E402
import run_gazebo_benchmark as rb        # noqa: E402
import benchmark_baselines_gazebo as bb  # noqa: E402
import benchmark_pareto as bp            # noqa: E402

DREAM_MAX_WINDOW = 5000
DREAM_ARC_CAP = 1.0


def record_forward(mode):
    """Run the deployed forward motion and return the (t,x,y,psi,s,v) recon
    list and the total forward length (m)."""
    node = sim._FakeDashboardNode(mode)
    x = y = yaw = 0.0
    t = 0.0
    dist = 0.0
    last_v = last_w = 0.0
    recon = [(0.0, x, y, yaw, 0.0, 0.0)]
    total = sim.nominal_forward_length(mode)
    while t < 300.0 and dist < total:
        node.cur_x, node.cur_y, node.cur_yaw = x, y, yaw
        res = sim.dashboard_control(node, mode)
        if res is None or res[0] is None:
            break
        v_des, w_des = res
        v = sim.slew(last_v, v_des, sim.dash.MAX_LIN_ACCEL * sim.DT)
        w = sim.slew(last_w, w_des, sim.dash.MAX_ANG_ACCEL * sim.DT)
        w = sim.clamp(w, -sim.dash.MAX_ANGULAR_VELOCITY,
                      sim.dash.MAX_ANGULAR_VELOCITY)
        v = max(0.0, v)
        last_v, last_w = v, w
        x += v * math.cos(yaw) * sim.DT
        y += v * math.sin(yaw) * sim.DT
        yaw = sim.norm(yaw + w * sim.DT)
        dist += v * sim.DT
        t += sim.DT
        recon.append((t, x, y, yaw, dist, v))
    return recon, total


def return_result(recon, idx, frac):
    """Run the deployed 20 Hz closed-loop return from the camera-loss pose
    (at `frac` of the forward length). Returns
    (final_err, completed, min_wall, min_obst, coll)."""
    n = len(recon)
    k = max(2, int(round(frac * n)))
    if k > n - 1:
        k = n - 1
    recon_fwd = recon[:k]

    # re-run the sampler on the truncated forward trajectory (no landmarks
    # after vision loss)
    idx = [i for i in idx if i < k]

    x = recon_fwd[-1][1]
    y = recon_fwd[-1][2]
    yaw = recon_fwd[-1][3]

    # landmarks as (x, y) waypoints
    forward = [(recon[i][1], recon[i][2]) for i in idx]
    if not forward:
        forward = [(recon[0][1], recon[0][2])]

    # pi-turn
    target = sim.norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = sim.norm(target - yaw)
        if abs(err) < 0.03:
            break
        w = sim.clamp(2.0 * err, -sim.RECOVERY_ROTATE_W, sim.RECOVERY_ROTATE_W)
        yaw = sim.norm(yaw + w * sim.DT)
        rt += sim.DT

    return_path = [(x, y)] + list(reversed(forward))
    if math.hypot(return_path[-1][0], return_path[-1][1]) > 1e-3:
        return_path.append((0.0, 0.0))
    samples = sim.build_samples(return_path)

    last_v = last_w = 0.0
    t = 0.0
    min_wall = float("inf")
    min_obst = float("inf")
    coll = None
    completed = False
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < sim.RECOVERY_FINAL_TOL:
            completed = True
            break
        gx, gy = sim.pure_pursuit_goal(x, y, samples)
        alpha = sim.norm(math.atan2(gy - y, gx - x) - yaw)
        v_des = sim.RECOVERY_BASE_SPEED * max(0.0, math.cos(alpha))
        approach = sim.clamp(dstart / sim.RECOVERY_APPROACH, 0.0, 1.0)
        v_des = min(v_des, max(0.02, sim.RECOVERY_BASE_SPEED * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / max(sim.RECOVERY_LOOKAHEAD, 1e-3)
        w_des = sim.clamp(w_des, -sim.RECOVERY_MAX_W, sim.RECOVERY_MAX_W)
        v = sim.slew(last_v, v_des, sim.RECOVERY_LIN_ACCEL * sim.DT)
        w = sim.slew(last_w, w_des, sim.RECOVERY_ANG_ACCEL * sim.DT)
        last_v, last_w = v, w
        v = max(0.0, v)
        w = sim.clamp(w, -sim.RECOVERY_MAX_W, sim.RECOVERY_MAX_W)
        x += v * math.cos(yaw) * sim.DT
        y += v * math.sin(yaw) * sim.DT
        yaw = sim.norm(yaw + w * sim.DT)
        t += sim.DT
        for (cx, cy) in sim.robot_corners(x, y, yaw):
            min_wall = min(min_wall, sim.WALL_Y - abs(cy))
        for (ox, oy) in sim.OBSTACLES:
            min_obst = min(min_obst, sim.dist_point_rect(ox, oy, x, y, yaw)
                           - sim.OBST_R)
        c = sim.collision_kind(x, y, yaw)
        if c:
            coll = c
    return math.hypot(x, y), completed, min_wall, min_obst, coll


SAMPLERS = {
    "two-rate": lambda recon: rb.baseline_sampler(recon),
    "sweep-angle": lambda recon: bb.sweep_angle(recon, angle=0.5),
    "sweep-angle+speed": lambda recon: bp.sweep_angle_speed(
        recon, angle=0.5, r2_th=0.98),
    "DREAM": lambda recon: rb.dream_sampler(
        recon, r2_threshold=0.98, max_window_size=DREAM_MAX_WINDOW,
        max_window_arc=DREAM_ARC_CAP),
}


def main():
    ot.configure(sim.WORLD)
    print("=" * 92)
    print("M7 corridor comparison (deployed 20 Hz controller, kinematic repro)")
    print("=" * 92)
    print(f"{'mode':>4} {'sampler':<16} {'frac':>5} {'N':>3} "
          f"{'final(m)':>10} {'completed':>9} {'wall':>7} {'obst':>7} {'coll'}")
    for mode in (1, 2, 3, 4):
        recon, _total = record_forward(mode)
        for sname, fn in SAMPLERS.items():
            full_idx = fn(recon)
            for frac in (0.7, 0.8, 0.9, 1.0):
                fe, done, mw, mo, cl = return_result(recon, full_idx, frac)
                print(f"{mode:>4} {sname:<16} {frac:>5.2f} "
                      f"{sum(1 for i in full_idx if i < int(frac*len(recon))):>3} "
                      f"{fe:>10.5f} {str(done):>9} {mw:>7.3f} {mo:>7.3f} {cl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
