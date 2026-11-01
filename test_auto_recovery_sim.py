#!/usr/bin/env python3
"""
test_auto_recovery_sim.py

Deterministic offline kinematic reproduction of the recovery-corridor demo.

It runs the four dashboard auto modes (1..4) using the *real* controller
functions imported from `recovery_dashboard_gui`, simulates a camera failure
at 90% of the forward distance, records landmarks with the DREAM sampler and
then executes the *new* closed-loop pure-pursuit recovery implemented in
`recovery_controller.cpp` (pi turn + odometry-feedback path following).

Pass criteria for every mode:
  * the robot returns to the origin: final Euclidean error < 0.05 m;
  * no collision with the corridor walls or the two obstacles;
  * a minimum wall margin >= 0.05 m and obstacle clearance >= 0.01 m.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 test_auto_recovery_sim.py
"""

import math
import os
import sys

WS = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(WS, "src", "vslam_recovery_core", "src")
sys.path.insert(0, SRC)

import optimal_trajectory as ot          # noqa: E402
import dream_core                        # noqa: E402
import recovery_dashboard_gui as dash    # noqa: E402

# ---------------------------------------------------------------------------
# Geometry (recovery_corridor.sdf)
# ---------------------------------------------------------------------------
WORLD = "recovery_corridor.sdf"
OBSTACLES = [(1.5, 0.3), (3.0, -0.3)]
OBST_R = 0.20
WALL_Y = 0.85
ROBOT_HW = 0.15   # half length along the chassis x axis (m)
ROBOT_HD = 0.10   # half width along the chassis y axis (m)

# Recovery controller constants (must mirror recovery_controller.cpp).
DT = 1.0 / 20.0
RECOVERY_SAMPLE_SPACING = 0.05
RECOVERY_LOOKAHEAD = 0.45
RECOVERY_BASE_SPEED = 0.12
RECOVERY_FINAL_TOL = 0.03
RECOVERY_APPROACH = 0.4
RECOVERY_MAX_W = 0.25
RECOVERY_ROTATE_W = 0.8
RECOVERY_LIN_ACCEL = 0.20
RECOVERY_ANG_ACCEL = 0.50

# DREAM parameters used by run_gazebo_demo.sh after the fix.
DREAM_R2_TH = 0.98
DREAM_MIN_WINDOW = 5
DREAM_MAX_WINDOW = 5000
DREAM_MIN_EVENT_SPACING = 0.2
DREAM_MAX_WINDOW_ARC = 1.0


def norm(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def slew(current, target, max_step):
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0.0 else -max_step)


class _NullLogger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class _FakeDashboardNode:
    """Minimal object exposing the attributes used by the real dashboard
    control methods (which are called as unbound functions)."""

    def __init__(self, mode):
        self.auto_mode = mode
        self.auto_waypoints = ot.waypoints(WORLD)
        self.auto_wp_idx = 0
        self.auto_phase = "turn"
        self.auto_active = True
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0

    def get_logger(self):
        return _NullLogger()

    def stop_auto_run(self):
        self.auto_active = False


def robot_corners(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    pts = []
    for sx in (-ROBOT_HW, ROBOT_HW):
        for sy in (-ROBOT_HD, ROBOT_HD):
            pts.append((x + sx * c - sy * s, y + sx * s + sy * c))
    return pts


def dist_point_rect(px, py, cx, cy, yaw):
    """Distance from (px, py) to the robot rectangle (0 = inside/on edge)."""
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


def collision_kind(x, y, yaw):
    for (cx, cy) in robot_corners(x, y, yaw):
        if not (-WALL_Y <= cy <= WALL_Y):
            return "wall"
    for (ox, oy) in OBSTACLES:
        if dist_point_rect(ox, oy, x, y, yaw) < OBST_R:
            return "obstacle"
    return None


def nominal_forward_length(mode):
    if mode in (1, 2):
        wps = ot.waypoints(WORLD)
        return sum(
            math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
            for i in range(len(wps) - 1))
    return ot._SAMPLED[2][-1]


def dashboard_control(node, mode):
    if mode in (1, 2):
        return dash.DashboardNode._control_waypoint(node)
    return dash.DashboardNode._control_trajectory(node)


def pure_pursuit_goal(x, y, samples):
    """Return the lookahead goal point along the dense return polyline."""
    best = 0
    best_d = float("inf")
    for i, (px, py) in enumerate(samples):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d = d
            best = i
    acc = 0.0
    for i in range(best, len(samples) - 1):
        acc += math.hypot(samples[i + 1][0] - samples[i][0],
                          samples[i + 1][1] - samples[i][1])
        if acc >= RECOVERY_LOOKAHEAD:
            return samples[i + 1]
    return samples[-1]


def build_samples(path):
    samples = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg / RECOVERY_SAMPLE_SPACING)))
        for j in range(n):
            f = j / n
            samples.append((x0 + dx * f, y0 + dy * f))
    samples.append(path[-1])
    return samples


def run_mode(mode):
    """Run forward motion + 90% camera failure + recovery. Return metrics."""
    node = _FakeDashboardNode(mode)
    x = y = 0.0
    yaw = 0.0
    t = 0.0
    dist = 0.0
    last_v = last_w = 0.0

    total = nominal_forward_length(mode)
    trigger_dist = 0.9 * total

    sampler = dream_core.DreamSampler(
        r2_threshold=DREAM_R2_TH,
        min_window_size=DREAM_MIN_WINDOW,
        max_window_size=DREAM_MAX_WINDOW,
        regress_speed=True,
        min_event_spacing=DREAM_MIN_EVENT_SPACING,
        max_window_arc=DREAM_MAX_WINDOW_ARC)

    # Recorded landmarks as (v, w, d, T, psi, x, y) — the YAML layout.
    landmarks = []
    unwrapped_psi = 0.0
    have_psi = False
    last_event_t = None
    last_lm_xy = None

    min_wall_margin = float("inf")
    min_obst_clear = float("inf")
    coll = None

    def update_clearance():
        nonlocal min_wall_margin, min_obst_clear, coll
        for (cx, cy) in robot_corners(x, y, yaw):
            min_wall_margin = min(min_wall_margin, WALL_Y - abs(cy))
        for (ox, oy) in OBSTACLES:
            min_obst_clear = min(
                min_obst_clear, dist_point_rect(ox, oy, x, y, yaw) - OBST_R)
        c = collision_kind(x, y, yaw)
        if c:
            coll = c

    # ------------------------------------------------------------------ #
    # Forward run
    # ------------------------------------------------------------------ #
    while t < 300.0:
        node.cur_x, node.cur_y, node.cur_yaw = x, y, yaw
        res = dashboard_control(node, mode)
        if res is None or res[0] is None:
            break
        v_des, w_des = res

        v = slew(last_v, v_des, dash.MAX_LIN_ACCEL * DT)
        w = slew(last_w, w_des, dash.MAX_ANG_ACCEL * DT)
        w = clamp(w, -dash.MAX_ANGULAR_VELOCITY, dash.MAX_ANGULAR_VELOCITY)
        v = max(0.0, v)
        last_v, last_w = v, w

        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw = norm(yaw + w * DT)
        dist += v * DT
        t += DT

        if not have_psi:
            unwrapped_psi = yaw
            have_psi = True
        else:
            unwrapped_psi += norm(yaw - unwrapped_psi)

        # DREAM sampling (mirrors dream_sampler_node + recovery_recorder_server).
        if sampler.update(t, x, y, dist):
            dt_event = t - last_event_t if last_event_t is not None else 1.0
            frequency = 1.0 / dt_event if dt_event > 1e-3 else 1.0
            last_event_t = t
            T = 1.0 / frequency if frequency > 1e-6 else 0.0
            d = 0.0
            if last_lm_xy is not None:
                d = math.hypot(x - last_lm_xy[0], y - last_lm_xy[1])
            if last_lm_xy is None or d >= 1e-4:
                landmarks.append((v, w, d, T, unwrapped_psi, x, y))
                last_lm_xy = (x, y)

        update_clearance()

        if dist >= trigger_dist:
            break

    if not landmarks:
        return dict(mode=mode, ok=False, reason="no landmarks recorded",
                    final_err=float("nan"), coll=coll, min_wall=min_wall_margin,
                    min_obst=min_obst_clear, n_lm=0)

    # ------------------------------------------------------------------ #
    # Camera failure: pi turn + closed-loop pure-pursuit return.
    # ------------------------------------------------------------------ #
    target = norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = norm(target - yaw)
        if abs(err) < 0.03:
            break
        w = clamp(2.0 * err, -RECOVERY_ROTATE_W, RECOVERY_ROTATE_W)
        yaw = norm(yaw + w * DT)
        t += DT
        rt += DT
        update_clearance()

    # Reconstruct forward waypoints (stored x/y) then reverse them.
    forward = [(lm[5], lm[6]) for lm in landmarks]
    return_path = [(x, y)] + list(reversed(forward))
    if math.hypot(return_path[-1][0], return_path[-1][1]) > 1e-3:
        return_path.append((0.0, 0.0))

    samples = build_samples(return_path)
    last_v = last_w = 0.0

    while t < 480.0:
        dstart = math.hypot(x, y)
        if dstart < RECOVERY_FINAL_TOL:
            break

        gx, gy = pure_pursuit_goal(x, y, samples)
        alpha = norm(math.atan2(gy - y, gx - x) - yaw)

        v_des = RECOVERY_BASE_SPEED * max(0.0, math.cos(alpha))
        approach = clamp(dstart / RECOVERY_APPROACH, 0.0, 1.0)
        v_des = min(v_des, max(0.02, RECOVERY_BASE_SPEED * approach))

        w_des = 2.0 * v_des * math.sin(alpha) / max(RECOVERY_LOOKAHEAD, 1e-3)
        w_des = clamp(w_des, -RECOVERY_MAX_W, RECOVERY_MAX_W)

        v = slew(last_v, v_des, RECOVERY_LIN_ACCEL * DT)
        w = slew(last_w, w_des, RECOVERY_ANG_ACCEL * DT)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = clamp(w, -RECOVERY_MAX_W, RECOVERY_MAX_W)

        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw = norm(yaw + w * DT)
        t += DT
        update_clearance()

    final_err = math.hypot(x, y)
    ok = (final_err < 0.05 and coll is None
          and min_wall_margin >= 0.05 and min_obst_clear >= 0.01)
    reason = ""
    if not ok:
        reason = (f"final_err={final_err:.3f} coll={coll} "
                  f"wall={min_wall_margin:.3f} obst={min_obst_clear:.3f}")

    return dict(mode=mode, ok=ok, reason=reason, final_err=final_err,
                coll=coll, min_wall=min_wall_margin,
                min_obst=min_obst_clear, n_lm=len(landmarks),
                x=x, y=y, t=t)


def main():
    ot.configure(WORLD)
    print("=" * 78)
    print("Offline DREAM auto-mode recovery check "
          "(camera failure at 90% of forward distance)")
    print("=" * 78)

    all_ok = True
    for mode in (1, 2, 3, 4):
        r = run_mode(mode)
        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_ok = False
        print(f"Mode {mode}: {status}  n_lm={r['n_lm']:>3}  "
              f"final_err={r['final_err']:.3f} m  "
              f"wall_margin={r['min_wall']:.3f} m  "
              f"obst_clear={r['min_obst']:.3f} m  coll={r['coll']}  "
              f"t={r['t']:.1f} s")
        if r["reason"]:
            print(f"          reason: {r['reason']}")

    print("-" * 78)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
