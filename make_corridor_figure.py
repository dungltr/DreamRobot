#!/usr/bin/env python3
"""
make_corridor_figure.py

Generate fig9_corridor_demo.pdf: the recovery-corridor system validation
figure. It reproduces the dashboard Auto 4 forward run and the closed-loop
reverse return (camera failure at 100% of the forward distance) using the real
controller functions, and overlays the corridor walls and the two obstacles.

Usage:  python3 make_corridor_figure.py
"""

import math
import os
import sys

WS = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(WS, "src", "vslam_recovery_core", "src")
sys.path.insert(0, SRC)

import optimal_trajectory as ot
import dream_core
import recovery_dashboard_gui as dash

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(WS, "publish")
DT = 1.0 / 20.0
OBSTACLES = [(1.5, 0.3), (3.0, -0.3)]
OBST_R = 0.20
WALL_Y = 0.85


def norm(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def slew(cur, tgt, step):
    d = tgt - cur
    if abs(d) <= step:
        return tgt
    return cur + (step if d > 0 else -step)


class _L:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class _Node:
    def __init__(self):
        self.auto_mode = 4
        self.auto_waypoints = ot.waypoints("recovery_corridor.sdf")
        self.auto_wp_idx = 0
        self.auto_phase = "turn"
        self.auto_active = True
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0

    def get_logger(self):
        return _L()

    def stop_auto_run(self):
        self.auto_active = False


def build_samples(path):
    s = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg / 0.05)))
        for j in range(n):
            f = j / n
            s.append((x0 + dx * f, y0 + dy * f))
    s.append(path[-1])
    return s


def lookahead_goal(x, y, samples, lookahead=0.45):
    best = 0
    bd = float("inf")
    for i, (px, py) in enumerate(samples):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < bd:
            bd = d
            best = i
    acc = 0.0
    for i in range(best, len(samples) - 1):
        acc += math.hypot(samples[i + 1][0] - samples[i][0],
                          samples[i + 1][1] - samples[i][1])
        if acc >= lookahead:
            return samples[i + 1]
    return samples[-1]


def run():
    node = _Node()
    x = y = 0.0
    yaw = 0.0
    t = 0.0
    dist = 0.0
    lv = lw = 0.0
    total = ot._SAMPLED[2][-1]
    trig = 0.9 * total

    sampler = dream_core.DreamSampler(
        r2_threshold=0.98, min_window_size=5, max_window_size=5000,
        regress_speed=True, min_event_spacing=0.2, max_window_arc=1.0)
    events = []
    upsi = 0.0
    have = False
    fwd = []
    trigger_pose = None

    while True:
        node.cur_x, node.cur_y, node.cur_yaw = x, y, yaw
        vd, wd = dash.DashboardNode._control_trajectory(node)
        if vd is None:
            break
        v = slew(lv, vd, dash.MAX_LIN_ACCEL * DT)
        w = slew(lw, wd, dash.MAX_ANG_ACCEL * DT)
        w = clamp(w, -dash.MAX_ANGULAR_VELOCITY, dash.MAX_ANGULAR_VELOCITY)
        v = max(0.0, v)
        lv, lw = v, w
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw = norm(yaw + w * DT)
        dist += v * DT
        t += DT
        fwd.append((x, y))
        if not have:
            upsi = yaw
            have = True
        else:
            upsi += norm(yaw - upsi)
        if sampler.update(t, x, y, dist):
            events.append((x, y))
        if dist >= trig:
            trigger_pose = (x, y)
            break

    # Recovery (pi-turn + closed-loop pure-pursuit over reversed waypoints).
    target = norm(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = norm(target - yaw)
        if abs(err) < 0.03:
            break
        w = clamp(2.0 * err, -0.8, 0.8)
        yaw = norm(yaw + w * DT)
        rt += DT

    path = [(x, y)] + list(reversed(events))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = build_samples(path)
    ret = [(x, y)]
    lv = lw = 0.0
    while True:
        dstart = math.hypot(x, y)
        if dstart < 0.03:
            break
        gx, gy = lookahead_goal(x, y, samples)
        alpha = norm(math.atan2(gy - y, gx - x) - yaw)
        v_des = 0.12 * max(0.0, math.cos(alpha))
        approach = clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, 0.12 * approach))
        w_des = 2.0 * v_des * math.sin(alpha) / 0.45
        w_des = clamp(w_des, -0.25, 0.25)
        v = slew(lv, v_des, 0.20 * DT)
        w = slew(lw, w_des, 0.50 * DT)
        lv, lw = v, w
        v = max(0.0, v)
        w = clamp(w, -0.25, 0.25)
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw = norm(yaw + w * DT)
        ret.append((x, y))

    return fwd, ret, events, trigger_pose


def main():
    ot.configure("recovery_corridor.sdf")
    os.makedirs(OUT, exist_ok=True)
    fwd, ret, events, trigger = run()
    fx = [p[0] for p in fwd]
    fy = [p[1] for p in fwd]
    rx = [p[0] for p in ret]
    ry = [p[1] for p in ret]

    fig, ax = plt.subplots(figsize=(6.4, 2.0))
    # corridor walls
    ax.axhline(WALL_Y, color="0.4", lw=1.0)
    ax.axhline(-WALL_Y, color="0.4", lw=1.0)
    ax.fill_between([-0.5, 12.5], -WALL_Y, WALL_Y, color="0.95", zorder=0)
    # obstacles
    for (ox, oy) in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), OBST_R, color="#c62828", alpha=0.9))
    ax.plot(fx, fy, "g-", lw=1.3, label="forward (Auto 4)")
    ax.plot(rx, ry, "b--", lw=1.1, label="closed-loop return")
    ax.plot([p[0] for p in events], [p[1] for p in events], "ro", ms=4,
            label=f"DREAM landmarks ({len(events)})")
    ax.plot(0, 0, "ko", ms=8)
    ax.annotate("start", (0.05, 0.1), fontsize=7)
    if trigger is not None:
        ax.plot(trigger[0], trigger[1], "m*", ms=12)
        ax.annotate("100% failure", (trigger[0] - 1.4, trigger[1] + 0.25),
                    fontsize=7)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    #ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.legend(
        fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_corridor_demo.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig9_corridor_demo.pdf  landmarks=%d" % len(events))
    print(f"     trigger=({trigger[0]:.2f},{trigger[1]:.2f}) "
          f"final_err={math.hypot(rx[-1], ry[-1]):.3f} m")


if __name__ == "__main__":
    main()
