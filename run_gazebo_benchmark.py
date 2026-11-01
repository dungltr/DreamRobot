#!/usr/bin/env python3
"""
run_gazebo_benchmark.py

End-to-end DREAM benchmark on Gazebo Fortress for the three worlds defined in
DREAMROS2.md Section 2.3:

  * World A (world_a_straight.sdf) : 15 m straight hallway   (max compression)
  * World B (world_b_slalom.sdf)   : 15 m, 4 obstacles slalom
  * World C (world_c_scurve.sdf)   : 20 m long S-curve circuit

For each world the script:
  1. starts Gazebo (headless) + ros_gz_bridge;
  2. drives the robot along a scripted velocity profile via /cmd_vel while
     recording the raw odometry /odom_raw;
  3. shuts the simulation down and saves the odometry to a CSV;
  4. offline: reconstructs the heading (the Gazebo quaternion is broken, so the
     heading is integrated from the angular rate), runs the baseline
     Shannon-Nyquist sampler and the DREAM sampler, and measures:
         N / DR   : landmark count and data reduction ratio,
         PE       : recovery position error (reverse replay of the landmarks).

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 run_gazebo_benchmark.py            # record + compare all 3 worlds
  python3 run_gazebo_benchmark.py --compare  # offline compare on recorded CSVs

Note: the recording mode starts Gazebo with the sensor-enabled worlds (the
camera sensor forces OGRE rendering), so it needs a virtual display when no
physical display is available, e.g.::

    xvfb-run -a python3 run_gazebo_benchmark.py

The --compare / --mc / --ablate modes are offline and need no display.
"""

import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_WS = os.path.dirname(os.path.realpath(__file__))
_WORLDS_DIR = os.path.join(_WS, "src", "vslam_recovery_core", "worlds")
_BRIDGE_CFG = os.path.join(_WS, "src", "vslam_recovery_core", "config",
                           "bridge_config.yaml")
_DATA_DIR = os.path.join(_WS, "bench_data")

WORLDS = [
    ("A", "world_a_straight.sdf"),
    ("B", "world_b_slalom.sdf"),
    ("C", "world_c_scurve.sdf"),
]

# Scripted velocity profiles: list of (v, w, duration_s) segments.
# The cos()/piecewise profiles weave symmetrically so the robot returns near
# the corridor centre (net heading ~ 0) rather than drifting to a wall.
PROFILES = {
    # A: straight hallway (max compression).
    "A": [(0.20, 0.0, 20.0)],
    # B: obstacle slalom (alternating left/right turns).
    "B": [(0.15, 0.0, 1.0), (0.15, 0.5, 2.5), (0.15, -0.5, 2.5),
          (0.15, 0.5, 2.5), (0.15, -0.5, 2.5), (0.15, 0.0, 2.0)],
    # C: long gentle S-curve.
    "C": [(0.12, 0.0, 2.0), (0.12, 0.2, 5.0), (0.12, -0.2, 5.0),
          (0.12, 0.0, 3.0)],
}

DRIVE_FREQ = 20.0       # Hz cmd_vel publishing
STARTUP_WAIT = 12.0     # s to wait for Gazebo + bridge


# ---------------------------------------------------------------------------
# Gazebo recording
# ---------------------------------------------------------------------------
def _start_proc(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)


def _kill(proc):
    """Terminate a launched process and its whole process group.

    Processes are started with start_new_session=True, so proc.pid is the
    group leader. Always escalate to SIGKILL -- even when the wrapper has
    already exited -- so the ``ros2 run`` child node cannot be orphaned and
    pollute the next run with stale /odom or /cmd_vel traffic.
    """
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    time.sleep(0.5)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def record_world(world, name, segments, node):
    """Drive the robot in `world` and return the recorded raw odometry list."""
    gz = _start_proc(["ign", "gazebo", "-s", "-r",
                      os.path.join(_WORLDS_DIR, world)])
    bridge = _start_proc(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                          "--ros-args", "-p", f"config_file:={_BRIDGE_CFG}"])

    try:
        deadline = time.time() + STARTUP_WAIT
        while time.time() < deadline and not node.got_odom():
            _spin(node, 0.1)

        node.clear_odom()
        for v, w, dur in segments:
            t0 = time.time()
            while time.time() - t0 < dur:
                node.publish_cmd_vel(v, w)
                _spin(node, 1.0 / DRIVE_FREQ)
        node.publish_cmd_vel(0.0, 0.0)
        _spin(node, 0.5)

        odom = node.get_odom()
        if odom:
            print(f"  [{name}] recorded {len(odom)} odometry samples "
                  f"({odom[-1][0] - odom[0][0]:.1f} s)")
        else:
            print(f"  [{name}] WARNING: no odometry recorded")
        return odom
    finally:
        # publish_cmd_vel can raise (e.g. RCLError when the ROS context was
        # already shut down); never let that skip the process cleanup below.
        try:
            node.publish_cmd_vel(0.0, 0.0)
        except Exception:
            pass
        _kill(bridge)
        _kill(gz)
        time.sleep(SHUTDOWN_WAIT)


# ---------------------------------------------------------------------------
# ROS node (lazy import so --compare runs without a sourced overlay)
# ---------------------------------------------------------------------------
_node = None
_rclpy = None


def _ensure_ros():
    global _node, _rclpy
    if _node is not None:
        return _node, _rclpy
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    rclpy.init(args=["run_gazebo_benchmark"])

    class BenchNode(Node):
        def __init__(self):
            super().__init__("gazebo_benchmark")
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.odom_sub = self.create_subscription(
                Odometry, "/odom_raw", self._odom_cb, 100)
            self._odom = []
            self._have = False

        def _odom_cb(self, msg):
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._odom.append((t,
                               msg.pose.pose.position.x,
                               msg.pose.pose.position.y,
                               msg.twist.twist.linear.x,
                               msg.twist.twist.angular.z))
            self._have = True

        def got_odom(self):
            return self._have

        def clear_odom(self):
            self._odom = []
            self._have = False

        def get_odom(self):
            return list(self._odom)

        def publish_cmd_vel(self, v, w):
            m = Twist()
            m.linear.x = v
            m.angular.z = w
            self.cmd_pub.publish(m)

    _node = BenchNode()
    _rclpy = rclpy
    return _node, _rclpy


def _spin(node, timeout):
    _rclpy.spin_once(node, timeout_sec=timeout)


# ---------------------------------------------------------------------------
# Offline comparison
# ---------------------------------------------------------------------------
def reconstruct(raw):
    """Return list of (t, x, y, psi, s, v); heading integrated from angular rate."""
    out = []
    psi = 0.0
    s = 0.0
    prev = None
    for t, x, y, v, w in raw:
        if prev is not None:
            dt = t - prev[0]
            if 0.0 < dt < 0.5:
                psi += w * dt
            s += math.hypot(x - prev[1], y - prev[2])
        out.append((t, x, y, psi, s, v))
        prev = (t, x, y)
    return out


def _norm_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def baseline_sampler(recon, straight_period=2.0, curve_period=1.0):
    """Return landmark indices via the 0.5/1.0 Hz yaw-window heuristic."""
    from collections import deque
    yaw_window = deque(maxlen=4)
    period = straight_period
    elapsed = 0.0
    last_t = None
    idx = []

    def is_curved():
        if len(yaw_window) < 4:
            return False
        y = list(yaw_window)
        d0, d1, d2 = y[1] - y[0], y[2] - y[1], y[3] - y[2]
        if (d0 > 0 and d1 > 0 and d2 > 0) or (d0 < 0 and d1 < 0 and d2 < 0):
            return True
        return any(math.pi <= abs(d) <= 2 * math.pi for d in (d0, d1, d2))

    for i, (t, x, y, psi, s, v) in enumerate(recon):
        yaw_window.append(psi)
        if last_t is not None:
            elapsed += t - last_t
        last_t = t
        period = curve_period if is_curved() else straight_period
        if elapsed >= period:
            elapsed = 0.0
            idx.append(i)
    return idx


def dream_sampler(recon, r2_threshold=0.98, regress_speed=True,
                 max_window_size=500):
    """Return landmark indices via the DREAM R^2-triggered sampler."""
    sys.path.insert(0, os.path.join(_WS, "src", "vslam_recovery_core", "src"))
    import dream_core
    sampler = dream_core.DreamSampler(
        r2_threshold=r2_threshold, regress_speed=regress_speed,
        max_window_size=max_window_size)
    idx = []
    for i, (t, x, y, psi, s, v) in enumerate(recon):
        if sampler.update(t, x, y, s):
            idx.append(i)
    return idx


def recovery_pe(recon, landmarks_idx):
    """Simulate the reverse replay and return the Euclidean return error (m)."""
    if len(landmarks_idx) < 2:
        return float("inf")

    ti = landmarks_idx[-1]
    x, y, yaw = recon[ti][1], recon[ti][2], recon[ti][3]
    yaw += math.pi                        # pi-turn
    dt = 1.0 / 50.0                       # replay integration step

    for k in range(len(landmarks_idx) - 1, -1, -1):
        i = landmarks_idx[k]
        v = recon[i][5]                   # recorded linear velocity
        target = _norm_angle(recon[i][3] + math.pi)
        # Segment duration = time to the PREVIOUS landmark (0 for the start).
        dur = recon[i][0] - recon[landmarks_idx[k - 1]][0] if k > 0 else 0.0
        t = 0.0
        while t < dur:
            err = _norm_angle(target - yaw)
            w = max(-0.25, min(0.25, 2.0 * err))
            yaw += w * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            t += dt
    return math.hypot(x, y)


# ---------------------------------------------------------------------------
# Closed-loop pure-pursuit reverse replay (matches recovery_controller.cpp).
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _slew(current, target, max_step):
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + (max_step if delta > 0.0 else -max_step)


def _polyline_samples(path, spacing=0.05):
    """Dense interpolation of a polyline for a smooth lookahead point."""
    samples = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg = math.hypot(dx, dy)
        n = max(1, int(math.ceil(seg / spacing)))
        for j in range(n):
            f = j / n
            samples.append((x0 + dx * f, y0 + dy * f))
    samples.append(path[-1])
    return samples


def _lookahead_goal(x, y, samples, lookahead=0.45):
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
        if acc >= lookahead:
            return samples[i + 1]
    return samples[-1]


def recovery_pe_closed_loop(recon, landmarks_idx):
    """Closed-loop reverse replay: pi-turn + pure-pursuit over the recorded
    waypoints, using odometry position/heading feedback only (no camera).

    Returns the Euclidean distance of the final pose to the origin.
    """
    if len(landmarks_idx) < 2:
        return float("inf")

    # Forward path positions recorded at each landmark.
    forward = [(recon[i][1], recon[i][2]) for i in landmarks_idx]

    # Failure pose = end of the recorded forward trajectory.
    x = recon[-1][1]
    y = recon[-1][2]
    yaw = recon[-1][3]

    dt = 1.0 / 50.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    # Pi-turn (closed-loop on heading).
    target = _norm_angle(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = _norm_angle(target - yaw)
        if abs(err) < 0.03:
            break
        w = _clamp(2.0 * err, -0.8, 0.8)
        yaw = _norm_angle(yaw + w * dt)
        rt += dt

    # Return path: failure pose -> recorded waypoints in reverse -> origin.
    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = _polyline_samples(path)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart = math.hypot(x, y)
        if dstart < kFinalTolerance:
            break
        gx, gy = _lookahead_goal(x, y, samples, kLookahead)
        alpha = _norm_angle(math.atan2(gy - y, gx - x) - yaw)

        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = _clamp(dstart / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))

        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = _clamp(w_des, -0.25, 0.25)

        v = _slew(last_v, v_des, 0.20 * dt)
        w = _slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = _clamp(w, -0.25, 0.25)

        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = _norm_angle(yaw + w * dt)
        t += dt
    return math.hypot(x, y)


# ---------------------------------------------------------------------------
# Monte-Carlo + ablation (offline, on the recorded CSVs)
# ---------------------------------------------------------------------------
def reconstruct_noisy(raw, sigma_v, sigma_w, rng):
    """Re-integrate (x, y, psi, s) from noisy (v, w) — the standard odometry
    noise model (velocity noise -> integrated drift) — returning the same
    (t, x, y, psi, s, v) tuple layout as reconstruct()."""
    out = []
    x = 0.0
    y = 0.0
    psi = 0.0
    s = 0.0
    prev = None
    for (t, _x, _y, v, w) in raw:
        vn = max(0.0, v + sigma_v * rng.gauss(0.0, 1.0))
        wn = w + sigma_w * rng.gauss(0.0, 1.0)
        if prev is not None:
            dt = t - prev
            if 0.0 < dt < 0.5:
                psi += wn * dt
                x += vn * math.cos(psi) * dt
                y += vn * math.sin(psi) * dt
                s += vn * dt
        out.append((t, x, y, psi, s, vn))
        prev = t
    return out


def _mean_ci(values, z=1.96):
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, z * (var / len(values)) ** 0.5


def run_mc(csv_path, name, runs=20, sigma_v=0.01, sigma_w=0.02,
           r2_threshold=0.98):
    """Monte-Carlo: re-integrate the recorded odometry `runs` times with noisy
    velocity/angular-rate and report the mean +/- 95% CI of DR and PE."""
    import random
    raw = []
    with open(csv_path, "r") as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    recon = reconstruct(raw)
    b_idx = baseline_sampler(recon)
    pe_b = recovery_pe_closed_loop(recon, b_idx)

    ns, drs, pes = [], [], []
    for run in range(runs):
        rng = random.Random(5000 + run)
        p = reconstruct_noisy(raw, sigma_v, sigma_w, rng)
        d_idx = dream_sampler(p, r2_threshold=r2_threshold)
        ns.append(len(d_idx))
        drs.append(1.0 - len(d_idx) / len(b_idx))
        pes.append(recovery_pe_closed_loop(p, d_idx))

    n_m, n_c = _mean_ci(ns)
    dr_m, dr_c = _mean_ci(drs)
    pe_m, pe_c = _mean_ci([v for v in pes if v != float("inf")])
    print(f"  [{name}] Monte-Carlo ({runs} runs, sigma_v={sigma_v}, "
          f"sigma_w={sigma_w}): N_dream={n_m:.1f}±{n_c:.1f}  "
          f"DR={dr_m:6.1%}±{dr_c:.1%}  "
          f"PE_dream_cl={pe_m:.3f}±{pe_c:.3f} m  (PE_base_cl={pe_b:.3f} m)")
    return {"world": name, "n": (n_m, n_c), "dr": (dr_m, dr_c),
            "pe": (pe_m, pe_c), "pe_base": pe_b}


def ablation(csv_path, name, r2_threshold=0.98):
    """Ablation: contribution of the speed signal and the effect of window size."""
    recon = load_recon(csv_path)
    b_idx = baseline_sampler(recon)
    pe_b = recovery_pe_closed_loop(recon, b_idx)

    print(f"  [{name}] ablation (R2_th={r2_threshold}, N_base={len(b_idx)}, "
          f"PE_base_cl={pe_b:.3f} m)")
    print(f"    {'config':<28} | {'N_dream':>7} | {'DR':>7} | {'PE_cl':>9}")
    # (a) speed signal on/off
    for rs in (True, False):
        d_idx = dream_sampler(recon, r2_threshold=r2_threshold, regress_speed=rs)
        dr = 1.0 - len(d_idx) / len(b_idx)
        pe = recovery_pe_closed_loop(recon, d_idx)
        print(f"    {'regress_speed=' + str(rs):<28} | {len(d_idx):>7} | "
              f"{dr:>6.1%} | {pe:>8.3f}")
    # (b) window size sweep
    for mw in (50, 100, 250, 500):
        d_idx = dream_sampler(recon, r2_threshold=r2_threshold,
                              max_window_size=mw)
        dr = 1.0 - len(d_idx) / len(b_idx)
        pe = recovery_pe_closed_loop(recon, d_idx)
        print(f"    {'max_window_size=' + str(mw):<28} | {len(d_idx):>7} | "
              f"{dr:>6.1%} | {pe:>8.3f}")


def load_recon(csv_path):
    raw = []
    with open(csv_path, "r") as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    return reconstruct(raw)


def compare(csv_path, name):
    if not os.path.exists(csv_path):
        print(f"  [{name}] missing {csv_path}")
        return None

    raw = []
    with open(csv_path, "r") as fh:
        for row in csv.reader(fh):
            if row:
                raw.append(tuple(float(v) for v in row))
    recon = reconstruct(raw)
    n_total = len(recon)

    b_idx = baseline_sampler(recon)
    d_idx = dream_sampler(recon)

    dr = 1.0 - len(d_idx) / len(b_idx) if b_idx else 0.0
    pe_b = recovery_pe(recon, b_idx)
    pe_d = recovery_pe(recon, d_idx)
    pe_b_cl = recovery_pe_closed_loop(recon, b_idx)
    pe_d_cl = recovery_pe_closed_loop(recon, d_idx)

    print(f"  [{name}] N_total={n_total}  N_baseline={len(b_idx)} "
          f"N_dream={len(d_idx)}  DR={dr:6.1%}  "
          f"PE_baseline={pe_b:.3f} m  PE_dream={pe_d:.3f} m  "
          f"PE_base_cl={pe_b_cl:.3f} m  PE_dream_cl={pe_d_cl:.3f} m")
    return {"world": name, "n_total": n_total, "n_baseline": len(b_idx),
            "n_dream": len(d_idx), "dr": dr, "pe_baseline": pe_b,
            "pe_dream": pe_d, "pe_baseline_cl": pe_b_cl,
            "pe_dream_cl": pe_d_cl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SHUTDOWN_WAIT = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="only run the offline comparison on recorded CSVs")
    ap.add_argument("--mc", action="store_true",
                    help="run Monte-Carlo noise analysis on recorded CSVs")
    ap.add_argument("--ablate", action="store_true",
                    help="run ablation (speed signal + window size) on CSVs")
    ap.add_argument("--runs", type=int, default=20,
                    help="Monte-Carlo runs (--mc)")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="heading noise sigma for Monte-Carlo (--mc)")
    ap.add_argument("--r2_threshold", type=float, default=0.98)
    args = ap.parse_args()

    os.makedirs(_DATA_DIR, exist_ok=True)

    if args.compare or args.mc or args.ablate:
        for name, _w in WORLDS:
            csv_path = os.path.join(_DATA_DIR, f"odom_{name}.csv")
            if args.compare:
                compare(csv_path, name)
            if args.mc:
                run_mc(csv_path, name, runs=args.runs, sigma_w=args.noise,
                       r2_threshold=args.r2_threshold)
            if args.ablate:
                ablation(csv_path, name, r2_threshold=args.r2_threshold)
        return 0

    node, _rclpy = _ensure_ros()
    results = []
    for name, world in WORLDS:
        print(f"\n=== World {name} ({world}) ===")
        odom = record_world(world, name, PROFILES[name], node)
        if not odom:
            print(f"  [{name}] no odometry recorded; skipping.")
            continue
        csv_path = os.path.join(_DATA_DIR, f"odom_{name}.csv")
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerows(odom)
        print(f"  [{name}] saved {csv_path}")
        res = compare(csv_path, name)
        if res:
            results.append(res)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'world':>6} | {'N_base':>7} | {'N_dream':>7} | {'DR':>7} | "
          f"{'PE_base':>9} | {'PE_dream':>9}")
    print("-" * 72)
    for r in results:
        print(f"{r['world']:>6} | {r['n_baseline']:>7} | {r['n_dream']:>7} | "
              f"{r['dr']:>6.1%} | {r['pe_baseline']:>8.3f} | "
              f"{r['pe_dream']:>8.3f}")
    print("-" * 72)

    if node is not None:
        node.destroy_node()
        _rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
