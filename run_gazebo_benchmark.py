#!/usr/bin/env python3
"""
run_gazebo_benchmark.py

End-to-end DREAM benchmark on Gazebo Fortress for the three benchmark worlds:

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
import re
import signal
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
_WS = os.path.dirname(os.path.realpath(__file__))
_WORLDS_DIR = os.path.join(_WS, "src", "vslam_recovery_core", "worlds")
_BRIDGE_CFG = os.path.join(_WS, "src", "vslam_recovery_core", "config",
                           "bridge_config.yaml")
_DATA_DIR = os.path.join(_WS, "bench_data")

# Scripted open-loop velocity profiles shared with the live GUI dashboard, so
# the monitored run and the reported benchmark drive the robot identically.
sys.path.insert(0, os.path.join(_WS, "src", "vslam_recovery_core", "src"))
from benchmark_profiles import PROFILES  # noqa: E402

WORLDS = [
    ("A", "world_a_straight.sdf"),
    ("B", "world_b_slalom.sdf"),
    ("C", "world_c_scurve.sdf"),
]

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


def make_headless_world(src_path, dst):
    """Strip the camera/IMU sensor plugins so Gazebo runs at a stable real-time
    factor (the odometry does not depend on the camera)."""
    with open(src_path) as fh:
        src = fh.read()
    src = re.sub(
        r'\s*<plugin filename="ignition-gazebo-sensors-system".*?</plugin>\n',
        '\n', src, flags=re.S)
    src = re.sub(r'\s*<sensor[^>]*>.*?</sensor>\n', '\n', src, flags=re.S)
    src = re.sub(
        r'\s*<plugin filename="ignition-gazebo-imu-system".*?</plugin>\n',
        '\n', src, flags=re.S)
    with open(dst, "w") as fh:
        fh.write(src)


def record_world(world, name, segments, node):
    """Drive the robot in `world` and return the recorded raw odometry list.

    The camera sensor is stripped (it is irrelevant to the odometry but forces
    OGRE rendering, which lowers the real-time factor and makes the recorded
    trajectory length machine-dependent), and each segment is driven for `dur`
    seconds of *simulation* time (measured from the odometry timestamps) so the
    recording is deterministic regardless of the real-time factor.
    """
    headless_dir = tempfile.mkdtemp(prefix="dream_bench_")
    headless_world = os.path.join(headless_dir, world)
    make_headless_world(os.path.join(_WORLDS_DIR, world), headless_world)

    gz = _start_proc(["ign", "gazebo", "-s", "-r", headless_world])
    bridge = _start_proc(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                          "--ros-args", "-p", f"config_file:={_BRIDGE_CFG}"])

    try:
        deadline = time.time() + STARTUP_WAIT
        while time.time() < deadline and not node.got_odom():
            _spin(node, 0.1)

        node.clear_odom()
        for v, w, dur in segments:
            start = node.latest_time()
            while start is None:
                node.publish_cmd_vel(v, w)
                _spin(node, 1.0 / DRIVE_FREQ)
                start = node.latest_time()
            while node.latest_time() - start < dur:
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

        def latest_time(self):
            """Simulation time of the most recent odometry sample (or None)."""
            return self._odom[-1][0] if self._odom else None

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


def dream_sampler_deadband(recon, path_threshold=0.98, vel_threshold=0.30,
                           v_max=0.20, min_window_size=5,
                           max_window_size=500, min_event_spacing=0.2):
    """Noise-robust DREAM variant: keep the two-dimensional path linearity but
    replace the R^2 of s(t) with a max-speed-normalized linear-velocity deadband
    (|v - v_anchor|/v_max <= vel_threshold). The deadband ignores the small
    velocity fluctuations that make s(t)-R^2 collapse on short windows, while
    still firing on a genuine acceleration ramp (a ~100% change)."""
    sys.path.insert(0, os.path.join(_WS, "src", "vslam_recovery_core", "src"))
    import dream_core
    path = dream_core.PathLinearityAccumulator()
    window = []
    anchor = None
    last_event_t = None
    anchor_v = None
    idx = []

    for i, (t, x, y, psi, s, v) in enumerate(recon):
        if anchor is None:
            anchor = t
            last_event_t = t
            anchor_v = v
            window.append((0.0, x, y, v))
            path.add(x, y)
            idx.append(i)
            continue
        tl = max(0.0, t - anchor)
        window.append((tl, x, y, v))
        path.add(x, y)
        while len(window) > max_window_size:
            _, x0, y0, _ = window.pop(0)
            path.remove(x0, y0)
        if len(window) < min_window_size:
            continue
        lin = path.linearity()
        e_v = abs(v - anchor_v) / v_max
        if lin >= path_threshold and e_v <= vel_threshold:
            continue
        if t - last_event_t < min_event_spacing:
            continue
        last_event_t = t
        anchor = t
        anchor_v = v
        window = [window[-1]]
        path = dream_core.PathLinearityAccumulator()
        path.add(window[0][1], window[0][2])
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


def recovery_pe_closed_loop_noisy(recon, landmarks_idx, sigma_v, sigma_w, rng):
    """Closed-loop reverse replay with odometry noise injected during the
    RETURN phase (not just the forward phase): the pure-pursuit controller
    steers from a dead-reckoned odometry estimate that drifts from the true
    pose, while the true pose integrates the commanded velocities. Returns the
    true Euclidean distance of the final pose to the origin."""
    if len(landmarks_idx) < 2:
        return float("inf")

    forward = [(recon[i][1], recon[i][2]) for i in landmarks_idx]

    # Failure pose = end of the recorded forward trajectory (estimate == true).
    x = recon[-1][1]
    y = recon[-1][2]
    yaw = recon[-1][3]
    x_est, y_est, yaw_est = x, y, yaw

    dt = 1.0 / 50.0
    kBaseSpeed = 0.12
    kLookahead = 0.45
    kFinalTolerance = 0.03

    # Pi-turn (closed-loop on the estimated heading).
    target = _norm_angle(yaw + math.pi)
    rt = 0.0
    while rt < 12.0:
        err = _norm_angle(target - yaw_est)
        if abs(err) < 0.03:
            break
        w = _clamp(2.0 * err, -0.8, 0.8)
        yaw = _norm_angle(yaw + w * dt)
        yaw_est = _norm_angle(yaw_est + (w + sigma_w * rng.gauss(0.0, 1.0)) * dt)
        rt += dt

    path = [(x, y)] + list(reversed(forward))
    if math.hypot(path[-1][0], path[-1][1]) > 1e-3:
        path.append((0.0, 0.0))
    samples = _polyline_samples(path)

    last_v = 0.0
    last_w = 0.0
    t = 0.0
    while t < 180.0:
        dstart_est = math.hypot(x_est, y_est)
        if dstart_est < kFinalTolerance:
            break
        # Controller steers from the drifted odometry estimate.
        gx, gy = _lookahead_goal(x_est, y_est, samples, kLookahead)
        alpha = _norm_angle(math.atan2(gy - y_est, gx - x_est) - yaw_est)

        v_des = kBaseSpeed * max(0.0, math.cos(alpha))
        approach = _clamp(dstart_est / 0.4, 0.0, 1.0)
        v_des = min(v_des, max(0.02, kBaseSpeed * approach))

        w_des = 2.0 * v_des * math.sin(alpha) / max(kLookahead, 1e-3)
        w_des = _clamp(w_des, -0.25, 0.25)

        v = _slew(last_v, v_des, 0.20 * dt)
        w = _slew(last_w, w_des, 0.50 * dt)
        last_v = v
        last_w = w
        v = max(0.0, v)
        w = _clamp(w, -0.25, 0.25)

        # True pose integrates the commanded velocities.
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = _norm_angle(yaw + w * dt)

        # Odometry estimate integrates noisy velocities (dead-reckoning drift).
        vn = v + sigma_v * rng.gauss(0.0, 1.0)
        wn = w + sigma_w * rng.gauss(0.0, 1.0)
        x_est += vn * math.cos(yaw_est) * dt
        y_est += vn * math.sin(yaw_est) * dt
        yaw_est = _norm_angle(yaw_est + wn * dt)

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
# Noise robustness (reviewer round-7 additions)
# ---------------------------------------------------------------------------
def run_noise_robustness(runs=40, sigma_v=0.01, sigma_w=0.02, path_th=0.98,
                         vel_th=0.30):
    """Two experiments:
      (M1) return-phase odometry noise: the closed-loop replay integrates a
           dead-reckoned estimate, so the same noise affects the RETURN phase;
      (M2) a noise-robust speed channel: replace s(t)-R^2 with a velocity
           deadband and compare landmark counts under the same noise.
    Prints mean +/- 95% CI tables for the three worlds."""
    print("=" * 78)
    print(f"Noise robustness (runs={runs}, sigma_v={sigma_v}, sigma_w={sigma_w})")
    print("=" * 78)
    import random
    for name, _w in WORLDS:
        csv_path = os.path.join(_DATA_DIR, f"odom_{name}.csv")
        raw = []
        with open(csv_path, "r") as fh:
            for row in csv.reader(fh):
                if row:
                    raw.append(tuple(float(v) for v in row))
        recon = reconstruct(raw)
        n_base = len(baseline_sampler(recon))
        fwd_b, fwd_d, both_b, both_d = [], [], [], []
        r2_ns, db_ns = [], []
        for run in range(runs):
            rng = random.Random(5000 + run)
            p = reconstruct_noisy(raw, sigma_v, sigma_w, rng)
            b = baseline_sampler(p)
            d = dream_sampler(p)
            db = dream_sampler_deadband(p, path_threshold=path_th,
                                        vel_threshold=vel_th)
            fwd_b.append(recovery_pe_closed_loop(p, b))
            fwd_d.append(recovery_pe_closed_loop(p, d))
            r1 = random.Random(5000 + run)
            both_b.append(recovery_pe_closed_loop_noisy(p, b, sigma_v, sigma_w, r1))
            r1 = random.Random(5000 + run)
            both_d.append(recovery_pe_closed_loop_noisy(p, d, sigma_v, sigma_w, r1))
            r2_ns.append(len(d))
            db_ns.append(len(db))
        mb, cb = _mean_ci(both_b)
        md, cd = _mean_ci(both_d)
        fb, _ = _mean_ci(fwd_b)
        fd, _ = _mean_ci(fwd_d)
        mr2, cr2 = _mean_ci(r2_ns)
        mdb, cdb = _mean_ci(db_ns)
        print(f"[{name}] N_base={n_base}")
        print(f"  M1 PE (m): fwd-noise-only base={fb:.3f} dream={fd:.3f} | "
              f"both-phase base={mb:.3f}+/-{cb:.3f} dream={md:.3f}+/-{cd:.3f} "
              f"(max {max(both_b):.3f}/{max(both_d):.3f})")
        print(f"  M2 N: R2_s channel {mr2:.1f}+/-{cr2:.1f} | "
              f"velocity-deadband {mdb:.1f}+/-{cdb:.1f}")
    return 0


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
    ap.add_argument("--noise-robustness", action="store_true",
                    help="run return-phase noise + noise-robust speed channel")
    ap.add_argument("--runs", type=int, default=20,
                    help="Monte-Carlo runs (--mc)")
    ap.add_argument("--noise", type=float, default=0.02,
                    help="heading noise sigma for Monte-Carlo (--mc)")
    ap.add_argument("--r2_threshold", type=float, default=0.98)
    args = ap.parse_args()

    os.makedirs(_DATA_DIR, exist_ok=True)

    if args.noise_robustness:
        return run_noise_robustness(runs=args.runs, sigma_w=args.noise)

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
