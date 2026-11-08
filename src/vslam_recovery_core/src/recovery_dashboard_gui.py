#!/usr/bin/env python3
"""
recovery_dashboard_gui.py

tkinter + rclpy + matplotlib dashboard for the blind navigation recovery demo.

Real-time telemetry (from /odom):
  * instantaneous velocity (v, w), pose (x, y),
  * number of path landmarks recorded in the YAML file,
  * current path segment (straight 0.5 Hz / curved 1.0 Hz),
  * forward distance D and recovery distance CD.

Real-time diagnostic plots (updated at ~20 Hz):
  * yaw angle variation vs time,
  * recorded angular velocity w and linear velocity v vs time, with the
    sampling-frequency transition regions (0.5 Hz straight / 1.0 Hz curved)
    highlighted.

Auto-run modes (closed-loop following through the 12 m corridor):
  1. straight + in-place rotation, fixed speed v = 0.08 m/s;
  2. straight + in-place rotation, variable speed (0.05 m/s near the
     obstacles, 0.15 m/s on the straights);
  3. continuous optimal curve, fixed LSV = 0.086 m/s (ASV = 0.097 rad/s
     nominal, hard safety clamp |w| <= 0.25 rad/s);
  4. continuous optimal curve, variable speed profile 0.04 .. 0.086 m/s
     through the S-curve, 0.20 m/s on the final straight.

Modes 3 and 4 track a C2-continuous quintic-spline path (optimal_trajectory.py)
with adaptive pure-pursuit lookahead (0.45 m in the S-curve, 0.8 m on straights)
so the motion is smooth (no velocity/acceleration discontinuities) and the
obstacles are avoided with 0.55 m centre clearance.

Other controls:
  * manual teleop (buttons + arrow keys + Stop);
  * "CUT CAMERA SIGNAL NOW (return from this pose)" -> /camera_failure_trigger;
  * auto camera cutoff at 70 / 80 / 90 / 100% of the forward distance
    (default 100%);
  * "Compute Position Error (PE %)" -> RM and PE (Equation 2);
  * "Benchmark Experiment (open-loop profile)" -> drives the scripted
    open-loop (v, omega) profile of the currently loaded benchmark world, then
    auto-cuts the camera and returns (same profiles as run_gazebo_benchmark.py).
"""

import math
import os
import re
import sys
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

# Allow importing the co-located optimal-trajectory helper after colcon
# installation (scripts live in the same lib/<pkg> directory).
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32, String

import optimal_trajectory as ot
from benchmark_profiles import (
    PROFILES, profile_duration, benchmark_key_for_world,
)

# ---------------------------------------------------------------------------
# Waypoints for the segmented turn-then-drive modes (Features 1 and 2).
# The apexes and centre crossing match the optimal path clearance geometry.
# ---------------------------------------------------------------------------
WAYPOINTS = [
    (0.0, 0.0),
    (1.5, -0.25),
    (2.25, 0.0),
    (3.0, 0.25),
    (4.0, 0.0),
    (12.0, 0.0),
]

AUTO_CONTROL_RATE = 20.0       # Hz
BENCHMARK_DRIVE_FREQ = 20.0   # Hz, open-loop profile publishing
AUTO_WP_TOLERANCE = 0.10       # m
AUTO_FINAL_TOLERANCE = 0.05    # m
MANUAL_SPEED = 0.12            # m/s, fixed manual-teleop speed

# Hard safety limits (absolute values).
MAX_ANGULAR_VELOCITY = ot.MAX_ANGULAR_VELOCITY   # 0.25 rad/s
MAX_LIN_ACCEL = 0.20            # m/s^2 (slew-rate limit for smooth ramps)
MAX_ANG_ACCEL = 0.50            # rad/s^2

# Per-mode speed profiles.
MODE1_LIN = 0.08                # fixed linear speed (m/s)
MODE2_LIN_OBSTACLE = 0.05       # m/s, near obstacles
MODE2_LIN_STRAIGHT = 0.15       # m/s, on straights
MODE2_ANG = 0.25                # rad/s, in-place / corrective turn
MODE3_LSV = 0.086               # Section 5.1 linear set velocity
MODE3_ASV = 0.097               # Section 5.1 angular set velocity (nominal)
MODE4_LIN_CURVE_MIN = 0.04      # m/s, entering the S-curve
MODE4_LIN_CURVE_MAX = 0.086     # m/s, leaving the S-curve
MODE4_LIN_STRAIGHT = 0.20       # m/s, final straight

STRAIGHT_FREQ = 0.5             # Hz
CURVE_FREQ = 1.0                # Hz

_LM_RE = re.compile(r"^landmark_(\d+):\s*\[")
_PLOT_MAX_POINTS = 600          # ~30 s at 20 Hz


def clamp(value, low, high):
    return max(low, min(high, value))


def slew(current, target, max_step):
    """Move `current` towards `target` by at most `max_step` (slew limiter)."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step if delta > 0.0 else current - max_step


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def sanitize_quaternion(q):
    """Return a normalized (x, y, z, w) tuple; identity when NaN or zero-norm."""
    vals = (q.x, q.y, q.z, q.w)
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return (0.0, 0.0, 0.0, 1.0)
    norm2 = sum(v * v for v in vals)
    if norm2 < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / math.sqrt(norm2)
    return (vals[0] * inv, vals[1] * inv, vals[2] * inv, vals[3] * inv)


class DashboardNode(Node):
    def __init__(self):
        super().__init__('recovery_dashboard_gui')

        self.odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self.trigger_topic = self.declare_parameter(
            'failure_trigger_topic', '/camera_failure_trigger').value
        self.yaml_file = self.declare_parameter(
            'yaml_file',
            os.path.expanduser('~/codewhale_recoveryRobotDREAM/path_landmarks.yaml')).value
        self.auto_run = self.declare_parameter('auto_run', False).value
        self.world = self.declare_parameter('world', 'recovery_corridor.sdf').value
        self.strategy = self.declare_parameter('strategy', 'geometry').value
        ot.configure(self.world)

        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.trigger_sub = self.create_subscription(Bool, self.trigger_topic, self.trigger_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)
        self.mode_pub = self.create_publisher(Int32, 'auto_mode', 10)
        self.strategy_pub = self.create_publisher(String, 'dream_strategy', 10)

        # Live state.
        self.v = 0.0
        self.w = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        self.yaw_window = []
        self.state = 'forward'           # 'forward' | 'recovery'
        self.forward_dist = 0.0
        self.recovery_dist = 0.0
        self.last_pos = None
        self.pe_computed = False
        self.pe_result = None

        # Automatic camera-cutoff threshold (percentage of the forward path).
        self.failure_percent = 100      # 70 / 80 / 90 / 100
        self.failure_fired = False      # latch: fire the auto-cutoff once per run

        # Closed-loop auto-navigation state.
        self.auto_mode = 3
        self.auto_waypoints = ot.waypoints(self.world)
        self.auto_wp_idx = 0
        self.auto_phase = 'turn'
        self.auto_active = False
        self.auto_timer = None
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0

        # Benchmark experiment state (open-loop World A/B/C profiles).
        self.benchmark_active = False
        self.benchmark_timer = None
        self.benchmark_key = None
        self.benchmark_segments = []
        self.benchmark_start = None

        self.get_logger().info(f"Dashboard node ready. yaml_file='{self.yaml_file}'")

    # ------------------------------------------------------------------ #
    # Subscriptions
    # ------------------------------------------------------------------ #
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            self.get_logger().warn(
                'NaN/infinite position in /odom; ignoring frame.',
                throttle_duration_sec=5.0)
            return

        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z
        self.cur_x = p.x
        self.cur_y = p.y

        qx, qy, qz, qw = sanitize_quaternion(msg.pose.pose.orientation)
        self.cur_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                  1.0 - 2.0 * (qy * qy + qz * qz))
        self.yaw_window.append(self.cur_yaw)
        if len(self.yaw_window) > 4:
            self.yaw_window.pop(0)

        if self.last_pos is not None:
            step = math.hypot(p.x - self.last_pos[0], p.y - self.last_pos[1])
            if self.state == 'forward':
                self.forward_dist += step
            else:
                self.recovery_dist += step
        self.last_pos = (p.x, p.y)

    def trigger_cb(self, msg):
        if msg.data and self.state == 'forward':
            self.state = 'recovery'
            # Stop any active auto-run / manual stream so the dashboard stops
            # publishing /cmd_vel and lets recovery_controller drive the return.
            self.stop_auto_run()
            self.get_logger().info('Camera failure trigger observed.')

    # ------------------------------------------------------------------ #
    # Segment classification (same rule as landmark_sampler_client)
    # ------------------------------------------------------------------ #
    def segment_state(self):
        if len(self.yaw_window) < 4:
            return 'straight'
        d0 = self.yaw_window[1] - self.yaw_window[0]
        d1 = self.yaw_window[2] - self.yaw_window[1]
        d2 = self.yaw_window[3] - self.yaw_window[2]
        if (d0 > 0 and d1 > 0 and d2 > 0) or (d0 < 0 and d1 < 0 and d2 < 0):
            return 'curved'
        for d in (d0, d1, d2):
            if math.pi <= abs(d) <= 2.0 * math.pi:
                return 'curved'
        return 'straight'

    def landmark_count(self):
        try:
            with open(self.yaml_file, 'r') as fh:
                return sum(1 for line in fh if _LM_RE.match(line.strip()))
        except (OSError, IOError):
            return 0

    # ------------------------------------------------------------------ #
    # Closed-loop auto navigation (4 modes)
    # ------------------------------------------------------------------ #
    def start_auto_run(self, mode):
        self.auto_mode = mode
        self.mode_pub.publish(Int32(data=mode))
        self.auto_waypoints = ot.waypoints(self.world)
        self.auto_wp_idx = 0
        self.auto_phase = 'turn'
        self.auto_active = True
        # A fresh run starts from the origin: reset the travelled distances
        # and re-arm the automatic camera cutoff for this run.
        self.state = 'forward'
        self.forward_dist = 0.0
        self.recovery_dist = 0.0
        self.last_pos = None
        self.failure_fired = False
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0
        if self.auto_timer is None:
            self.auto_timer = self.create_timer(
                1.0 / AUTO_CONTROL_RATE, self.auto_control_tick)
        self.get_logger().info(f'Auto run mode {mode} started.')

    def stop_auto_run(self):
        stopped = False
        if self.auto_active or self.auto_timer is not None:
            self.auto_active = False
            if self.auto_timer is not None:
                self.auto_timer.cancel()
                self.auto_timer = None
            self.auto_last_v = 0.0
            self.auto_last_w = 0.0
            stopped = True
        if self.benchmark_active or self.benchmark_timer is not None:
            self.benchmark_active = False
            if self.benchmark_timer is not None:
                self.benchmark_timer.cancel()
                self.benchmark_timer = None
            stopped = True
        if stopped:
            self.cmd_pub.publish(Twist())
            self.get_logger().info('Auto/benchmark run stopped.')

    def emergency_stop(self):
        self.stop_auto_run()
        self.cmd_pub.publish(Twist())
        self.get_logger().warn('EMERGENCY STOP pressed.')

    def auto_control_tick(self):
        if not self.auto_active:
            return

        if self.auto_mode in (1, 2):
            v_des, w_des = self._control_waypoint()
        else:
            v_des, w_des = self._control_trajectory()

        if v_des is None:
            # Mission finished (the robot reached 100% of the forward path).
            # If the camera has not been cut yet (threshold = 100%), cut it
            # now so the robot returns from the end pose.
            if not self.failure_fired and self.state == 'forward':
                self.failure_fired = True
                self.get_logger().info('Auto camera cutoff at 100% (end of path).')
                self.trigger_failure()
            return  # mission finished / already stopped

        # Slew-rate limits -> smooth acceleration, no velocity/steering jumps.
        dt = 1.0 / AUTO_CONTROL_RATE
        v = slew(self.auto_last_v, v_des, MAX_LIN_ACCEL * dt)
        w = slew(self.auto_last_w, w_des, MAX_ANG_ACCEL * dt)
        w = clamp(w, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        v = max(0.0, v)
        self.auto_last_v = v
        self.auto_last_w = w

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

        # Automatic camera cutoff: when the forward distance reaches the
        # selected percentage of the nominal path length, fail the camera now
        # (the recovery controller then drives the return from this pose).
        if (not self.failure_fired and self.state == 'forward'
                and self.forward_dist >= (self.failure_percent / 100.0)
                * self.nominal_forward_length()):
            self.failure_fired = True
            self.get_logger().info(
                f'Auto camera cutoff at {self.failure_percent}% '
                f'(D={self.forward_dist:.3f} m).')
            self.trigger_failure()

    def _control_waypoint(self):
        """Segmented turn-then-drive following for Features 1 and 2."""
        if self.auto_wp_idx >= len(self.auto_waypoints):
            self.stop_auto_run()
            return None, None

        tx, ty = self.auto_waypoints[self.auto_wp_idx]
        dx = tx - self.cur_x
        dy = ty - self.cur_y
        dist = math.hypot(dx, dy)

        is_final = self.auto_wp_idx == len(self.auto_waypoints) - 1
        tol = AUTO_FINAL_TOLERANCE if is_final else AUTO_WP_TOLERANCE

        if dist < tol:
            self.get_logger().info(f'Reached waypoint {self.auto_wp_idx}.')
            self.auto_wp_idx += 1
            self.auto_phase = 'turn'
            if self.auto_wp_idx >= len(self.auto_waypoints):
                self.stop_auto_run()
                return None, None
            tx, ty = self.auto_waypoints[self.auto_wp_idx]
            dx = tx - self.cur_x
            dy = ty - self.cur_y
            dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        heading_error = normalize_angle(target_yaw - self.cur_yaw)

        lin = 0.0
        ang = 0.0

        if self.auto_phase == 'turn':
            lin = 0.0
            ang = clamp(2.0 * heading_error, -MODE2_ANG, MODE2_ANG)
            if abs(heading_error) < 0.04:
                self.auto_phase = 'drive'
        else:
            ang = clamp(1.5 * heading_error, -MODE2_ANG, MODE2_ANG)
            if self.auto_mode == 1:
                lin = MODE1_LIN
            else:
                # Slow near the obstacle zone, faster on the clear straights.
                near_obstacle = 0.8 <= self.cur_x <= 4.2
                lin = MODE2_LIN_OBSTACLE if near_obstacle else MODE2_LIN_STRAIGHT
                lin = min(lin, max(0.0, dist) + 0.02)

        return lin, ang

    def _control_trajectory(self):
        """Continuous quintic-spline following for Features 3 and 4."""
        x = self.cur_x
        y = self.cur_y
        yaw = self.cur_yaw

        # Arrival at the corridor end.
        if x >= ot.GOAL_X - AUTO_FINAL_TOLERANCE:
            self.get_logger().info('Reached the end of the optimal trajectory.')
            self.stop_auto_run()
            return None, None

        lookahead = ot.adaptive_lookahead(x)
        gx, gy, _gyaw = ot.lookahead_point(x, y, lookahead)

        # Pure pursuit: steer toward the lookahead point.
        alpha = normalize_angle(math.atan2(gy - y, gx - x) - yaw)

        if self.auto_mode == 3:
            v_des = MODE3_LSV
        else:
            if x < ot.CURVE_END_X:
                # Ramp 0.04 -> 0.086 m/s through the S-curve.
                frac = clamp(x / ot.CURVE_END_X, 0.0, 1.0)
                v_des = MODE4_LIN_CURVE_MIN + \
                    (MODE4_LIN_CURVE_MAX - MODE4_LIN_CURVE_MIN) * frac
            else:
                v_des = MODE4_LIN_STRAIGHT

        # Slow down proportionally when far off-course to avoid cutting corners.
        v_des = v_des * max(0.0, math.cos(alpha))

        w_des = 2.0 * v_des * math.sin(alpha) / max(lookahead, 1e-3)
        w_des = clamp(w_des, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)

        return v_des, w_des

    # ------------------------------------------------------------------ #
    # Benchmark experiment (open-loop World A/B/C profiles)
    # ------------------------------------------------------------------ #
    def start_benchmark(self, key):
        """Drive the scripted open-loop profile for benchmark World A/B/C,
        then cut the camera at the configured percentage (default 100%) and
        hand control to the recovery controller."""
        if key not in PROFILES:
            self.get_logger().error(f'Unknown benchmark world {key!r}.')
            return
        self.stop_auto_run()
        self.benchmark_key = key
        self.benchmark_segments = list(PROFILES[key])
        self.benchmark_start = self.get_clock().now()
        self.benchmark_active = True
        # Fresh run from the origin: reset travelled distance and re-arm the
        # automatic camera cutoff, mirroring start_auto_run().
        self.state = 'forward'
        self.forward_dist = 0.0
        self.recovery_dist = 0.0
        self.last_pos = None
        self.failure_fired = False
        self.pe_computed = False
        if self.benchmark_timer is None:
            self.benchmark_timer = self.create_timer(
                1.0 / BENCHMARK_DRIVE_FREQ, self.benchmark_tick)
        self.get_logger().info(
            f'Benchmark run {key} started '
            f'({len(self.benchmark_segments)} segments).')

    def benchmark_tick(self):
        if not self.benchmark_active:
            return
        elapsed = (self.get_clock().now()
                   - self.benchmark_start).nanoseconds * 1e-9
        total = profile_duration(self.benchmark_key)

        # Camera auto-cutoff at the selected fraction of the nominal profile.
        if elapsed >= (self.failure_percent / 100.0) * total:
            self.stop_benchmark()
            if not self.failure_fired and self.state == 'forward':
                self.failure_fired = True
                self.get_logger().info(
                    f'Benchmark camera cutoff at {self.failure_percent}% '
                    f'(t={elapsed:.2f} s).')
                self.trigger_failure()
            return

        # Select the current (v, w) segment by cumulative duration.
        acc = 0.0
        v = w = 0.0
        for seg_v, seg_w, seg_dur in self.benchmark_segments:
            if elapsed < acc + seg_dur:
                v, w = seg_v, seg_w
                break
            acc += seg_dur

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

    def stop_benchmark(self):
        self.benchmark_active = False
        if self.benchmark_timer is not None:
            self.benchmark_timer.cancel()
            self.benchmark_timer = None
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------------ #
    # Manual teleop
    # ------------------------------------------------------------------ #
    def manual_move(self, linear_x, angular_z):
        self.stop_auto_run()
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = clamp(angular_z, -MAX_ANGULAR_VELOCITY,
                                MAX_ANGULAR_VELOCITY)
        self.cmd_pub.publish(twist)

    def manual_stop(self):
        self.stop_auto_run()
        self.cmd_pub.publish(Twist())

    def stop_motion(self):
        """Stop the manual teleop stream without cancelling an active auto run."""
        self.cmd_pub.publish(Twist())

    def trigger_failure(self):
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)
        self.state = 'recovery'
        # Stop the dashboard's own cmd_vel stream immediately so it does not
        # fight the recovery controller during the return trip.
        self.stop_auto_run()
        self.get_logger().info('Camera failure trigger published.')

    def nominal_forward_length(self):
        """Total forward path length (m) for the current auto mode."""
        if self.auto_mode in (1, 2):
            wps = ot.waypoints(self.world)
            return sum(math.hypot(wps[i + 1][0] - wps[i][0],
                                  wps[i + 1][1] - wps[i][1])
                       for i in range(len(wps) - 1))
        return ot._SAMPLED[2][-1]

    def set_failure_percent(self, pct):
        """Set the automatic camera-cutoff threshold and re-arm it."""
        self.failure_percent = int(pct)
        self.failure_fired = False
        self.get_logger().info(
            f'Camera auto-cutoff set to {self.failure_percent}%.')

    def compute_error(self):
        d_forward = self.forward_dist
        d_recovery = self.recovery_dist
        if d_forward <= 1e-6:
            return None
        pe = abs(d_forward - d_recovery) / d_forward * 100.0
        rm = d_forward - d_recovery
        self.pe_result = (d_forward, d_recovery, pe, rm)
        self.pe_computed = True
        return self.pe_result


class DashboardApp:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Blind Navigation Recovery - Dashboard')
        self.root.geometry('820x900')

        self._last_yaml_read = 0.0
        self._landmarks = 0
        self._stopped_ticks = 0

        # Real-time plot history (sampled at the GUI tick rate).
        self._t0 = time.time()
        self.times = []
        self.yaw_hist = []
        self.v_hist = []
        self.w_hist = []
        self.seg_hist = []

        pad = {'padx': 10, 'pady': 3}

        # ---- Scrollable container ---------------------------------------- #
        container = tk.Frame(self.root)
        container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.vsb = ttk.Scrollbar(container, orient='vertical',
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.scrollable_frame = tk.Frame(self.canvas)
        self._window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor='nw')

        self.scrollable_frame.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        # Wheel support for both Windows/macOS and X11 Linux.
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.scrollable_frame.bind_all('<Button-4>', self._on_linux_scroll)
        self.scrollable_frame.bind_all('<Button-5>', self._on_linux_scroll)

        # ---- [1] Title + telemetry ---------------------------------------- #
        title = tk.Label(self.scrollable_frame,
                         text='Blind Navigation Recovery - Dashboard',
                         font=('TkDefaultFont', 15, 'bold'))
        title.pack(fill='x', padx=10, pady=(8, 2))

        tele = tk.LabelFrame(self.scrollable_frame, text='Telemetry')
        tele.pack(fill='x', padx=10, pady=4)

        self.lbl_vel = tk.Label(tele, text='Velocity: v = 0.000 m/s, w = 0.000 rad/s',
                                font=('TkDefaultFont', 12, 'bold'))
        self.lbl_vel.pack(anchor='w', **pad)

        self.lbl_pose = tk.Label(tele, text='Position: x = 0.000 m, y = 0.000 m',
                                 font=('TkDefaultFont', 11))
        self.lbl_pose.pack(anchor='w', **pad)

        self.lbl_seg = tk.Label(tele, text='Segment: Straight (0.5 Hz)',
                                font=('TkDefaultFont', 11))
        self.lbl_seg.pack(anchor='w', **pad)

        self.lbl_lm = tk.Label(tele, text='Path landmarks recorded: 0',
                               font=('TkDefaultFont', 11))
        self.lbl_lm.pack(anchor='w', **pad)

        self.lbl_dist = tk.Label(tele, text='Distance: D = 0.000 m, CD = 0.000 m',
                                 font=('TkDefaultFont', 11))
        self.lbl_dist.pack(anchor='w', **pad)

        self.lbl_err = tk.Label(tele, text='Error: not measured yet',
                                font=('TkDefaultFont', 11))
        self.lbl_err.pack(anchor='w', **pad)

        # ---- [2] Auto-run modes + emergency stop -------------------------- #
        modes = tk.LabelFrame(self.scrollable_frame, text='4 Autonomous Modes')
        modes.pack(fill='x', padx=10, pady=4)

        tk.Button(modes, text='Auto 1: Straight + Turn (fixed speed v=0.08 m/s)',
                  command=lambda: self.node.start_auto_run(1),
                  height=1).pack(fill='x', pady=2)
        tk.Button(modes, text='Auto 2: Straight + Turn (adaptive speed 0.05/0.15 m/s)',
                  command=lambda: self.node.start_auto_run(2),
                  height=1).pack(fill='x', pady=2)
        tk.Button(modes, text='Auto 3: Continuous curve (LSV=0.086 / ASV=0.097)',
                  command=lambda: self.node.start_auto_run(3),
                  height=1).pack(fill='x', pady=2)
        tk.Button(modes, text='Auto 4: Continuous curve (profile 0.04-0.086-0.20 m/s)',
                  command=lambda: self.node.start_auto_run(4),
                  height=1).pack(fill='x', pady=2)

        self.btn_strategy = tk.Button(
            modes, text=f'DREAM Strategy: {self.node.strategy}',
            command=self.on_toggle_strategy, height=1,
            bg='#455a64', fg='white')
        self.btn_strategy.pack(fill='x', pady=2)

        tk.Button(modes, text='■ EMERGENCY STOP', command=self.node.emergency_stop,
                  height=2, bg='#c62828', fg='white',
                  font=('TkDefaultFont', 12, 'bold')).pack(fill='x', pady=(4, 2))

        # ---- [2c] Failure simulation (camera cut) ------------------------- #
        actions = tk.LabelFrame(self.scrollable_frame, text='Failure Simulation')
        actions.pack(fill='x', padx=10, pady=4)

        tk.Button(actions, text='CUT CAMERA SIGNAL NOW (return from this pose)',
                  command=self.node.trigger_failure, height=2,
                  bg='#c62828', fg='white',
                  font=('TkDefaultFont', 12, 'bold')).pack(fill='x', pady=3)

        tk.Label(actions, text='Auto cut camera signal at (forward distance):',
                 fg='#455a64').pack(anchor='w', padx=6)
        self.fail_pct_var = tk.StringVar(value='100')
        pct_frame = tk.Frame(actions)
        pct_frame.pack(anchor='w', padx=6, pady=(0, 2))
        for pct in (70, 80, 90, 100):
            tk.Radiobutton(
                pct_frame, text=f'{pct}%', value=str(pct),
                variable=self.fail_pct_var,
                command=lambda p=pct: self.on_fail_percent(p)
            ).pack(side='left', padx=6)
        self.lbl_cutoff = tk.Label(
            actions, text='Camera will auto-cut at 100% of the path.',
            fg='#777777')
        self.lbl_cutoff.pack(anchor='w', padx=6, pady=(0, 2))

        tk.Button(actions, text='Compute Position Error (PE %)',
                  command=self.on_calculate_error, height=1).pack(fill='x', pady=3)

        # ---- [2b] Benchmark experiment (open-loop profile) --------------- #
        bench = tk.LabelFrame(
            self.scrollable_frame,
            text='Benchmark Experiment (open-loop profile)')
        bench.pack(fill='x', padx=10, pady=4)

        self.lbl_world = tk.Label(
            bench, text=f'Loaded world: {self.node.world.rsplit("/", 1)[-1]}',
            fg='#777777')
        self.lbl_world.pack(anchor='w', padx=6, pady=(2, 0))

        tk.Button(bench, text='▶ Run benchmark (auto-cut → return)',
                  command=self.on_run_benchmark, height=1,
                  bg='#37474f', fg='white').pack(fill='x', pady=(4, 2))

        # ---- [3] Manual teleop -------------------------------------------- #
        teleop = tk.LabelFrame(self.scrollable_frame, text='Manual Teleop')
        teleop.pack(fill='x', padx=10, pady=4)

        grid = tk.Frame(teleop)
        grid.pack(pady=4)

        tk.Button(grid, text='▲ Forward', width=10, height=2,
                  command=lambda: self._teleop(+1, 0)).grid(row=0, column=1, padx=4, pady=2)
        tk.Button(grid, text='◀ Left', width=10, height=2,
                  command=lambda: self._teleop(0, +1)).grid(row=1, column=0, padx=4, pady=2)
        tk.Button(grid, text='■ Stop', width=10, height=2, bg='#fdd835',
                  command=self.node.manual_stop).grid(row=1, column=1, padx=4, pady=2)
        tk.Button(grid, text='▶ Right', width=10, height=2,
                  command=lambda: self._teleop(0, -1)).grid(row=1, column=2, padx=4, pady=2)
        tk.Button(grid, text='▼ Backward', width=10, height=2,
                  command=lambda: self._teleop(-1, 0)).grid(row=2, column=1, padx=4, pady=2)

        tk.Label(teleop, text='Keys: ↑↓←→ move, Space stop',
                 fg='#555555').pack(pady=(0, 2))

        # ---- [5] Real-time plots (bottom of the window) ------------------- #
        plot_frame = tk.LabelFrame(self.scrollable_frame,
                                   text='Real-time plots')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        self.fig = Figure(figsize=(6.6, 3.6), dpi=100)
        self.ax_yaw = self.fig.add_subplot(2, 1, 1)
        self.ax_vel = self.fig.add_subplot(2, 1, 2, sharex=self.ax_yaw)
        self.fig.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill='both', expand=True)

        # Keyboard bindings.
        self.root.bind('<Up>', lambda e: self._teleop(+1, 0))
        self.root.bind('<Down>', lambda e: self._teleop(-1, 0))
        self.root.bind('<Left>', lambda e: self._teleop(0, +1))
        self.root.bind('<Right>', lambda e: self._teleop(0, -1))
        self.root.bind('<space>', lambda e: self.node.manual_stop())
        self.root.bind('<KeyRelease>', self._on_key_release)

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

        # Optional auto-start: drive the optimal trajectory without a button
        # press, so the robot moves on its own for unattended demos.
        if getattr(self.node, 'auto_run', False):
            self.root.after(2500, lambda: self.node.start_auto_run(4))

        # Gazebo and RViz map their windows after the dashboard and can cover
        # it (especially without a window manager). Re-raise the dashboard
        # periodically during startup so it stays visible in the foreground.
        self._raise_count = 0
        self._raise_window_periodic()
        self._tick()

    def _raise_window(self):
        """Bring the dashboard window to the foreground once."""
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.update_idletasks()
            self.root.attributes('-topmost', False)
        except tk.TclError:
            pass

    def _raise_window_periodic(self):
        """Re-raise every 2 s for the first ~60 s so late-mapping Gazebo/RViz
        windows do not end up covering the dashboard."""
        self._raise_window()
        self._raise_count += 1
        if self._raise_count < 30:
            self.root.after(2000, self._raise_window_periodic)

    # ------------------------------------------------------------------ #
    # Scroll helpers
    # ------------------------------------------------------------------ #
    def _on_frame_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event):
        delta = -1 * int(event.delta / 120) if event.delta else 0
        self.canvas.yview_scroll(delta, 'units')

    def _on_linux_scroll(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self.canvas.yview_scroll(1, 'units')

    # ------------------------------------------------------------------ #
    # Teleop helpers
    # ------------------------------------------------------------------ #
    def _max_lin(self):
        return MANUAL_SPEED

    def _teleop(self, lin_sign, ang_sign):
        self.node.manual_move(self._max_lin() * lin_sign, 0.2 * ang_sign)

    def _on_key_release(self, _event):  # noqa: ARG002
        # Releasing a key stops manual teleop motion only. Do NOT cancel an
        # active auto run here: <KeyRelease> fires for *every* key, so calling
        # manual_stop() would kill Auto modes 1-4 on any key release.
        self.node.stop_motion()

    # ------------------------------------------------------------------ #
    # GUI event handlers
    # ------------------------------------------------------------------ #
    def on_toggle_strategy(self):
        """Cycle the DREAM sampling strategy (geometry -> velocity -> speed_error)."""
        order = ['geometry', 'velocity', 'speed_error']
        cur = self.node.strategy
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else order[0]
        self.node.strategy = nxt
        self.node.strategy_pub.publish(String(data=nxt))
        self.btn_strategy.config(text=f'DREAM Strategy: {nxt}')

    def on_run_benchmark(self):
        """Drive the open-loop profile of the currently loaded world."""
        key = benchmark_key_for_world(self.node.world)
        if key is None:
            messagebox.showinfo(
                'No benchmark profile',
                'The loaded world has no open-loop benchmark profile. '
                'Use the 4 Autonomous Modes for the corridor demo.')
            return
        self.node.start_benchmark(key)

    def on_fail_percent(self, pct):
        """Apply the chosen automatic camera-cutoff percentage."""
        self.node.set_failure_percent(pct)
        self.lbl_cutoff.config(
            text=f'Camera will auto-cut at {pct}% of the path.')

    def on_calculate_error(self):
        result = self.node.compute_error()
        if result is None:
            messagebox.showinfo('Position error', 'No forward distance (D) data yet.')
            return
        d_forward, d_recovery, pe, rm = result
        text = (f'D (forward)    = {d_forward:.3f} m\n'
                f'CD (return)    = {d_recovery:.3f} m\n'
                f'RM (remaining) = {rm:.3f} m\n'
                f'PE (error %)   = {pe:.3f} %')
        self.lbl_err.config(text=f'Error: PE = {pe:.3f} %, RM = {rm:.3f} m')
        messagebox.showinfo('Position Error (Equation 2)', text)

    def on_close(self):
        self.node.stop_auto_run()
        self.root.destroy()

    # ------------------------------------------------------------------ #
    # Real-time plot update
    # ------------------------------------------------------------------ #
    def _update_plots(self):
        if not self.times:
            return
        t = self.times
        seg = self.node.segment_state()
        freq_text = f'{CURVE_FREQ:.1f} Hz (curved)' if seg == 'curved' \
            else f'{STRAIGHT_FREQ:.1f} Hz (straight)'

        self.ax_yaw.clear()
        self.ax_vel.clear()

        # Highlight the curved regions (sampling switches to 1.0 Hz).
        for ax in (self.ax_yaw, self.ax_vel):
            start = None
            for i, s in enumerate(self.seg_hist):
                if s == 'curved' and start is None:
                    start = i
                elif s != 'curved' and start is not None:
                    ax.axvspan(t[start], t[i - 1], color='orange', alpha=0.15)
                    start = None
            if start is not None:
                ax.axvspan(t[start], t[-1], color='orange', alpha=0.15)

        self.ax_yaw.plot(t, self.yaw_hist, 'b-', linewidth=0.9)
        self.ax_yaw.set_ylabel('Yaw [rad]')
        self.ax_yaw.set_title(f'Yaw angle variation  (sampling {freq_text})', fontsize=9)
        self.ax_yaw.grid(True, alpha=0.3)

        self.ax_vel.plot(t, self.w_hist, 'r-', linewidth=0.9, label='w [rad/s]')
        self.ax_vel.plot(t, self.v_hist, 'g-', linewidth=0.9, label='v [m/s]')
        self.ax_vel.axhline(0.0, color='k', linewidth=0.5)
        self.ax_vel.set_ylabel('v [m/s], w [rad/s]')
        self.ax_vel.set_xlabel('time [s]')
        self.ax_vel.legend(loc='upper right', fontsize=7)
        self.ax_vel.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas_plot.draw_idle()

    # ------------------------------------------------------------------ #
    # Periodic GUI + rclpy spin
    # ------------------------------------------------------------------ #
    def _tick(self):
        # Reschedule FIRST so the GUI/control loop can never die: a plotting
        # or telemetry exception below must not stop the ROS control timer.
        self.root.after(50, self._tick)
        try:
            rclpy.spin_once(self.node, timeout_sec=0)
        except Exception as exc:  # noqa: BLE001
            print(f'[dashboard] spin error: {exc}')

        now = time.time()
        if now - self._last_yaml_read > 1.0:
            self._landmarks = self.node.landmark_count()
            self._last_yaml_read = now

        seg = self.node.segment_state()
        seg_text = ('Curved (1.0 Hz)' if seg == 'curved' else 'Straight (0.5 Hz)')
        self.lbl_vel.config(
            text=f'Velocity: v = {self.node.v:.3f} m/s, w = {self.node.w:.3f} rad/s')
        self.lbl_pose.config(
            text=f'Position: x = {self.node.cur_x:.3f} m, y = {self.node.cur_y:.3f} m')
        self.lbl_seg.config(text=f'Segment: {seg_text}')
        self.lbl_lm.config(text=f'Path landmarks recorded: {self._landmarks}')
        try:
            nominal = self.node.nominal_forward_length()
            target = (self.node.failure_percent / 100.0) * nominal
        except Exception:
            nominal = target = 0.0
        self.lbl_dist.config(
            text=f'Distance: D = {self.node.forward_dist:.3f} m / '
                 f'{nominal:.3f} m (auto-cut at {target:.3f} m), '
                 f'CD = {self.node.recovery_dist:.3f} m')

        # Append a sample (~20 Hz) and refresh the plots.
        self.times.append(now - self._t0)
        self.yaw_hist.append(self.node.cur_yaw)
        self.v_hist.append(self.node.v)
        self.w_hist.append(self.node.w)
        self.seg_hist.append(seg)
        if len(self.times) > _PLOT_MAX_POINTS:
            self.times = self.times[-_PLOT_MAX_POINTS:]
            self.yaw_hist = self.yaw_hist[-_PLOT_MAX_POINTS:]
            self.v_hist = self.v_hist[-_PLOT_MAX_POINTS:]
            self.w_hist = self.w_hist[-_PLOT_MAX_POINTS:]
            self.seg_hist = self.seg_hist[-_PLOT_MAX_POINTS:]
        try:
            self._update_plots()
        except Exception as exc:  # noqa: BLE001
            print(f'[dashboard] plot error: {exc}')

        # Auto-compute RM/PE once the robot stops after back travel.
        if (self.node.state == 'recovery' and not self.node.pe_computed
                and self.node.recovery_dist > 0.05
                and abs(self.node.v) < 0.01 and abs(self.node.w) < 0.01):
            self._stopped_ticks += 1
            if self._stopped_ticks > 40:  # ~2 s of standstill
                result = self.node.compute_error()
                if result is not None:
                    _, _, pe, rm = result
                    self.lbl_err.config(
                        text=f'Error: PE = {pe:.3f} %, RM = {rm:.3f} m')
                    self.node.get_logger().info(
                        f'Auto position error: PE={pe:.3f}% RM={rm:.3f}m')
        else:
            self._stopped_ticks = 0


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        DashboardApp(node)
        node.get_logger().info('Dashboard GUI started.')
        import tkinter
        tkinter.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
