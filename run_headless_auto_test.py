#!/usr/bin/env python3
"""
run_headless_auto_test.py

Headless Gazebo + ROS 2 end-to-end test of the four dashboard auto modes with
a simulated camera failure at 100% of the forward distance.

The script starts the real stack — Gazebo server (sensor-free headless world),
ros_gz_bridge, odom_sanitizer, recovery_recorder_server, dream_sampler_node and
recovery_controller — plus a headless auto-driver that reuses the exact
dashboard control methods (`recovery_dashboard_gui`).

For every mode it measures the final return error and checks that the robot
never collides with the corridor walls or the two obstacles. A mode passes when
the robot returns within 0.15 m and the safety margins stay positive.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 run_headless_auto_test.py             # all modes 1..4
  python3 run_headless_auto_test.py --mode 4    # only Auto 4
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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

WS = os.path.dirname(os.path.realpath(__file__))
SRC = os.path.join(WS, "src", "vslam_recovery_core", "src")
sys.path.insert(0, SRC)

import optimal_trajectory as ot               # noqa: E402
import recovery_dashboard_gui as dash         # noqa: E402

WORLD_SRC = os.path.join(WS, "src", "vslam_recovery_core", "worlds",
                         "recovery_corridor.sdf")
OBSTACLES = [(1.5, 0.3), (3.0, -0.3)]
OBST_R = 0.20
WALL_Y = 0.85
ROBOT_HW = 0.15
ROBOT_HD = 0.10
DT = 1.0 / 20.0
PASS_ERROR = 0.15


def norm(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_headless_world(dst):
    src = open(WORLD_SRC).read()
    src = re.sub(r'\s*<plugin filename="ignition-gazebo-sensors-system".*?</plugin>\n',
                 '\n', src, flags=re.S)
    # Strip the camera / depth / IMU sensors only; keep the contact (bumper)
    # sensor so the physics contact events are still published.
    src = re.sub(r'\s*<sensor[^>]*type="(camera|depth_camera|imu)"[^>]*>.*?</sensor>\n',
                 '\n', src, flags=re.S)
    src = re.sub(r'\s*<plugin filename="ignition-gazebo-imu-system".*?</plugin>\n',
                 '\n', src, flags=re.S)
    with open(dst, "w") as fh:
        fh.write(src)


def write_bridge_config(dst):
    with open(dst, "w") as fh:
        fh.write(
            '- ros_topic_name: "/cmd_vel"\n'
            '  gz_topic_name: "/model/recovery_robot/cmd_vel"\n'
            '  ros_type_name: "geometry_msgs/msg/Twist"\n'
            '  gz_type_name: "gz.msgs.Twist"\n'
            '  direction: BIDIRECTIONAL\n'
            '- ros_topic_name: "/odom_raw"\n'
            '  gz_topic_name: "/model/recovery_robot/odometry_cov"\n'
            '  ros_type_name: "nav_msgs/msg/Odometry"\n'
            '  gz_type_name: "gz.msgs.OdometryWithCovariance"\n'
            '  direction: GZ_TO_ROS\n')


def start(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)


def kill(proc):
    """Terminate a launched process and its whole process group.

    Processes are started with start_new_session=True, so proc.pid is the
    group leader. The ``ros2 run`` wrapper spawns the actual node as a child
    in the same group; we signal the GROUP (not just the wrapper) and always
    escalate to SIGKILL -- even when the wrapper has already exited -- so the
    child node cannot be orphaned. An orphaned node would keep publishing to
    /odom, /cmd_vel, etc. and pollute the next run.
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


class AutoDriver(Node):
    """Headless replacement for the dashboard's closed-loop auto navigation."""

    def __init__(self, mode):
        super().__init__("headless_auto_driver")
        self.mode = mode
        self.auto_mode = mode
        self.world = "recovery_corridor.sdf"
        ot.configure(self.world)
        self.auto_waypoints = ot.waypoints(self.world)
        self.auto_wp_idx = 0
        self.auto_phase = "turn"
        self.auto_active = False
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_yaw = 0.0
        self.forward_dist = 0.0
        self.last_pos = None
        self.triggered = False
        self.trigger_t = None
        self.completed_t = None
        self.log = []  # (wall_t, x, y, yaw, v_cmd, w_cmd) at 20 Hz

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.trigger_pub = self.create_publisher(Bool, "/camera_failure_trigger", 10)
        self.timer = self.create_timer(DT, self.tick)

    def stop_auto_run(self):
        self.auto_active = False
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0
        self.cmd_pub.publish(Twist())

    def start_auto_run(self):
        self.auto_waypoints = ot.waypoints(self.world)
        self.auto_wp_idx = 0
        self.auto_phase = "turn"
        self.auto_active = True
        self.auto_last_v = 0.0
        self.auto_last_w = 0.0

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            return
        self.cur_x = p.x
        self.cur_y = p.y
        q = msg.pose.pose.orientation
        sy = 2.0 * (q.w * q.z + q.x * q.y)
        cy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.cur_yaw = math.atan2(sy, cy)
        if self.last_pos is not None:
            self.forward_dist += math.hypot(p.x - self.last_pos[0],
                                            p.y - self.last_pos[1])
        self.last_pos = (p.x, p.y)

    def tick(self):
        if self.triggered or not self.auto_active:
            return
        if self.auto_mode in (1, 2):
            res = dash.DashboardNode._control_waypoint(self)
        else:
            res = dash.DashboardNode._control_trajectory(self)
        if res is None or res[0] is None:
            # Forward path complete (100% of the nominal distance): fail the
            # camera now and hand control to the recovery controller.
            self.trigger()
            return
        v_des, w_des = res
        v = dash.slew(self.auto_last_v, v_des, dash.MAX_LIN_ACCEL * DT)
        w = dash.slew(self.auto_last_w, w_des, dash.MAX_ANG_ACCEL * DT)
        w = clamp(w, -dash.MAX_ANGULAR_VELOCITY, dash.MAX_ANGULAR_VELOCITY)
        v = max(0.0, v)
        self.auto_last_v = v
        self.auto_last_w = w
        tw = Twist()
        tw.linear.x = v
        tw.angular.z = w
        self.cmd_pub.publish(tw)

    def trigger(self):
        if self.triggered:
            return
        self.triggered = True
        self.trigger_t = time.time()
        self.stop_auto_run()
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)


def nominal_length(mode):
    if mode in (1, 2):
        wps = ot.waypoints("recovery_corridor.sdf")
        return sum(math.hypot(wps[i + 1][0] - wps[i][0],
                              wps[i + 1][1] - wps[i][1])
                   for i in range(len(wps) - 1))
    return ot._SAMPLED[2][-1]


def robot_corners(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    pts = []
    for sx in (-ROBOT_HW, ROBOT_HW):
        for sy in (-ROBOT_HD, ROBOT_HD):
            pts.append((x + sx * c - sy * s, y + sx * s + sy * c))
    return pts


def dist_point_rect(px, py, cx, cy, yaw):
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


def run_mode(mode, timeout=600, sigma_v=0.0, sigma_w=0.0, yaw_bias_rate=0.0):
    tmp = tempfile.mkdtemp(prefix="dream_auto_")
    world_hd = os.path.join(tmp, "world.sdf")
    bridge_cfg = os.path.join(tmp, "bridge.yaml")
    yaml_file = os.path.join(tmp, "path_landmarks.yaml")
    make_headless_world(world_hd)
    write_bridge_config(bridge_cfg)

    gz = start(["ign", "gazebo", "-s", "-r", world_hd])
    time.sleep(8.0)
    # Stream the physics contact (bumper) topic to a log so any engine-level
    # collision is recorded (not just the kinematic footprint check).
    contact_topic = ("/world/recovery_corridor/model/recovery_robot/"
                     "link/chassis/sensor/bumper/contact")
    contact_log = os.path.join(tmp, "contact.log")
    contact_proc = start(["sh", "-c",
                          f"ign topic -e -t {contact_topic} > {contact_log} 2>&1"])
    # Stream the world-frame dynamic pose so the true (physics) endpoint can be
    # reported separately from the odometry-frame endpoint.
    gt_topic = "/world/recovery_corridor/dynamic_pose/info"
    gt_log = os.path.join(tmp, "gt.log")
    gt_proc = start(["sh", "-c",
                     f"ign topic -e -t {gt_topic} > {gt_log} 2>&1"])
    bridge = start(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                    "--ros-args", "-p", f"config_file:={bridge_cfg}"])
    time.sleep(3.0)
    sanitizer = start(["ros2", "run", "vslam_recovery_core", "odom_sanitizer.py",
                      "--ros-args", "-p", f"sigma_v:={sigma_v}",
                      "-p", f"sigma_w:={sigma_w}",
                      "-p", f"yaw_bias_rate:={yaw_bias_rate}"])
    recorder = start(["ros2", "run", "vslam_recovery_core", "recovery_recorder_server",
                      "--ros-args", "-p", f"yaml_file:={yaml_file}",
                      "-p", "odom_topic:=/odom"])
    sampler = start(["ros2", "run", "vslam_recovery_core", "dream_sampler_node.py",
                     "--ros-args", "-p", "odom_topic:=/odom",
                     "-p", "r2_threshold:=0.98",
                     "-p", "max_window_size:=5000",
                     "-p", "max_window_arc:=1.0"])
    controller = start(["ros2", "run", "vslam_recovery_core", "recovery_controller",
                        "--ros-args", "-p", f"yaml_file:={yaml_file}",
                        "-p", "odom_topic:=/odom",
                        "-p", "cmd_vel_topic:=/cmd_vel",
                        "-p", "failure_trigger_topic:=/camera_failure_trigger",
                        "-p", "rotate_angular_velocity:=0.8",
                        "-p", "drive_frequency:=20.0"])
    time.sleep(3.0)

    driver = AutoDriver(mode)
    driver.start_auto_run()

    trigger_dist = 1.0 * nominal_length(mode)
    min_wall = 1e9
    min_obst = 1e9
    coll = None
    start_t = time.time()
    t_trigger = None
    pose_log = []  # (elapsed_s, x, y, yaw) at each spin iteration
    completed = False

    try:
        while time.time() - start_t < timeout:
            rclpy.spin_once(driver, timeout_sec=0.01)
            x, y, yaw = driver.cur_x, driver.cur_y, driver.cur_yaw
            pose_log.append((time.time() - start_t, x, y, yaw))
            for (cx, cy) in robot_corners(x, y, yaw):
                min_wall = min(min_wall, WALL_Y - abs(cy))
            for (ox, oy) in OBSTACLES:
                min_obst = min(min_obst, dist_point_rect(ox, oy, x, y, yaw) - OBST_R)

            if not driver.triggered and driver.forward_dist >= trigger_dist:
                driver.trigger()

            if driver.triggered and t_trigger is None:
                t_trigger = time.time()

            if driver.triggered and t_trigger is not None:
                if time.time() - t_trigger > 2.0 and math.hypot(x, y) < 0.06:
                    completed = True
                    break
                if time.time() - t_trigger > 120.0:
                    break

        time.sleep(0.2)
        rclpy.spin_once(driver, timeout_sec=0.2)
        final_x, final_y = driver.cur_x, driver.cur_y
        final_err = math.hypot(final_x, final_y)
    finally:
        # These can raise (e.g. RCLError when the context was already shut
        # down by an interrupted run); never let that skip the cleanup below.
        try:
            driver.stop_auto_run()
        except Exception:
            pass
        try:
            driver.destroy_node()
        except Exception:
            pass
        kill(controller)
        kill(sampler)
        kill(recorder)
        kill(sanitizer)
        kill(bridge)
        kill(contact_proc)
        kill(gt_proc)
        kill(gz)

    if min_wall < 0.0:
        coll = "wall"
    elif min_obst < 0.0:
        coll = "obstacle"
    ok = (final_err < PASS_ERROR and coll is None
          and min_wall >= 0.0 and min_obst >= 0.0)

    # Persist the ground-truth trajectory (odom_sanitizer noise is off in the
    # demo, so /odom equals the Gazebo odometry = physics ground truth) and the
    # event timestamps for reproducibility.
    csv_path = os.path.join(WS, "bench_data", f"corridor_run_{mode}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["elapsed_s", "x", "y", "yaw"])
        w.writerows(pose_log)

    # Count physics contact events (engine-level collision blocks) recorded by
    # the bumper contact sensor. The chassis always touches the ground plane,
    # so only contacts whose counterpart is a wall or an obstacle count as a
    # collision. The message is `contact { collision1 { name: ... } collision2
    # { name: ... } ... }`.
    n_contacts = -1
    try:
        with open(contact_log) as fh:
            txt = fh.read()
        blocks = txt.split("contact {")
        n_contacts = sum(1 for b in blocks[1:]
                         if "wall" in b or "obstacle" in b)
    except Exception:
        pass

    # Parse the world-frame (physics) endpoint from the dynamic-pose log: the
    # last `pose { name: "recovery_robot" ... position { x: ... y: ... } }`.
    gt_x = gt_y = None
    try:
        import re
        txt = open(gt_log).read()
        m = re.findall(
            r'pose \{[^}]*name: "recovery_robot"[^}]*position \{\s*x: ([^\s]+)\s+y: ([^\s]+)',
            txt)
        if m:
            gt_x, gt_y = float(m[-1][0]), float(m[-1][1])
    except Exception:
        pass
    gt_err = math.hypot(gt_x, gt_y) if gt_x is not None else float("nan")
    return dict(mode=mode, ok=ok, final_err=final_err, final_x=final_x,
                final_y=final_y, min_wall=min_wall, min_obst=min_obst,
                coll=coll, completed=completed,
                trigger_t=t_trigger - start_t if t_trigger else None,
                csv=csv_path, n_contacts=n_contacts, gt_err=gt_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=0, help="1-4; 0 = all modes")
    ap.add_argument("--sigma-v", type=float, default=0.0,
                    help="odom_sanitizer linear-velocity noise std (m/s)")
    ap.add_argument("--sigma-w", type=float, default=0.0,
                    help="odom_sanitizer angular-velocity noise std (rad/s)")
    ap.add_argument("--yaw-bias-rate", type=float, default=0.0,
                    help="odom_sanitizer heading random-walk bias (rad/s/sqrt s)")
    ap.add_argument("--runs", type=int, default=1,
                    help="number of runs per mode for a distribution")
    args = ap.parse_args()
    modes = [args.mode] if args.mode else [1, 2, 3, 4]

    rclpy.init(args=["run_headless_auto_test"])
    try:
        print("=" * 78)
        print("Headless Gazebo auto-mode recovery test "
              f"(noise sigma_v={args.sigma_v}, sigma_w={args.sigma_w}, "
              f"yaw_bias_rate={args.yaw_bias_rate}, runs={args.runs})")
        print("=" * 78)
        all_ok = True
        for m in modes:
            ferrs = []
            contacts = []
            for r_i in range(args.runs):
                r = run_mode(m, sigma_v=args.sigma_v, sigma_w=args.sigma_w,
                             yaw_bias_rate=args.yaw_bias_rate)
                if not r["ok"]:
                    all_ok = False
                ferrs.append(r["final_err"])
                contacts.append(r["n_contacts"])
                status = "PASS" if r["ok"] else "FAIL"
                print(f"Mode {m} run {r_i}: {status}  "
                      f"final_err(odom)={r['final_err']:.5f} m  "
                      f"gt_err(world)={r['gt_err']:.5f} m  "
                      f"completed={r['completed']}  "
                      f"wall={r['min_wall']:.3f} m  obst={r['min_obst']:.3f} m  "
                      f"coll={r['coll']}  contacts={r['n_contacts']}",
                      flush=True)
            if args.runs > 1 and ferrs:
                print(f"Mode {m}: final_err min/mean/max = "
                      f"{min(ferrs):.5f} / {sum(ferrs)/len(ferrs):.5f} / "
                      f"{max(ferrs):.5f} m, contacts={contacts}")
        print("-" * 78)
        print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    finally:
        # An interrupted run (SIGINT/SIGTERM) may have already shut the
        # context down via rclpy's signal handler; guard the duplicate call.
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
