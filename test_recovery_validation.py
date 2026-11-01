#!/usr/bin/env python3
"""
test_recovery_validation.py

Independent QA / peer-review testbench for the blind-navigation recovery layer
described in:

  "An approach towards mobile robot recovery due to vision sensor failure in
   vSLAM systems using ROS" - Mulubika & Schreve, Robotica (2025).

The script launches the three recovery nodes, drives a simulated /odom
trajectory (one straight segment followed by one curved segment), verifies the
recorded path_landmarks.yaml, triggers the camera-failure recovery, records the
/cmd_vel commands emitted during back travel and checks the sampled time-based
motion-control logic against the paper.

Run it after sourcing the workspace overlay:

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 test_recovery_validation.py
"""

import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Configuration (values follow the paper and the default node parameters)
# ---------------------------------------------------------------------------
STRAIGHT_V = 0.1      # m/s, straight segment
STRAIGHT_W = 0.0      # rad/s
STRAIGHT_DUR = 6.0    # s

CURVE_V = 0.08        # m/s, curved segment
CURVE_W = 0.1         # rad/s
CURVE_DUR = 4.0       # s

ODOM_RATE = 50.0      # Hz
ROTATE_W = 0.8        # rad/s (controller default rotate_angular_velocity)
DRIVE_FREQ = 20.0     # Hz (controller default drive_frequency)
T_STRAIGHT = 2.0      # s (f = 0.5 Hz)
T_CURVE = 1.0         # s (f = 1.0 Hz)
MAX_YAML_BYTES = 16 * 1024  # 16 KB per Table VI

# ---------------------------------------------------------------------------
# ROS 2 imports (require the overlay to be sourced)
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool
    from recovery_interfaces.srv import RecordLandmark
except ImportError as exc:  # pragma: no cover
    print(f"[FATAL] Could not import rclpy / interfaces: {exc}")
    print("        Source the workspace overlay first:")
    print("        source /opt/ros/humble/setup.bash && source install/setup.bash")
    sys.exit(2)


def spin_for(node, seconds):
    """Spin the node for a wall-clock duration (no busy sleep)."""
    end = time.time() + seconds
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)


class ValidationNode(Node):
    """Publishes a fake /odom trajectory, triggers recovery and records /cmd_vel.

    After the scripted forward trajectory the node switches to a
    cmd_vel-driven diff-drive model so the recovery controller's closed-loop
    feedback (yaw from /odom) has a real plant to drive.
    """

    def __init__(self):
        super().__init__("test_recovery_validation")
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.trigger_pub = self.create_publisher(Bool, "/camera_failure_trigger", 10)
        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 100)
        self.record_cli = self.create_client(RecordLandmark, "record_landmark")

        self.traj_time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.final_pose = None
        self.odom_timer = None
        self.cmd_follow_timer = None
        self.cmd_samples = []  # list of (wall_time, linear.x, angular.z)
        self.last_cmd = Twist()

    def _cmd_cb(self, msg):
        self.last_cmd = msg
        self.cmd_samples.append((time.time(), msg.linear.x, msg.angular.z))

    def publish_trigger(self):
        msg = Bool()
        msg.data = True
        self.trigger_pub.publish(msg)

    def _publish_odom(self, v, w):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w
        self.odom_pub.publish(msg)

    def start_trajectory(self):
        self.odom_timer = self.create_timer(1.0 / ODOM_RATE, self._traj_cb)

    def _traj_cb(self):
        dt = 1.0 / ODOM_RATE
        t = self.traj_time
        if t < STRAIGHT_DUR:
            v, w = STRAIGHT_V, STRAIGHT_W
        else:
            v, w = CURVE_V, CURVE_W

        # Differential-drive integration of the ground-truth trajectory.
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += w * dt
        self.traj_time += dt
        self._publish_odom(v, w)

        if self.traj_time >= STRAIGHT_DUR + CURVE_DUR:
            self.final_pose = (self.x, self.y, self.yaw)
            if self.odom_timer is not None:
                self.odom_timer.cancel()
                self.odom_timer = None

    def enable_cmd_follow(self):
        """Switch the fake odom to a cmd_vel-driven diff-drive plant."""
        if self.odom_timer is not None:
            self.odom_timer.cancel()
            self.odom_timer = None
        if self.cmd_follow_timer is None:
            self.cmd_follow_timer = self.create_timer(1.0 / ODOM_RATE, self._cmd_follow_cb)

    def _cmd_follow_cb(self):
        dt = 1.0 / ODOM_RATE
        v = self.last_cmd.linear.x
        w = self.last_cmd.angular.z
        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt
        self.yaw += w * dt
        self._publish_odom(v, w)


# ---------------------------------------------------------------------------
# Node process management
# ---------------------------------------------------------------------------
def launch_nodes(yaml_path):
    """Launch the three recovery nodes as subprocesses."""
    procs = []
    base = ["ros2", "run", "vslam_recovery_core"]
    commands = [
        ["recovery_recorder_server", "--ros-args",
         "-p", f"yaml_file:={yaml_path}",
         "-p", "odom_topic:=/odom",
         "-p", "service_name:=record_landmark"],
        ["landmark_sampler_client", "--ros-args",
         "-p", "odom_topic:=/odom",
         "-p", "service_name:=record_landmark",
         "-p", "straight_frequency:=0.5",
         "-p", "curve_frequency:=1.0",
         "-p", "monitor_period:=0.1",
         "-p", "window_size:=4"],
        ["recovery_controller", "--ros-args",
         "-p", f"yaml_file:={yaml_path}",
         "-p", "odom_topic:=/odom",
         "-p", "cmd_vel_topic:=/cmd_vel",
         "-p", "failure_trigger_topic:=/camera_failure_trigger",
         "-p", "straight_period:=2.0",
         "-p", "curve_period:=1.0",
         "-p", "rotate_angular_velocity:=0.8",
         "-p", "drive_frequency:=20.0"],
    ]
    for cmd in commands:
        procs.append(subprocess.Popen(
            base + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True))
    return procs


def stop_nodes(procs):
    # `ros2 run` may spawn the actual node as a child; kill the whole process
    # group so no orphaned recorder/sampler/controller keeps running and
    # pollutes a later session (a real observed failure mode).
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    time.sleep(1.0)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# YAML parsing (no external dependency; format is `landmark_XXXX: [v,w,d,T,psi]`)
# ---------------------------------------------------------------------------
_LM_RE = re.compile(r"^landmark_(\d+):\s*\[([^\]]*)\]")


def parse_yaml(path):
    landmarks = []
    with open(path, "r") as fh:
        for line in fh:
            m = _LM_RE.match(line.strip())
            if not m:
                continue
            values = [float(x.strip()) for x in m.group(2).split(",") if x.strip() != ""]
            if len(values) >= 5:
                landmarks.append({
                    "index": int(m.group(1)),
                    "v": values[0], "w": values[1], "d": values[2],
                    "T": values[3], "psi": values[4],
                })
    landmarks.sort(key=lambda lm: lm["index"])
    return landmarks


def close(value, target, tol):
    return abs(value - target) <= tol


# ---------------------------------------------------------------------------
# cmd_vel analysis
# ---------------------------------------------------------------------------
def analyze_cmd_vel(samples):
    """Verify the closed-loop recovery: pi rotation then forward return drive."""
    rotation = [(t, v, w) for (t, v, w) in samples if abs(v) < 1e-6 and abs(w) > 0.3]
    drive = [(t, v, w) for (t, v, w) in samples if abs(v) > 1e-6]

    # Integrate the in-place rotation angular velocity to estimate the turned
    # angle (the closed-loop turn ramps its rate down near the target).
    rot_angle = 0.0
    for i, (_t, _v, w) in enumerate(rotation):
        dt = 1.0 / DRIVE_FREQ
        if i + 1 < len(rotation):
            dt = rotation[i + 1][0] - _t
        dt = min(max(dt, 0.0), 0.5)
        rot_angle += abs(w) * dt

    seg_set = sorted({(round(v, 3), round(w, 3)) for (_t, v, w) in samples})
    return {
        "rotation_detected": len(rotation) > 5,
        "rotation_angle": rot_angle,
        "rotation_angle_ok": close(rot_angle, math.pi, 0.35),
        "reverse_drive_detected": len(drive) > 5,
        "segments": seg_set,
    }


def integrate_back_pose(final_pose, samples):
    """Integrate the recorded /cmd_vel commands to estimate the return position."""
    x, y, yaw = final_pose
    dt_default = 1.0 / DRIVE_FREQ
    for i, (t, v, w) in enumerate(samples):
        if i + 1 < len(samples):
            dt = samples[i + 1][0] - t
        else:
            dt = dt_default
        dt = min(max(dt, 0.0), 0.5)
        yaw += w * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
    return x, y, yaw


# ---------------------------------------------------------------------------
# Main test sequence
# ---------------------------------------------------------------------------
def main():
    if "ROS_DISTRO" not in os.environ:
        print("[FATAL] ROS 2 environment not sourced.")
        return 2

    results = []
    tmpdir = tempfile.mkdtemp(prefix="recovery_test_")
    yaml_path = os.path.join(tmpdir, "path_landmarks.yaml")

    print("=" * 72)
    print("Blind navigation recovery - automated validation testbench")
    print("=" * 72)
    print(f"temp workspace : {tmpdir}")

    procs = launch_nodes(yaml_path)
    rclpy.init(args=sys.argv)
    node = ValidationNode()

    try:
        # Step 0: wait for the recorder service to register.
        print("\n[STEP A.0] Waiting for /record_landmark service ...")
        ready = node.record_cli.wait_for_service(timeout_sec=20.0)
        if not ready:
            print("FAIL: service /record_landmark did not become ready.")
            return 1
        print("OK: service /record_landmark is ready.")

        # Step A: publish the simulated /odom trajectory.
        print("\n[STEP A] Publishing /odom trajectory: "
              f"straight ({STRAIGHT_V} m/s, {STRAIGHT_DUR}s) + "
              f"curved ({CURVE_V} m/s, {CURVE_W} rad/s, {CURVE_DUR}s)")
        node.start_trajectory()
        spin_for(node, STRAIGHT_DUR + CURVE_DUR + 1.0)
        print(f"OK: trajectory finished. final forward pose = "
              f"({node.final_pose[0]:.3f}, {node.final_pose[1]:.3f}, yaw {node.final_pose[2]:.3f})")

        # Let the last asynchronous service request land and write the YAML.
        time.sleep(3.0)

        # Step B: inspect the recorded landmarks.
        print("\n[STEP B] Inspecting path_landmarks.yaml")
        if not os.path.exists(yaml_path):
            print("FAIL: YAML file was not created.")
            return 1
        file_size = os.path.getsize(yaml_path)
        landmarks = parse_yaml(yaml_path)
        n_landmarks = len(landmarks)
        straight_ts = [lm for lm in landmarks if close(lm["T"], T_STRAIGHT, 0.05)]
        curved_ts = [lm for lm in landmarks if close(lm["T"], T_CURVE, 0.05)]

        size_ok = file_size < MAX_YAML_BYTES
        count_ok = n_landmarks >= 2
        t_straight_ok = len(straight_ts) >= 1
        t_curved_ok = len(curved_ts) >= 1
        t_valid_ok = all(close(lm["T"], T_STRAIGHT, 0.05) or close(lm["T"], T_CURVE, 0.05)
                         for lm in landmarks)

        print(f"  landmark count        : {n_landmarks}")
        print(f"  file size             : {file_size} bytes (< {MAX_YAML_BYTES} bytes)")
        print(f"  straight T samples    : {len(straight_ts)} (T={T_STRAIGHT}s)")
        print(f"  curved T samples      : {len(curved_ts)} (T={T_CURVE}s)")
        for lm in landmarks:
            seg = "straight" if close(lm["T"], T_STRAIGHT, 0.05) else "curved"
            print(f"    landmark_{lm['index']:04d}: v={lm['v']:.3f} w={lm['w']:.3f} "
                  f"d={lm['d']:.3f} T={lm['T']:.3f} psi={lm['psi']:.3f} [{seg}]")

        # Step C/D: trigger recovery and record /cmd_vel.
        print("\n[STEP C] Triggering /camera_failure_trigger and recording /cmd_vel")
        node.cmd_samples.clear()
        node.enable_cmd_follow()
        for _ in range(3):
            node.publish_trigger()
            spin_for(node, 0.2)

        rotation_time = math.pi / ROTATE_W
        back_time = sum(lm["T"] for lm in landmarks)
        spin_for(node, rotation_time + back_time + 6.0)
        print(f"  recorded {len(node.cmd_samples)} /cmd_vel samples")

        analysis = analyze_cmd_vel(node.cmd_samples)
        print("\n[STEP D] closed-loop recovery verification")
        print(f"  rotation detected     : {analysis['rotation_detected']}")
        print(f"  rotated angle         : {analysis['rotation_angle']:.3f} rad (~pi)")
        print(f"  rotation angle ~= pi  : {analysis['rotation_angle_ok']}")
        print(f"  reverse drive detected: {analysis['reverse_drive_detected']}")
        print(f"  segment (v,w) set     : {analysis['segments']}")

        # Theoretical position error, equation (4) / Table IV semantics.
        d_forward = STRAIGHT_V * STRAIGHT_DUR + CURVE_V * CURVE_DUR
        d_back = sum(lm["v"] * lm["T"] for lm in landmarks)
        pe = abs(d_forward - d_back) / d_forward * 100.0

        # The fake odom is now cmd_vel-driven; its final pose measures the
        # actual return accuracy (should be back near the origin).
        return_error = math.hypot(node.x, node.y)
        back_pose = (node.x, node.y, node.yaw)

        print("\n[THEORETICAL POSITION ERROR - eq. (4)]")
        print(f"  ground-truth forward distance D = {d_forward:.3f} m")
        print(f"  commanded back-travel distance CD = {d_back:.3f} m")
        print(f"  PE = |D - CD| / D * 100 = {pe:.3f} %")
        print(f"  measured return error = {return_error:.3f} m")

        # Aggregate verdicts.
        checks = {
            "A_odom_trajectory": node.final_pose is not None,
            "B_yaml_created": os.path.exists(yaml_path),
            "B_count": count_ok,
            "B_size_lt_16KB": size_ok,
            "B_T_straight": t_straight_ok,
            "B_T_curved": t_curved_ok,
            "B_T_valid": t_valid_ok,
            "D_rotation_pi": analysis["rotation_detected"] and analysis["rotation_angle_ok"],
            "D_reverse_drive": analysis["reverse_drive_detected"],
            "D_return_near_origin": return_error < 0.35,
        }
        results.append(("n_landmarks", n_landmarks))
        results.append(("file_bytes", file_size))
        results.append(("pe_percent", pe))
        results.append(("return_error_m", return_error))

        all_pass = all(checks.values())
        print("\n" + "=" * 72)
        print("VERDICT")
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        print("=" * 72)
        print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")

        return 0 if all_pass else 1

    finally:
        stop_nodes(procs)
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
