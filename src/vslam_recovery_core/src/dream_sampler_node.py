#!/usr/bin/env python3
"""
dream_sampler_node.py

Event-triggered odometry sampling via the Dynamic Regression Algorithm (DREAM)
for the blind mobile-robot recovery layer (ROS 2).

This node is the DREAM counterpart of the fixed-rate Shannon-Nyquist sampler
(landmark_sampler_client). Instead of toggling between two hand-tuned rates
(0.5 Hz straight / 1.0 Hz curved) from a yaw-difference heuristic, it drives
the recording decision from the goodness-of-fit of local linear models.

Algorithm (per dream_core.py):

  1. Buffer the odometry stream (t, x, y, s) where (x, y) is the recorded
     position and s the cumulative travelled distance.
  2. Over a growing window, evaluate the path linearity (the principal-axis
     fraction of the (x, y) covariance) and the speed linearity (the R^2 of
     s(t)); the combined goodness is G = min(path linearity, speed R^2).
  3. Linear regime  (G >= R^2_th): steady straight-line or uniform motion ->
     no landmark is logged (middleware/storage suppressed).
  4. Transition event (G <  R^2_th): a turn or a speed change -> flush the
     boundary landmark M = {v, w, d, T, psi, x, y} through the existing
     RecordLandmark service and start a fresh window.

The result is an adaptive sampling-rate controller: sparse landmarks on
straight, steady segments and dense landmarks only where the trajectory
actually changes. The single tunable parameter is the threshold R^2_th.

The recorded landmarks are consumed unchanged by the existing
recovery_recorder_server (service) and recovery_controller (reverse replay),
so this node can be swapped in place of landmark_sampler_client without any
change to the rest of the stack.
"""

import math
import os
import sys

# Allow importing the co-located pure-Python core after colcon installation
# (scripts are installed into the same lib/<pkg> directory).
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from recovery_interfaces.srv import RecordLandmark

import dream_core


class DreamSamplerNode(Node):
    """DREAM event-triggered landmark sampler (thin ROS 2 wrapper)."""

    def __init__(self):
        super().__init__("dream_sampler_node")

        self.odom_topic = self.declare_parameter("odom_topic", "/odom").value
        self.service_name = self.declare_parameter("service_name", "record_landmark").value
        r2_threshold = self.declare_parameter("r2_threshold", 0.98).value
        min_window_size = self.declare_parameter("min_window_size", 5).value
        max_window_size = self.declare_parameter("max_window_size", 500).value
        regress_speed = self.declare_parameter("regress_speed", True).value
        min_event_spacing = self.declare_parameter("min_event_spacing", 0.2).value
        max_window_arc = self.declare_parameter("max_window_arc", 0.0).value

        self.sampler = dream_core.DreamSampler(
            r2_threshold=r2_threshold,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            regress_speed=regress_speed,
            min_event_spacing=min_event_spacing,
            max_window_arc=(max_window_arc if max_window_arc > 0.0 else None))

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.record_cli = self.create_client(RecordLandmark, self.service_name)

        # Odometry state for the (psi, s) stream.
        self._unwrapped_psi = 0.0
        self._have_psi = False
        self._cum_s = 0.0
        self._last_x = None
        self._last_y = None
        self._last_event_stamp = None

        self.get_logger().info(
            f"DreamSamplerNode ready. odom='{self.odom_topic}' "
            f"service='{self.service_name}' R2_th={r2_threshold:.3f} "
            f"min_window={min_window_size} max_window={max_window_size} "
            f"max_window_arc={max_window_arc} "
            f"regress_speed={regress_speed}")

    # ------------------------------------------------------------------ #
    # Odometry callback
    # ------------------------------------------------------------------ #
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if not all(math.isfinite(v) for v in (x, y)):
            self.get_logger().warn(
                "Non-finite position in /odom; ignoring frame.",
                throttle_duration_sec=5.0)
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Unwrapped heading (prevents a spurious +/-pi wrap transition).
        q = msg.pose.pose.orientation
        psi = dream_core.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        if not self._have_psi:
            self._unwrapped_psi = psi
            self._have_psi = True
        else:
            self._unwrapped_psi = dream_core.unwrap_delta(self._unwrapped_psi, psi)

        # Cumulative travelled distance.
        if self._last_x is not None:
            self._cum_s += math.hypot(x - self._last_x, y - self._last_y)
        self._last_x = x
        self._last_y = y

        if self.sampler.update(stamp, x, y, self._cum_s):
            # A transition (or the start landmark) occurred: flush a landmark.
            dt = stamp - self._last_event_stamp if self._last_event_stamp is not None else 1.0
            frequency = 1.0 / dt if dt > 1e-3 else 1.0
            self._last_event_stamp = stamp
            self._record_landmark(frequency)

    # ------------------------------------------------------------------ #
    # Landmark recording (reuses the existing RecordLandmark service)
    # ------------------------------------------------------------------ #
    def _record_landmark(self, frequency):
        if not self.record_cli.service_is_ready():
            self.get_logger().warn(
                f"Service '{self.service_name}' not ready; skipping landmark.",
                throttle_duration_sec=5.0)
            return

        request = RecordLandmark.Request()
        request.theta = self._unwrapped_psi
        request.frequency = frequency

        n = self.sampler.n_events
        future = self.record_cli.call_async(request)
        future.add_done_callback(lambda f: self._record_done(f, n))

    def _record_done(self, future, n):
        try:
            response = future.result()
            if response is not None and response.ret:
                self.get_logger().info(
                    f"Landmark #{n} recorded (DREAM transition, "
                    f"R2_th={self.sampler.r2_threshold:.3f})")
            else:
                self.get_logger().warn("Landmark recording request failed.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Landmark service call raised: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = DreamSamplerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
