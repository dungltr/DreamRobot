#!/usr/bin/env python3
"""
odom_sanitizer.py

Republishes the raw odometry coming from the Gazebo bridge under a clean topic.

  * subscribes to /odom_raw (nav_msgs/msg/Odometry, from ros_gz_bridge);
  * normalizes the pose quaternion (NaN / zero-norm -> identity, otherwise
    divide each component by the norm);
  * zeroes any NaN position / twist components;
  * republishes nav_msgs/msg/Odometry on /odom with a clean, standard frame
    pair ('odom' -> 'base_link') and the original covariance preserved.

This removes the RViz 2 red error "Message contained unnormalized quaternion"
and the TF_DENORMALIZED_QUATERNION / TF_NAN_INPUT warnings downstream.
"""

import math
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def normalize_quaternion(q):
    """Normalize a quaternion in place; fall back to identity when invalid."""
    vals = (q.x, q.y, q.z, q.w)
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
        q.w = 1.0
        return
    norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-6:
        q.x = 0.0
        q.y = 0.0
        q.z = 0.0
        q.w = 1.0
        return
    q.x /= norm
    q.y /= norm
    q.z /= norm
    q.w /= norm


def sanitize_vector(v):
    if not all(math.isfinite(x) for x in (v.x, v.y, v.z)):
        v.x = 0.0
        v.y = 0.0
        v.z = 0.0


class OdomSanitizer(Node):
    def __init__(self):
        super().__init__('odom_sanitizer')

        self.raw_topic = self.declare_parameter('raw_odom_topic', '/odom_raw').value
        self.out_topic = self.declare_parameter('odom_topic', '/odom').value
        self.frame_id = self.declare_parameter('frame_id', 'odom').value
        self.child_frame_id = self.declare_parameter(
            'child_frame_id', 'base_link').value

        # The Gazebo OdometryPublisher emits an all-zero orientation
        # quaternion (norm = 0) even though its twist.angular.z is correct.
        # A zero-norm quaternion destroys every downstream yaw computation,
        # so we reconstruct the heading by integrating the measured yaw rate
        # and always publish a clean, unit-norm orientation.
        self.yaw = 0.0
        self.last_stamp = None

        # Optional sensor-noise injection. Defaults
        # are 0.0 so the node stays transparent unless explicitly configured:
        #   sigma_v       : Gaussian std of the linear-velocity feedback (m/s);
        #   sigma_w       : Gaussian std of the angular-velocity feedback (rad/s);
        #   yaw_bias_rate : random-walk bias drift on the heading (rad/s per sqrt s).
        self.sigma_v = self.declare_parameter('sigma_v', 0.0).value
        self.sigma_w = self.declare_parameter('sigma_w', 0.0).value
        self.yaw_bias_rate = self.declare_parameter('yaw_bias_rate', 0.0).value
        self._bias = 0.0

        self.sub = self.create_subscription(Odometry, self.raw_topic, self.callback, 10)
        self.pub = self.create_publisher(Odometry, self.out_topic, 10)

        self.get_logger().info(
            f"OdomSanitizer ready: '{self.raw_topic}' -> '{self.out_topic}' "
            f"({self.frame_id} -> {self.child_frame_id}) "
            f"sigma_v={self.sigma_v} sigma_w={self.sigma_w} "
            f"yaw_bias_rate={self.yaw_bias_rate}")

    def callback(self, msg):
        out = Odometry()
        out.header = msg.header
        out.header.frame_id = self.frame_id
        out.child_frame_id = self.child_frame_id
        out.pose = msg.pose          # copies pose + 6x6 covariance
        out.twist = msg.twist        # copies twist + 6x6 covariance

        # Inject sensor noise into the velocity feedback (if configured).
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        if self.sigma_v > 0.0:
            v += self.sigma_v * random.gauss(0.0, 1.0)
        if self.sigma_w > 0.0:
            w += self.sigma_w * random.gauss(0.0, 1.0)

        # Integrate yaw from the (noisy) angular velocity, plus a random-walk
        # bias drift. Guard the time delta so a timestamp reset or duplicate
        # stamp cannot blow up the integral.
        dt = 0.0
        if self.last_stamp is not None:
            dt = (msg.header.stamp.sec - self.last_stamp.sec) + \
                 (msg.header.stamp.nanosec - self.last_stamp.nanosec) * 1e-9
        if 0.0 < dt < 0.5:
            if self.yaw_bias_rate > 0.0:
                self._bias += self.yaw_bias_rate * random.gauss(0.0, 1.0) * math.sqrt(dt)
            self.yaw += (w + self._bias) * dt
        self.last_stamp = msg.header.stamp

        normalize_quaternion(out.pose.pose.orientation)
        sanitize_vector(out.pose.pose.position)
        sanitize_vector(out.twist.twist.linear)
        sanitize_vector(out.twist.twist.angular)

        # Publish the (possibly noisy) velocity feedback.
        out.twist.twist.linear.x = v
        out.twist.twist.angular.z = w

        # Override the (broken) pose orientation with the integrated yaw.
        out.pose.pose.orientation.x = 0.0
        out.pose.pose.orientation.y = 0.0
        out.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        out.pose.pose.orientation.w = math.cos(self.yaw / 2.0)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomSanitizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
