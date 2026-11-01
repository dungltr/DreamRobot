#!/usr/bin/env python3
"""
path_visualizer.py

ROS 2 node that reads /odom and publishes two nav_msgs/msg/Path streams for
RViz 2, so the forward and recovery trajectories can be compared side by side
(as in Figures 12 and 13 of the paper):

  * /trajectory/forward_path  (green)  - normal forward motion
  * /trajectory/recovery_path (red)    - back travel after vision failure

It also broadcasts the odom -> base_link transform from the odometry pose so
that RViz can be run with `odom` as the fixed frame.
"""

import math
import os
import sys

# Allow importing the co-located trajectory helper after colcon installation
# (scripts are installed into the same lib/<pkg> directory).
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Bool
import tf2_ros

import optimal_trajectory as ot


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


def valid_position(x, y, z):
    return all(math.isfinite(v) for v in (x, y, z))


class PathVisualizer(Node):
    """Track the robot trajectory and split it into forward / recovery paths."""

    def __init__(self):
        super().__init__('path_visualizer')

        self.odom_topic = self.declare_parameter('odom_topic', '/odom').value
        self.trigger_topic = self.declare_parameter(
            'failure_trigger_topic', '/camera_failure_trigger').value
        self.forward_topic = self.declare_parameter(
            'forward_path_topic', '/trajectory/forward_path').value
        self.recovery_topic = self.declare_parameter(
            'recovery_path_topic', '/trajectory/recovery_path').value
        self.planned_topic = self.declare_parameter(
            'planned_path_topic', '/trajectory/planned_path').value
        self.map_frame = self.declare_parameter('map_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        world = self.declare_parameter('world', 'recovery_corridor.sdf').value
        ot.configure(world)

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.trigger_sub = self.create_subscription(
            Bool, self.trigger_topic, self.trigger_callback, 10)

        self.forward_pub = self.create_publisher(Path, self.forward_topic, 10)
        self.recovery_pub = self.create_publisher(Path, self.recovery_topic, 10)
        self.planned_pub = self.create_publisher(Path, self.planned_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.forward_path = Path()
        self.forward_path.header.frame_id = self.map_frame
        self.recovery_path = Path()
        self.recovery_path.header.frame_id = self.map_frame

        self.recovering = False

        # Publish the static optimal reference trajectory once and then
        # periodically so RViz (even a late-joining one) can render it.
        self.publish_planned_path()
        self.planned_timer = self.create_timer(1.0, self.publish_planned_path)

        self.get_logger().info(
            f"PathVisualizer ready. forward='{self.forward_topic}' "
            f"recovery='{self.recovery_topic}' planned='{self.planned_topic}' "
            f"frame='{self.map_frame}'")

    def publish_planned_path(self):
        """Publish the quintic-spline reference path for RViz."""
        xs, ys = ot.planned_path_points()
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        for x, y in zip(xs, ys):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            # Heading tangent to the path (identity roll/pitch).
            yaw = ot.SPLINE.heading(x)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        self.planned_pub.publish(path)

    def trigger_callback(self, msg):
        if msg.data and not self.recovering:
            self.recovering = True
            # Seed the recovery path from the last forward pose so the red path
            # starts exactly where the green path ended (visual continuity).
            if self.forward_path.poses:
                self.recovery_path = Path()
                self.recovery_path.header.frame_id = self.map_frame
                self.recovery_path.poses.append(self.forward_path.poses[-1])
            self.get_logger().info('Camera failure detected; recording recovery path.')

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        if not valid_position(p.x, p.y, p.z):
            self.get_logger().warn(
                'NaN/infinite position in /odom; ignoring frame.', throttle_duration_sec=5.0)
            return

        # Fix TF_DENORMALIZED_QUATERNION / TF_NAN_INPUT before publishing TF.
        qx, qy, qz, qw = sanitize_quaternion(msg.pose.pose.orientation)

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.map_frame
        pose.pose = msg.pose.pose
        pose.pose.position.x = p.x
        pose.pose.position.y = p.y
        pose.pose.position.z = p.z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        if self.recovering:
            self.recovery_path.header.stamp = msg.header.stamp
            self.recovery_path.poses.append(pose)
            self.recovery_pub.publish(self.recovery_path)
        else:
            self.forward_path.header.stamp = msg.header.stamp
            self.forward_path.poses.append(pose)
            self.forward_pub.publish(self.forward_path)

        # Publish the odom -> base_link transform.
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
