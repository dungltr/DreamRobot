#!/usr/bin/env python3
"""
path_visualizer.py

ROS 2 node that reads /odom and publishes two nav_msgs/msg/Path streams for
RViz 2, so the forward and recovery trajectories can be compared side by side
(cf. manuscript Fig. 11, the RViz view):

  * /trajectory/forward_path  (green)  - normal forward motion
  * /trajectory/recovery_path (red)    - back travel after vision failure

It also broadcasts the odom -> base_link transform from the odometry pose so
that RViz can be run with `odom` as the fixed frame.
"""

import math
import os
import re
import sys

# Allow importing the co-located trajectory helper after colcon installation
# (scripts are installed into the same lib/<pkg> directory).
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from std_msgs.msg import Bool, Int32
from visualization_msgs.msg import Marker
import tf2_ros

import optimal_trajectory as ot

_LM_RE = re.compile(r'^landmark_(\d+):\s*\[(.*)\]\s*$')


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
        self.auto_mode_topic = self.declare_parameter('auto_mode_topic', 'auto_mode').value
        self.landmarks_topic = self.declare_parameter(
            'landmarks_topic', '/trajectory/landmarks').value
        self.yaml_file = self.declare_parameter('yaml_file', 'path_landmarks.yaml').value
        self.world = self.declare_parameter('world', 'recovery_corridor.sdf').value
        ot.configure(self.world)

        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 10)
        self.trigger_sub = self.create_subscription(
            Bool, self.trigger_topic, self.trigger_callback, 10)
        self.mode_sub = self.create_subscription(
            Int32, self.auto_mode_topic, self.mode_callback, 10)

        self.forward_pub = self.create_publisher(Path, self.forward_topic, 10)
        self.recovery_pub = self.create_publisher(Path, self.recovery_topic, 10)
        self.planned_pub = self.create_publisher(Path, self.planned_topic, 10)
        self.landmark_pub = self.create_publisher(Marker, self.landmarks_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.forward_path = Path()
        self.forward_path.header.frame_id = self.map_frame
        self.recovery_path = Path()
        self.recovery_path.header.frame_id = self.map_frame

        self.recovering = False
        self.current_mode = 3

        # Publish the static optimal reference trajectory once and then
        # periodically so RViz (even a late-joining one) can render it.
        self.publish_planned_path()
        self.planned_timer = self.create_timer(1.0, self.publish_planned_path)
        self.landmark_timer = self.create_timer(1.0, self.publish_landmarks)

        self.get_logger().info(
            f"PathVisualizer ready. forward='{self.forward_topic}' "
            f"recovery='{self.recovery_topic}' planned='{self.planned_topic}' "
            f"frame='{self.map_frame}'")

    def publish_planned_path(self):
        """Publish the planned path for the active auto mode.

        Modes 3/4 follow the continuous quintic spline; modes 1/2 follow the
        straight-segment waypoint polyline, so the yellow reference must switch
        with the mode instead of always showing the same spline.
        """
        if self.current_mode in (1, 2):
            xs, ys, yaws = ot.waypoint_polyline_points(self.world)
        else:
            xs, ys = ot.planned_path_points()
            yaws = [ot.SPLINE.heading(x) for x in xs]
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        for x, y, yaw in zip(xs, ys, yaws):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = self.map_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        self.planned_pub.publish(path)

    def mode_callback(self, msg):
        """Repaint the planned path when the dashboard switches auto mode."""
        if msg.data != self.current_mode:
            self.current_mode = msg.data
            self.publish_planned_path()

    def publish_landmarks(self):
        """Publish the recorded DREAM landmarks as spheres so the sparse
        sampling (and the resulting chord corner-cutting on the return) is
        visible against the forward/recovery paths."""
        pts = self._read_landmarks()
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'dream_landmarks'
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        for (x, y) in pts:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05
            marker.points.append(p)
        self.landmark_pub.publish(marker)

    def _read_landmarks(self):
        """Return [(x, y), ...] parsed from the recorded landmarks YAML file."""
        pts = []
        try:
            with open(self.yaml_file, 'r') as fh:
                for line in fh:
                    m = _LM_RE.match(line.strip())
                    if not m:
                        continue
                    try:
                        vals = [float(v) for v in m.group(2).split(',')]
                    except ValueError:
                        continue
                    if len(vals) >= 7:
                        pts.append((vals[5], vals[6]))
        except (OSError, IOError):
            pass
        return pts

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
