# recovery_system.launch.py
#
# Launch the three blind-navigation recovery nodes together on a shared,
# flexible parameter configuration:
#
#   1. recovery_recorder_server  : RecordLandmark service server + YAML writer
#   2. landmark_sampler_client   : adaptive Shannon-Nyquist sampler / client
#      (or dream_sampler_node   : DREAM event-triggered sampler, when
#       `use_dream:=true`)
#   3. recovery_controller       : ExecuteRecovery action server + recovery
#                                  trigger (/camera_failure_trigger)

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Resolve the workspace root from the launch script location so the recorder
# and controller share the same landmarks file even when launched from another
# working directory.
_LAUNCH_DIR = os.path.dirname(os.path.realpath(__file__))  # <ws>/src/vslam_recovery_core/launch
_PKG_DIR = os.path.dirname(_LAUNCH_DIR)                    # <ws>/src/vslam_recovery_core
_SRC_DIR = os.path.dirname(_PKG_DIR)                       # <ws>/src
_WS_DIR = os.path.dirname(_SRC_DIR)                        # <ws>
_DEFAULT_YAML = os.path.join(_WS_DIR, "path_landmarks.yaml")


def generate_launch_description():
    # Shared configuration arguments.
    odom_topic = LaunchConfiguration("odom_topic")
    service_name = LaunchConfiguration("service_name")
    yaml_file = LaunchConfiguration("yaml_file")

    action_name = LaunchConfiguration("action_name")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    failure_trigger_topic = LaunchConfiguration("failure_trigger_topic")

    straight_frequency = LaunchConfiguration("straight_frequency")
    curve_frequency = LaunchConfiguration("curve_frequency")
    straight_period = LaunchConfiguration("straight_period")
    curve_period = LaunchConfiguration("curve_period")
    monitor_period = LaunchConfiguration("monitor_period")
    window_size = LaunchConfiguration("window_size")
    rotate_angular_velocity = LaunchConfiguration("rotate_angular_velocity")
    drive_frequency = LaunchConfiguration("drive_frequency")

    # DREAM sampler selection + tuning parameters.
    use_dream = LaunchConfiguration("use_dream")
    r2_threshold = LaunchConfiguration("r2_threshold")
    min_window_size = LaunchConfiguration("min_window_size")
    max_window_size = LaunchConfiguration("max_window_size")
    regress_speed = LaunchConfiguration("regress_speed")
    min_event_spacing = LaunchConfiguration("min_event_spacing")

    return LaunchDescription([
        DeclareLaunchArgument("odom_topic", default_value="/odom"),
        DeclareLaunchArgument("service_name", default_value="record_landmark"),
        DeclareLaunchArgument("yaml_file", default_value=_DEFAULT_YAML),

        DeclareLaunchArgument("action_name", default_value="execute_recovery"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument(
            "failure_trigger_topic", default_value="/camera_failure_trigger"),

        # Shannon-Nyquist sampling frequencies (f = 1/T).
        DeclareLaunchArgument("straight_frequency", default_value="0.5"),
        DeclareLaunchArgument("curve_frequency", default_value="1.0"),
        DeclareLaunchArgument("straight_period", default_value="2.0"),
        DeclareLaunchArgument("curve_period", default_value="1.0"),
        DeclareLaunchArgument("monitor_period", default_value="0.1"),
        DeclareLaunchArgument("window_size", default_value="4"),
        DeclareLaunchArgument("rotate_angular_velocity", default_value="0.8"),
        DeclareLaunchArgument("drive_frequency", default_value="20.0"),

        # DREAM sampler: use_dream:=true swaps the Shannon-Nyquist sampler for
        # the R^2-triggered DREAM sampler.
        DeclareLaunchArgument("use_dream", default_value="false"),
        DeclareLaunchArgument("r2_threshold", default_value="0.98"),
        DeclareLaunchArgument("min_window_size", default_value="5"),
        DeclareLaunchArgument("max_window_size", default_value="500"),
        DeclareLaunchArgument("regress_speed", default_value="true"),
        DeclareLaunchArgument("min_event_spacing", default_value="0.2"),

        # 1. Service server that records path landmarks to YAML.
        Node(
            package="vslam_recovery_core",
            executable="recovery_recorder_server",
            name="recovery_recorder_server",
            output="screen",
            parameters=[{
                "odom_topic": odom_topic,
                "service_name": service_name,
                "yaml_file": yaml_file,
            }],
        ),

        # 2a. Adaptive Shannon-Nyquist sampler / service client (default).
        Node(
            package="vslam_recovery_core",
            executable="landmark_sampler_client",
            name="landmark_sampler_client",
            output="screen",
            condition=UnlessCondition(use_dream),
            parameters=[{
                "odom_topic": odom_topic,
                "service_name": service_name,
                "straight_frequency": straight_frequency,
                "curve_frequency": curve_frequency,
                "monitor_period": monitor_period,
                "window_size": window_size,
            }],
        ),

        # 2b. DREAM event-triggered sampler / service client (use_dream:=true).
        Node(
            package="vslam_recovery_core",
            executable="dream_sampler_node.py",
            name="dream_sampler_node",
            output="screen",
            condition=IfCondition(use_dream),
            parameters=[{
                "odom_topic": odom_topic,
                "service_name": service_name,
                "r2_threshold": r2_threshold,
                "min_window_size": min_window_size,
                "max_window_size": max_window_size,
                "regress_speed": regress_speed,
                "min_event_spacing": min_event_spacing,
            }],
        ),

        # 3. Action server + recovery trigger handler.
        Node(
            package="vslam_recovery_core",
            executable="recovery_controller",
            name="recovery_controller",
            output="screen",
            parameters=[{
                "action_name": action_name,
                "cmd_vel_topic": cmd_vel_topic,
                "failure_trigger_topic": failure_trigger_topic,
                "odom_topic": odom_topic,
                "yaml_file": yaml_file,
                "straight_period": straight_period,
                "curve_period": curve_period,
                "rotate_angular_velocity": rotate_angular_velocity,
                "drive_frequency": drive_frequency,
            }],
        ),
    ])
