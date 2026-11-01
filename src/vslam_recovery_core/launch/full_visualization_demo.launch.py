# full_visualization_demo.launch.py
#
# Complete blind navigation recovery demo: Gazebo Fortress + ros_gz_bridge +
# the three core recovery nodes + RViz 2 trajectory visualization + GUI
# dashboard.
#
# Components launched:
#   1. Gazebo Fortress loading worlds/recovery_corridor.sdf (differential
#      drive robot + camera in a 1.7 m corridor with two staggered obstacles).
#   2. ros_gz_bridge (config/bridge_config.yaml).
#   3. robot_state_publisher (urdf/recovery_robot.urdf) for RViz RobotModel.
#   4. recovery_recorder_server, landmark_sampler_client, recovery_controller.
#   5. path_visualizer.py + rviz2 (config/recovery_view.rviz).
#   6. recovery_dashboard_gui.py.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Resolve the workspace root from the launch script location so the recorder,
# controller and dashboard share the same landmarks file.
_LAUNCH_DIR = os.path.dirname(os.path.realpath(__file__))  # <ws>/src/vslam_recovery_core/launch
_PKG_DIR = os.path.dirname(_LAUNCH_DIR)                    # <ws>/src/vslam_recovery_core
_SRC_DIR = os.path.dirname(_PKG_DIR)                       # <ws>/src
_WS_DIR = os.path.dirname(_SRC_DIR)                        # <ws>
_DEFAULT_YAML = os.path.join(_WS_DIR, 'path_landmarks.yaml')


def _load_urdf(path):
    with open(path, 'r') as fh:
        return fh.read()


def generate_launch_description():
    pkg = 'vslam_recovery_core'
    pkg_share = get_package_share_directory(pkg)
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(pkg_share, 'worlds', 'recovery_corridor.sdf')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge_config.yaml')
    rviz_config = os.path.join(pkg_share, 'config', 'recovery_view.rviz')
    urdf_path = os.path.join(pkg_share, 'urdf', 'recovery_robot.urdf')

    # Shared configuration.
    yaml_file = LaunchConfiguration('yaml_file')
    odom_topic = LaunchConfiguration('odom_topic')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    failure_trigger_topic = LaunchConfiguration('failure_trigger_topic')
    service_name = LaunchConfiguration('service_name')
    action_name = LaunchConfiguration('action_name')

    # Gazebo args: server-only (-s) when headless is requested.
    gz_args = LaunchConfiguration('gz_args')
    rviz_enabled = LaunchConfiguration('rviz')

    # DREAM sampler selection + tuning parameters.
    use_dream = LaunchConfiguration('use_dream')
    r2_threshold = LaunchConfiguration('r2_threshold')
    min_window_size = LaunchConfiguration('min_window_size')
    max_window_size = LaunchConfiguration('max_window_size')
    regress_speed = LaunchConfiguration('regress_speed')
    min_event_spacing = LaunchConfiguration('min_event_spacing')

    # Sensor-noise injection (odom_sanitizer).
    sigma_v = LaunchConfiguration('sigma_v')
    sigma_w = LaunchConfiguration('sigma_w')
    yaw_bias_rate = LaunchConfiguration('yaw_bias_rate')

    # 1. Gazebo Fortress.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    # 2. ros_gz_bridge.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output='screen',
    )

    # 2b. odom_sanitizer: /odom_raw -> clean /odom (normalized quaternion).
    odom_sanitizer = Node(
        package=pkg,
        executable='odom_sanitizer.py',
        name='odom_sanitizer',
        output='screen',
        parameters=[{
            'raw_odom_topic': '/odom_raw',
            'odom_topic': '/odom',
            'frame_id': 'odom',
            'child_frame_id': 'base_link',
            'sigma_v': sigma_v,
            'sigma_w': sigma_w,
            'yaw_bias_rate': yaw_bias_rate,
        }],
    )

    # 3. robot_state_publisher for the RViz RobotModel display.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': _load_urdf(urdf_path)}],
        output='screen',
    )

    # 3b. Static camera transforms.
    #      base_link -> camera_link comes from the URDF fixed joint via
    #      robot_state_publisher (do NOT add a duplicate static transform here:
    #      it would publish camera_link twice and break the TF tree).
    #
    #      camera_link -> camera_optical_frame applies the z-forward optical
    #      frame convention used by ROS camera nodes.
    static_camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'camera_link',
                   '--child-frame-id', 'camera_optical_frame'],
    )

    #      The gz-sim camera sensor publishes image/camera_info with the
    #      scoped frame id `recovery_robot/camera_link/camera`. Connect that
    #      frame to our ROS `camera_link` so RViz can resolve the camera pose
    #      for the 3D overlay instead of reporting
    #      "No transform from [recovery_robot/camera_link/camera]".
    static_gz_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['--frame-id', 'camera_link',
                   '--child-frame-id', 'recovery_robot/camera_link/camera'],
    )

    # 4. Core recovery nodes.
    recorder = Node(
        package=pkg,
        executable='recovery_recorder_server',
        name='recovery_recorder_server',
        output='screen',
        parameters=[{
            'odom_topic': odom_topic,
            'service_name': service_name,
            'yaml_file': yaml_file,
        }],
    )

    sampler = Node(
        package=pkg,
        executable='landmark_sampler_client',
        name='landmark_sampler_client',
        output='screen',
        condition=UnlessCondition(use_dream),
        parameters=[{
            'odom_topic': odom_topic,
            'service_name': service_name,
            'straight_frequency': 0.5,
            'curve_frequency': 1.0,
            'monitor_period': 0.1,
            'window_size': 4,
        }],
    )

    # 4b. DREAM event-triggered sampler (use_dream:=true).
    dream_sampler = Node(
        package=pkg,
        executable='dream_sampler_node.py',
        name='dream_sampler_node',
        output='screen',
        condition=IfCondition(use_dream),
        parameters=[{
            'odom_topic': odom_topic,
            'service_name': service_name,
            'r2_threshold': r2_threshold,
            'min_window_size': min_window_size,
            'max_window_size': max_window_size,
            'regress_speed': regress_speed,
            'min_event_spacing': min_event_spacing,
        }],
    )

    controller = Node(
        package=pkg,
        executable='recovery_controller',
        name='recovery_controller',
        output='screen',
        parameters=[{
            'action_name': action_name,
            'cmd_vel_topic': cmd_vel_topic,
            'failure_trigger_topic': failure_trigger_topic,
            'odom_topic': odom_topic,
            'yaml_file': yaml_file,
            'straight_period': 2.0,
            'curve_period': 1.0,
            'rotate_angular_velocity': 0.8,
            'drive_frequency': 20.0,
        }],
    )

    # 5. Trajectory visualization.
    visualizer = Node(
        package=pkg,
        executable='path_visualizer.py',
        name='path_visualizer',
        output='screen',
        parameters=[{
            'odom_topic': odom_topic,
            'failure_trigger_topic': failure_trigger_topic,
            'forward_path_topic': '/trajectory/forward_path',
            'recovery_path_topic': '/trajectory/recovery_path',
            'planned_path_topic': '/trajectory/planned_path',
            'map_frame': 'odom',
            'base_frame': 'base_link',
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(rviz_enabled),
    )

    # 6. GUI dashboard.
    dashboard = Node(
        package=pkg,
        executable='recovery_dashboard_gui.py',
        name='recovery_dashboard_gui',
        output='screen',
        parameters=[{
            'odom_topic': odom_topic,
            'cmd_vel_topic': cmd_vel_topic,
            'failure_trigger_topic': failure_trigger_topic,
            'yaml_file': yaml_file,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('yaml_file', default_value=_DEFAULT_YAML),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument(
            'failure_trigger_topic', default_value='/camera_failure_trigger'),
        DeclareLaunchArgument('service_name', default_value='record_landmark'),
        DeclareLaunchArgument('action_name', default_value='execute_recovery'),
        DeclareLaunchArgument(
            'gz_args', default_value=f'-r {world_path}',
            description='Gazebo Fortress arguments (use -s for headless server).'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz 2.'),
        DeclareLaunchArgument('use_dream', default_value='false',
                              description='Use the DREAM event-triggered sampler.'),
        DeclareLaunchArgument('r2_threshold', default_value='0.98'),
        DeclareLaunchArgument('min_window_size', default_value='5'),
        DeclareLaunchArgument('max_window_size', default_value='500'),
        DeclareLaunchArgument('regress_speed', default_value='true'),
        DeclareLaunchArgument('min_event_spacing', default_value='0.2'),
        DeclareLaunchArgument('sigma_v', default_value='0.0',
                              description='odometry linear-velocity noise std (m/s)'),
        DeclareLaunchArgument('sigma_w', default_value='0.0',
                              description='odometry angular-velocity noise std (rad/s)'),
        DeclareLaunchArgument('yaw_bias_rate', default_value='0.0',
                              description='heading random-walk bias (rad/s per sqrt s)'),
        gz_sim,
        bridge,
        odom_sanitizer,
        robot_state_publisher,
        static_camera_optical_tf,
        static_gz_camera_tf,
        recorder,
        sampler,
        dream_sampler,
        controller,
        visualizer,
        rviz,
        dashboard,
    ])
