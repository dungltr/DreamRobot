cmake_minimum_required(VERSION 3.8)
project(vslam_recovery_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(recovery_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)

# Shared safety-check header used by the C++ nodes.
include_directories(include)

# ---------------------------------------------------------------------------
# recovery_recorder_server : RecordLandmark service server, writes YAML
# ---------------------------------------------------------------------------
add_executable(recovery_recorder_server src/recovery_recorder_server.cpp)
ament_target_dependencies(recovery_recorder_server
  rclcpp
  nav_msgs
  geometry_msgs
  recovery_interfaces
)
target_link_libraries(recovery_recorder_server yaml-cpp)

# ---------------------------------------------------------------------------
# landmark_sampler_client : adaptive sampling client, requests landmark saving
# ---------------------------------------------------------------------------
add_executable(landmark_sampler_client src/landmark_sampler_client.cpp)
ament_target_dependencies(landmark_sampler_client
  rclcpp
  nav_msgs
  geometry_msgs
  recovery_interfaces
)

# ---------------------------------------------------------------------------
# recovery_controller : ExecuteRecovery action server + recovery trigger
# ---------------------------------------------------------------------------
add_executable(recovery_controller src/recovery_controller.cpp)
ament_target_dependencies(recovery_controller
  rclcpp
  rclcpp_action
  nav_msgs
  geometry_msgs
  std_msgs
  recovery_interfaces
)
target_link_libraries(recovery_controller yaml-cpp)

install(TARGETS
  recovery_recorder_server
  landmark_sampler_client
  recovery_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Python visualization / dashboard / plotting / sanitizer nodes + the shared
# optimal-trajectory helper imported by path_visualizer and the dashboard.
install(PROGRAMS
  src/path_visualizer.py
  src/recovery_dashboard_gui.py
  src/trajectory_plotter.py
  src/odom_sanitizer.py
  src/optimal_trajectory.py
  src/dream_sampler_node.py
  src/dream_core.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
