// recovery_controller.cpp
//
// ExecuteRecovery action server and blind navigation trigger.
//
// The node listens to /camera_failure_trigger (std_msgs/msg/Bool). When the
// trigger becomes true it executes the blind navigation recovery sequence:
//   1. rotate the robot in place by pi radians through /cmd_vel;
//   2. read path_landmarks.yaml and reconstruct the recorded waypoints;
//   3. follow those waypoints back to the origin with a closed-loop
//      pure-pursuit controller (odometry feedback only, no camera).
//
// The same node exposes an ExecuteRecovery action server so an external
// action client can also submit individual five-variable motion goals.

#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "recovery_interfaces/action/execute_recovery.hpp"
#include "yaml-cpp/yaml.h"
#include "vslam_recovery_core/safety_checks.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double MAX_LIN_ACCEL = 0.20;   // m/s^2, return slew-rate limit
constexpr double MAX_ANG_ACCEL = 0.50;   // rad/s^2, return slew-rate limit

/// \brief Smooth slew limiter used by the closed-loop return controller.
double slew(double current, double target, double max_step)
{
  const double delta = target - current;
  if (std::fabs(delta) <= max_step) {
    return target;
  }
  return current + (delta > 0.0 ? max_step : -max_step);
}

/// \brief Normalize an angle to [-pi, pi].
double normalize_angle(double a)
{
  while (a > kPi) {
    a -= 2.0 * kPi;
  }
  while (a < -kPi) {
    a += 2.0 * kPi;
  }
  return a;
}

/// \brief Clamp `v` into [lo, hi].
double clamp_value(double v, double lo, double hi)
{
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
}

}  // namespace

class RecoveryController : public rclcpp::Node
{
public:
  using ExecuteRecovery = recovery_interfaces::action::ExecuteRecovery;
  using GoalHandleExecuteRecovery = rclcpp_action::ServerGoalHandle<ExecuteRecovery>;

  RecoveryController()
  : Node("recovery_controller"),
    recovering_(false),
    have_odom_(false),
    odom_distance_(0.0),
    last_x_(0.0),
    last_y_(0.0)
  {
    action_name_ = declare_parameter<std::string>("action_name", "execute_recovery");
    cmd_vel_topic_ = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    failure_trigger_topic_ =
      declare_parameter<std::string>("failure_trigger_topic", "/camera_failure_trigger");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    yaml_file_ = declare_parameter<std::string>("yaml_file", "path_landmarks.yaml");
    straight_period_ = declare_parameter<double>("straight_period", 2.0);
    curve_period_ = declare_parameter<double>("curve_period", 1.0);
    rotate_angular_velocity_ = declare_parameter<double>("rotate_angular_velocity", 0.8);
    max_angular_velocity_ = declare_parameter<double>("max_angular_velocity", 0.25);
    drive_frequency_ = declare_parameter<double>("drive_frequency", 20.0);

    cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    trigger_sub_ = create_subscription<std_msgs::msg::Bool>(
      failure_trigger_topic_, 10,
      std::bind(&RecoveryController::trigger_callback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&RecoveryController::odom_callback, this, std::placeholders::_1));

    action_server_ = rclcpp_action::create_server<ExecuteRecovery>(
      this,
      action_name_,
      std::bind(&RecoveryController::handle_goal, this,
        std::placeholders::_1, std::placeholders::_2),
      std::bind(&RecoveryController::handle_cancel, this, std::placeholders::_1),
      std::bind(&RecoveryController::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "RecoveryController ready. action='%s' cmd_vel='%s' trigger='%s' yaml='%s'",
      action_name_.c_str(), cmd_vel_topic_.c_str(),
      failure_trigger_topic_.c_str(), yaml_file_.c_str());
  }

private:
  struct Landmark
  {
    double v;
    double w;
    double d;
    double T;
    double psi;
    double x;
    double y;
    bool has_xy;
  };

  // -------------------------------------------------------------------------
  // Action server callbacks
  // -------------------------------------------------------------------------
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ExecuteRecovery::Goal> goal)
  {
    RCLCPP_INFO(get_logger(),
      "Received recovery goal: v=%.3f w=%.3f T=%.3f d=%.3f yaw=%.3f",
      goal->linear_x, goal->angular_z, goal->duration,
      goal->cmd_distance, goal->yaw_angle);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleExecuteRecovery>)
  {
    RCLCPP_INFO(get_logger(), "Recovery goal cancel requested.");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleExecuteRecovery> goal_handle)
  {
    // Execute the (short lived) motion goal on a separate thread so that the
    // node executor keeps processing subscriptions and services.
    std::thread{std::bind(&RecoveryController::execute, this, std::placeholders::_1),
      goal_handle}.detach();
  }

  void execute(const std::shared_ptr<GoalHandleExecuteRecovery> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<ExecuteRecovery::Feedback>();
    auto result = std::make_shared<ExecuteRecovery::Result>();

    feedback->fbk_msg.data = "Executing recovery motion goal";
    goal_handle->publish_feedback(feedback);

    const double start_distance = get_odom_distance();
    drive_for(goal->linear_x, goal->angular_z, goal->duration);
    const double end_distance = get_odom_distance();

    result->distance = have_odom_ ? (end_distance - start_distance) : goal->cmd_distance;

    feedback->fbk_msg.data = "Recovery motion goal completed";
    goal_handle->publish_feedback(feedback);
    goal_handle->succeed(result);
  }

  // -------------------------------------------------------------------------
  // Failure trigger and recovery sequence
  // -------------------------------------------------------------------------
  void trigger_callback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    if (!msg->data) {
      return;
    }
    if (recovering_.exchange(true)) {
      RCLCPP_WARN(get_logger(), "Recovery already in progress; ignoring new trigger.");
      return;
    }
    RCLCPP_INFO(get_logger(), "Camera failure trigger received; starting blind navigation.");
    // Snapshot the recorded forward path *before* any recovery motion so that
    // landmarks produced during the pi-turn / return travel cannot leak into
    // the return path.
    const std::vector<Landmark> landmarks = load_landmarks();
    std::thread(&RecoveryController::run_recovery, this, landmarks).detach();
  }

  void run_recovery(const std::vector<Landmark> & landmarks)
  {
    // Serialize with the ExecuteRecovery action goals so only one motion
    // source drives /cmd_vel at a time.
    std::lock_guard<std::mutex> lock(drive_mutex_);
    try {
      rotate_in_place();
      traverse_reversed_path(landmarks);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Recovery aborted: %s", e.what());
    }
    publish_twist(0.0, 0.0);
    recovering_ = false;
    RCLCPP_INFO(get_logger(), "Recovery sequence finished.");
  }

  void rotate_in_place()
  {
    if (rotate_angular_velocity_ <= 1e-6) {
      RCLCPP_WARN(get_logger(), "Invalid rotate_angular_velocity; skipping rotation.");
      return;
    }

    // Closed-loop pi turn: keep turning until the measured yaw has advanced
    // by pi, instead of assuming an exact open-loop duration.
    const double target = normalize_angle(get_yaw() + kPi);
    RCLCPP_INFO(get_logger(),
      "Rotating in place to yaw %.3f rad (pi turn, closed-loop).", target);

    rclcpp::Rate rate(drive_frequency_);
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(12.0));

    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      const double error = normalize_angle(target - get_yaw());
      if (std::fabs(error) < 0.03) {
        break;
      }
      const double w = clamp_value(2.0 * error,
        -rotate_angular_velocity_, rotate_angular_velocity_);
      publish_twist(0.0, w);
      rate.sleep();
    }
    publish_twist(0.0, 0.0);
  }

  void traverse_reversed_path(const std::vector<Landmark> & landmarks)
  {
    if (landmarks.empty()) {
      RCLCPP_WARN(get_logger(),
        "No landmarks found in '%s'; nothing to traverse.", yaml_file_.c_str());
      return;
    }

    // Drop zero-distance samples (duplicates / stops): they carry no path
    // information and would otherwise inject spurious return segments.
    std::vector<Landmark> path;
    path.reserve(landmarks.size());
    for (const auto & lm : landmarks) {
      if (!path.empty() && lm.d < 1e-4) {
        continue;
      }
      path.push_back(lm);
    }

    // Reconstruct the recorded forward path as a sequence of waypoints. New
    // recordings store (x, y) directly; legacy 5-field files are reconstructed
    // from the recorded distance/heading (approximate fallback).
    std::vector<std::array<double, 2>> forward;
    forward.reserve(path.size());
    double rx = 0.0;
    double ry = 0.0;
    for (const auto & lm : path) {
      if (lm.has_xy) {
        rx = lm.x;
        ry = lm.y;
      } else {
        const double psi = normalize_angle(lm.psi);
        rx += lm.d * std::cos(psi);
        ry += lm.d * std::sin(psi);
      }
      forward.push_back({rx, ry});
    }

    // Return path: current pose -> recorded waypoints in reverse -> origin.
    double cur_x = 0.0;
    double cur_y = 0.0;
    get_position(cur_x, cur_y);
    std::vector<std::array<double, 2>> return_path;
    return_path.reserve(forward.size() + 2);
    return_path.push_back({cur_x, cur_y});
    for (auto it = forward.rbegin(); it != forward.rend(); ++it) {
      return_path.push_back(*it);
    }
    if (std::hypot(return_path.back()[0], return_path.back()[1]) > 1e-3) {
      return_path.push_back({0.0, 0.0});
    }

    RCLCPP_INFO(get_logger(),
      "Back-travelling %zu landmark(s) (filtered from %zu) with closed-loop "
      "path following.",
      path.size(), landmarks.size());
    follow_path_pure_pursuit(return_path);
  }

  void follow_path_pure_pursuit(
    const std::vector<std::array<double, 2>> & path)
  {
    const double kSampleSpacing = 0.05;   // m, polyline interpolation step
    const double kLookahead = 0.45;       // m
    const double kBaseSpeed = 0.12;       // m/s
    const double kFinalTolerance = 0.03;  // m

    // Dense interpolation of the return polyline for a smooth lookahead point.
    std::vector<std::array<double, 2>> samples;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
      const double dx = path[i + 1][0] - path[i][0];
      const double dy = path[i + 1][1] - path[i][1];
      const double seg = std::hypot(dx, dy);
      const int n = std::max(
        1, static_cast<int>(std::ceil(seg / kSampleSpacing)));
      for (int j = 0; j < n; ++j) {
        const double f = static_cast<double>(j) / static_cast<double>(n);
        samples.push_back({path[i][0] + dx * f, path[i][1] + dy * f});
      }
    }
    samples.push_back(path.back());

    rclcpp::Rate rate(drive_frequency_);
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(180.0));

    double last_v = 0.0;
    double last_w = 0.0;
    const double dt = 1.0 / drive_frequency_;

    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      double x = 0.0;
      double y = 0.0;
      get_position(x, y);
      const double yaw = get_yaw();
      const double dstart = std::hypot(x, y);

      if (dstart < kFinalTolerance) {
        publish_twist(0.0, 0.0);
        break;
      }

      // Nearest sample to the robot.
      std::size_t best = 0;
      double best_d = std::numeric_limits<double>::max();
      for (std::size_t i = 0; i < samples.size(); ++i) {
        const double dx = samples[i][0] - x;
        const double dy = samples[i][1] - y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d) {
          best_d = d2;
          best = i;
        }
      }

      // Advance by the lookahead distance along the polyline.
      std::array<double, 2> goal = samples.back();
      double acc = 0.0;
      for (std::size_t i = best; i + 1 < samples.size(); ++i) {
        acc += std::hypot(samples[i + 1][0] - samples[i][0],
                          samples[i + 1][1] - samples[i][1]);
        if (acc >= kLookahead) {
          goal = samples[i + 1];
          break;
        }
      }

      const double alpha = normalize_angle(
        std::atan2(goal[1] - y, goal[0] - x) - yaw);

      double v_des = kBaseSpeed * std::max(0.0, std::cos(alpha));
      const double approach = clamp_value(dstart / 0.4, 0.0, 1.0);
      v_des = std::min(v_des, std::max(0.02, kBaseSpeed * approach));

      double w_des =
        2.0 * v_des * std::sin(alpha) / std::max(kLookahead, 1e-3);
      w_des = clamp_value(w_des, -max_angular_velocity_, max_angular_velocity_);

      const double v = slew(last_v, v_des, MAX_LIN_ACCEL * dt);
      const double w = slew(last_w, w_des, MAX_ANG_ACCEL * dt);
      last_v = v;
      last_w = w;
      publish_twist(std::max(0.0, v),
        clamp_value(w, -max_angular_velocity_, max_angular_velocity_));
      rate.sleep();
    }
    publish_twist(0.0, 0.0);
  }

  // -------------------------------------------------------------------------
  // Odometry (distance travelled) helpers
  // -------------------------------------------------------------------------
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Ignore frames with NaN / infinite position (prevents corrupting the
    // travelled-distance estimate).
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    if (!vslam_recovery_core::is_valid_position(x, y, msg->pose.pose.position.z)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "NaN position received from odom; ignoring frame.");
      return;
    }

    // Extract a sanitized yaw for closed-loop heading feedback.
    geometry_msgs::msg::Quaternion q = msg->pose.pose.orientation;
    vslam_recovery_core::sanitize_quaternion(q);
    const double sy = 2.0 * (q.w * q.z + q.x * q.y);
    const double cy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    latest_yaw_ = std::atan2(sy, cy);

    if (have_odom_) {
      const double dx = x - last_x_;
      const double dy = y - last_y_;
      odom_distance_ += std::hypot(dx, dy);
    }
    last_x_ = x;
    last_y_ = y;
    have_odom_ = true;
  }

  double get_odom_distance()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return odom_distance_;
  }

  void get_position(double & x, double & y)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    x = last_x_;
    y = last_y_;
  }

  double get_yaw()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return latest_yaw_;
  }

  // -------------------------------------------------------------------------
  // Velocity publishing
  // -------------------------------------------------------------------------
  void publish_twist(double linear_x, double angular_z)
  {
    geometry_msgs::msg::Twist msg;
    msg.linear.x = linear_x;
    msg.angular.z = angular_z;
    cmd_vel_pub_->publish(msg);
  }

  void drive_for(double linear_x, double angular_z, double duration)
  {
    std::lock_guard<std::mutex> lock(drive_mutex_);

    if (duration <= 0.0) {
      publish_twist(0.0, 0.0);
      return;
    }

    const auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(duration));
    const auto end_time = std::chrono::steady_clock::now() + duration_ns;

    rclcpp::Rate rate(drive_frequency_);
    while (rclcpp::ok() && std::chrono::steady_clock::now() < end_time) {
      publish_twist(linear_x, angular_z);
      rate.sleep();
    }
    publish_twist(0.0, 0.0);
  }

  // -------------------------------------------------------------------------
  // YAML loading
  // -------------------------------------------------------------------------
  std::vector<Landmark> load_landmarks()
  {
    std::vector<Landmark> landmarks;
    try {
      const YAML::Node root = YAML::LoadFile(yaml_file_);
      if (!root.IsMap()) {
        RCLCPP_WARN(get_logger(), "Landmarks file '%s' is not a map.", yaml_file_.c_str());
        return landmarks;
      }
      for (const auto & kv : root) {
        const YAML::Node & seq = kv.second;
        if (!seq.IsSequence() || seq.size() < 5) {
          continue;
        }
        Landmark lm;
        lm.v = seq[0].as<double>();
        lm.w = seq[1].as<double>();
        lm.d = seq[2].as<double>();
        lm.T = seq[3].as<double>();
        lm.psi = seq[4].as<double>();
        if (seq.size() >= 7) {
          lm.x = seq[5].as<double>();
          lm.y = seq[6].as<double>();
          lm.has_xy = true;
        } else {
          lm.x = 0.0;
          lm.y = 0.0;
          lm.has_xy = false;
        }
        landmarks.push_back(lm);
      }
    } catch (const YAML::Exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed loading landmarks file '%s': %s",
        yaml_file_.c_str(), e.what());
    }
    return landmarks;
  }

  std::string action_name_;
  std::string cmd_vel_topic_;
  std::string failure_trigger_topic_;
  std::string odom_topic_;
  std::string yaml_file_;
  double straight_period_;
  double curve_period_;
  double rotate_angular_velocity_;
  double max_angular_velocity_;
  double drive_frequency_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr trigger_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp_action::Server<ExecuteRecovery>::SharedPtr action_server_;

  std::atomic<bool> recovering_;
  std::mutex drive_mutex_;
  std::mutex data_mutex_;

  bool have_odom_;
  double odom_distance_;
  double last_x_;
  double last_y_;
  double latest_yaw_ = 0.0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RecoveryController>());
  rclcpp::shutdown();
  return 0;
}
