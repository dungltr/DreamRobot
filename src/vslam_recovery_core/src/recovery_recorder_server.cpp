// recovery_recorder_server.cpp
//
// ROS 2 Service Server for the RecordLandmark service.
//
// It subscribes to /odom (nav_msgs/msg/Odometry), converts the pose
// quaternion to the Euler yaw angle and, upon receiving a service request,
// extracts the lightweight path variables M = {v, w, d, T, psi, x, y}
// (the landmark tuple of the manuscript) and appends them to a human readable
// YAML file (path_landmarks.yaml) using yaml-cpp.
//
//   v   <- odom linear velocity (twist.twist.linear.x)
//   w   <- odom angular velocity (twist.twist.angular.z)
//   d   <- Euclidean distance to the previously recorded landmark
//   T   <- sampling period = 1 / request->frequency
//   psi <- request->theta (monitored yaw angle)
//   x   <- recorded position x
//   y   <- recorded position y

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "recovery_interfaces/srv/record_landmark.hpp"
#include "yaml-cpp/yaml.h"
#include "vslam_recovery_core/safety_checks.hpp"

namespace
{

double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  // Standard conversion from a quaternion to the yaw angle (around z).
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

class RecoveryRecorderServer : public rclcpp::Node
{
public:
  RecoveryRecorderServer()
  : Node("recovery_recorder_server"),
    have_odom_(false),
    latest_yaw_(0.0),
    have_last_(false),
    last_x_(0.0),
    last_y_(0.0)
  {
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    service_name_ = declare_parameter<std::string>("service_name", "record_landmark");
    yaml_file_ = declare_parameter<std::string>("yaml_file", "path_landmarks.yaml");

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&RecoveryRecorderServer::odom_callback, this, std::placeholders::_1));

    service_ = create_service<recovery_interfaces::srv::RecordLandmark>(
      service_name_,
      std::bind(&RecoveryRecorderServer::record_callback, this,
        std::placeholders::_1, std::placeholders::_2));

    // Resume from a previous recording session when the file already exists.
    load_landmarks();

    RCLCPP_INFO(get_logger(),
      "RecoveryRecorderServer ready. service='%s' odom='%s' yaml='%s'",
      service_name_.c_str(), odom_topic_.c_str(), yaml_file_.c_str());
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

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_odom_ = *msg;

    // Fix TF_DENORMALIZED_QUATERNION / TF_NAN_INPUT: sanitize the quaternion
    // and guard against NaN position before storing the odometry.
    vslam_recovery_core::sanitize_quaternion(latest_odom_.pose.pose.orientation);
    if (!vslam_recovery_core::is_valid_position(
        latest_odom_.pose.pose.position.x,
        latest_odom_.pose.pose.position.y,
        latest_odom_.pose.pose.position.z))
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "NaN position received from odom; zeroing position.");
      latest_odom_.pose.pose.position.x = 0.0;
      latest_odom_.pose.pose.position.y = 0.0;
      latest_odom_.pose.pose.position.z = 0.0;
    }

    have_odom_ = true;
    latest_yaw_ = quaternion_to_yaw(latest_odom_.pose.pose.orientation);
  }

  static std::string landmark_key(std::size_t index)
  {
    // Zero padded keys keep YAML map iteration in numeric order, which makes
    // the stored landmarks naturally sequential and simple to read back in
    // reverse during recovery.
    std::ostringstream oss;
    oss << "landmark_" << std::setw(4) << std::setfill('0') << index;
    return oss.str();
  }

  void load_landmarks()
  {
    try {
      YAML::Node root = YAML::LoadFile(yaml_file_);
      if (!root.IsMap()) {
        RCLCPP_WARN(get_logger(), "Existing landmarks file is not a map; starting empty.");
        return;
      }
      landmarks_.clear();
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
        // Newer recordings also store the absolute (x, y) position so the
        // recovery controller can follow the path back geometrically instead
        // of relying on an open-loop time replay.
        if (seq.size() >= 7) {
          lm.x = seq[5].as<double>();
          lm.y = seq[6].as<double>();
          lm.has_xy = true;
        } else {
          lm.x = 0.0;
          lm.y = 0.0;
          lm.has_xy = false;
        }
        landmarks_.push_back(lm);
      }
      RCLCPP_INFO(get_logger(),
        "Loaded %zu existing landmark(s) from %s", landmarks_.size(), yaml_file_.c_str());
    } catch (const YAML::Exception & e) {
      // The file may simply not exist yet on a fresh mission.
      RCLCPP_INFO(get_logger(), "No existing landmarks loaded: %s", e.what());
    }
  }

  bool write_yaml()
  {
    try {
      YAML::Emitter out;
      out << YAML::BeginMap;
      for (std::size_t i = 0; i < landmarks_.size(); ++i) {
        const Landmark & lm = landmarks_[i];
        out << YAML::Key << landmark_key(i);
        out << YAML::Value << YAML::Flow << YAML::BeginSeq
            << lm.v << lm.w << lm.d << lm.T << lm.psi
            << lm.x << lm.y
            << YAML::EndSeq;
      }
      out << YAML::EndMap;

      if (!out.good()) {
        RCLCPP_ERROR(get_logger(), "YAML emitter error: %s", out.GetLastError().c_str());
        return false;
      }

      std::ofstream fout(yaml_file_);
      if (!fout.is_open()) {
        RCLCPP_ERROR(get_logger(), "Cannot open YAML file for writing: %s", yaml_file_.c_str());
        return false;
      }
      fout << out.c_str();
      fout.close();
      return true;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed writing YAML file: %s", e.what());
      return false;
    }
  }

  void record_callback(
    const std::shared_ptr<recovery_interfaces::srv::RecordLandmark::Request> request,
    std::shared_ptr<recovery_interfaces::srv::RecordLandmark::Response> response)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (!have_odom_) {
      RCLCPP_WARN(get_logger(), "No odometry received yet; cannot record landmark.");
      response->ret = false;
      return;
    }

    const double v = latest_odom_.twist.twist.linear.x;
    const double w = latest_odom_.twist.twist.angular.z;
    const double x = latest_odom_.pose.pose.position.x;
    const double y = latest_odom_.pose.pose.position.y;

    double d = 0.0;
    if (have_last_) {
      d = std::hypot(x - last_x_, y - last_y_);
    }

    // Skip duplicate / stationary samples that carry no new path information
    // (e.g. a second sampler client, or the robot standing still). This keeps
    // the reversed back-travel from replaying useless zero-distance segments.
    if (have_last_ && d < 1e-4) {
      response->ret = true;
      return;
    }

    last_x_ = x;
    last_y_ = y;
    have_last_ = true;

    const double frequency = request->frequency;
    const double T = (frequency > 1e-6) ? (1.0 / frequency) : 0.0;
    const double psi = request->theta;

    Landmark lm{v, w, d, T, psi, x, y, true};
    landmarks_.push_back(lm);

    if (!write_yaml()) {
      landmarks_.pop_back();
      response->ret = false;
      return;
    }

    response->ret = true;
    RCLCPP_INFO(get_logger(),
      "Recorded landmark #%zu: v=%.3f m/s, w=%.3f rad/s, d=%.3f m, T=%.3f s, "
      "psi=%.3f rad, x=%.3f m, y=%.3f m",
      landmarks_.size() - 1, v, w, d, T, psi, x, y);
  }

  std::string odom_topic_;
  std::string service_name_;
  std::string yaml_file_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Service<recovery_interfaces::srv::RecordLandmark>::SharedPtr service_;

  std::mutex data_mutex_;
  nav_msgs::msg::Odometry latest_odom_;
  bool have_odom_;
  double latest_yaw_;

  bool have_last_;
  double last_x_;
  double last_y_;

  std::vector<Landmark> landmarks_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RecoveryRecorderServer>());
  rclcpp::shutdown();
  return 0;
}
