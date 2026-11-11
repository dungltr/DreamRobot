// landmark_sampler_client.cpp
//
// Adaptive Shannon-Nyquist path landmark sampler (service client side of the
// recovery-layer architecture).
//
// It subscribes to /odom and tracks the last four consecutive yaw angles. The
// segment type (straight / curved) is decided with the paper's rule:
//   - all three consecutive differences positive, or
//   - all three consecutive differences negative, or
//   - mixed signs with at least one difference magnitude in [pi, 2*pi]
//     => curved segment;
//   - otherwise the segment is straight.
//
// Straight segments are sampled at f = 0.5 Hz (T = 2.0 s) and curved segments
// at f = 1.0 Hz (T = 1.0 s), per the Shannon-Nyquist sampling guidance. Every
// sampling period T an asynchronous RecordLandmark request is sent to the
// recovery recorder service.

#include <chrono>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "recovery_interfaces/srv/record_landmark.hpp"
#include "vslam_recovery_core/safety_checks.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

class LandmarkSamplerClient : public rclcpp::Node
{
public:
  using RecordLandmark = recovery_interfaces::srv::RecordLandmark;
  using RecordLandmarkClient = rclcpp::Client<RecordLandmark>;

  LandmarkSamplerClient()
  : Node("landmark_sampler_client"),
    current_period_(2.0),
    elapsed_since_sample_(0.0),
    latest_yaw_(0.0)
  {
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom");
    service_name_ = declare_parameter<std::string>("service_name", "record_landmark");
    const double straight_frequency = declare_parameter<double>("straight_frequency", 0.5);
    const double curve_frequency = declare_parameter<double>("curve_frequency", 1.0);
    monitor_period_ = declare_parameter<double>("monitor_period", 0.1);
    window_size_ = declare_parameter<int>("window_size", 4);
    yaw_deadband_ = declare_parameter<double>("yaw_deadband", 1e-9);

    straight_period_ = (straight_frequency > 1e-6) ? (1.0 / straight_frequency) : 2.0;
    curve_period_ = (curve_frequency > 1e-6) ? (1.0 / curve_frequency) : 1.0;
    current_period_ = straight_period_;

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&LandmarkSamplerClient::odom_callback, this, std::placeholders::_1));

    client_ = create_client<RecordLandmark>(service_name_);

    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(monitor_period_));
    monitor_timer_ = create_wall_timer(
      period_ns, std::bind(&LandmarkSamplerClient::monitor_tick, this));

    RCLCPP_INFO(get_logger(),
      "LandmarkSamplerClient ready. odom='%s' service='%s' "
      "straight_f=%.2f Hz (T=%.2f s) curve_f=%.2f Hz (T=%.2f s)",
      odom_topic_.c_str(), service_name_.c_str(),
      straight_frequency, straight_period_, curve_frequency, curve_period_);
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Fix TF_DENORMALIZED_QUATERNION / TF_NAN_INPUT: sanitize the quaternion
    // before converting it to the yaw angle.
    geometry_msgs::msg::Quaternion q = msg->pose.pose.orientation;
    vslam_recovery_core::sanitize_quaternion(q);
    double yaw = quaternion_to_yaw(q);
    if (!std::isfinite(yaw)) {
      yaw = 0.0;
    }

    latest_yaw_ = yaw;
    have_new_odom_ = true;
    yaw_window_.push_back(yaw);
    while (static_cast<int>(yaw_window_.size()) > window_size_) {
      yaw_window_.pop_front();
    }
  }

  bool is_curved_segment() const
  {
    if (static_cast<int>(yaw_window_.size()) < window_size_) {
      return false;
    }

    const std::vector<double> yaws(yaw_window_.begin(), yaw_window_.end());
    const double d0 = yaws[1] - yaws[0];
    const double d1 = yaws[2] - yaws[1];
    const double d2 = yaws[3] - yaws[2];

    // Mixed signs: still a turn when at least one difference magnitude falls
    // in the wrap-around range [pi, 2*pi]. A small deadband on the sign test
    // (yaw_deadband_) keeps floating-point yaw noise (~1e-19 rad) on a straight
    // segment from spuriously classifying the segment as curved.
    const bool all_positive =
      (d0 > yaw_deadband_) && (d1 > yaw_deadband_) && (d2 > yaw_deadband_);
    const bool all_negative =
      (d0 < -yaw_deadband_) && (d1 < -yaw_deadband_) && (d2 < -yaw_deadband_);
    if (all_positive || all_negative) {
      return true;
    }

    // Mixed signs: still a turn when at least one difference magnitude falls
    // in the wrap-around range [pi, 2*pi].
    const auto in_large_range = [](double d) {
      const double a = std::fabs(d);
      return (a >= kPi) && (a <= 2.0 * kPi);
    };
    return in_large_range(d0) || in_large_range(d1) || in_large_range(d2);
  }

  void monitor_tick()
  {
    const bool curved = is_curved_segment();
    const double new_period = curved ? curve_period_ : straight_period_;

    if (std::fabs(new_period - current_period_) > 1e-9) {
      current_period_ = new_period;
      elapsed_since_sample_ = 0.0;
      RCLCPP_INFO(get_logger(),
        "Switched to %s segment, sampling period T=%.2f s (f=%.2f Hz)",
        curved ? "curved" : "straight", current_period_, 1.0 / current_period_);
    }

    elapsed_since_sample_ += monitor_period_;
    if (elapsed_since_sample_ >= current_period_) {
      elapsed_since_sample_ = 0.0;
      // Only record a landmark when fresh odometry arrived since the previous
      // sample. If the robot has stopped or the odometry stream ended there is
      // no new path data, and re-sampling the cached values would store
      // duplicate landmarks.
      if (!have_new_odom_) {
        return;
      }
      have_new_odom_ = false;
      send_sample_request();
    }
  }

  void send_sample_request()
  {
    if (!client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Service '%s' not ready yet; skipping landmark sample.",
        service_name_.c_str());
      return;
    }

    auto request = std::make_shared<RecordLandmark::Request>();
    request->theta = latest_yaw_;
    request->frequency = 1.0 / current_period_;

    client_->async_send_request(
      request,
      [this](rclcpp::Client<RecordLandmark>::SharedFuture future) {
        const auto response = future.get();
        if (response && response->ret) {
          RCLCPP_INFO(get_logger(),
            "Landmark sampled and recorded (T=%.2f s, psi=%.3f rad)",
            current_period_, latest_yaw_);
        } else {
          RCLCPP_WARN(get_logger(), "Landmark recording request failed.");
        }
      });
  }

  std::string odom_topic_;
  std::string service_name_;
  double monitor_period_;
  int window_size_;
  double yaw_deadband_;
  double straight_period_;
  double curve_period_;
  double current_period_;
  double elapsed_since_sample_;
  double latest_yaw_;
  bool have_new_odom_ = false;

  std::deque<double> yaw_window_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Client<RecordLandmark>::SharedPtr client_;
  rclcpp::TimerBase::SharedPtr monitor_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkSamplerClient>());
  rclcpp::shutdown();
  return 0;
}
