// safety_checks.hpp
//
// Shared NaN / non-normalized quaternion and position guards used by the
// recovery nodes. These prevent the TF_DENORMALIZED_QUATERNION and
// TF_NAN_INPUT errors seen when the odometry source publishes NaN or
// zero-norm quaternions / positions.

#ifndef VSLAM_RECOVERY_CORE__SAFETY_CHECKS_HPP_
#define VSLAM_RECOVERY_CORE__SAFETY_CHECKS_HPP_

#include <cmath>

#include "geometry_msgs/msg/quaternion.hpp"

namespace vslam_recovery_core
{

/// \brief Replace a NaN / zero-norm quaternion with identity, otherwise
///        re-normalize it in place.
inline void sanitize_quaternion(geometry_msgs::msg::Quaternion & q)
{
  const bool has_nan =
    std::isnan(q.x) || std::isnan(q.y) || std::isnan(q.z) || std::isnan(q.w);
  const bool has_inf =
    std::isinf(q.x) || std::isinf(q.y) || std::isinf(q.z) || std::isinf(q.w);

  if (has_nan || has_inf) {
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
    return;
  }

  const double length2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (length2 < 1e-6) {
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
    return;
  }

  const double inv = 1.0 / std::sqrt(length2);
  q.x *= inv;
  q.y *= inv;
  q.z *= inv;
  q.w *= inv;
}

/// \brief True when all position components are finite.
inline bool is_valid_position(double x, double y, double z = 0.0)
{
  return !(std::isnan(x) || std::isnan(y) || std::isnan(z) ||
           std::isinf(x) || std::isinf(y) || std::isinf(z));
}

}  // namespace vslam_recovery_core

#endif  // VSLAM_RECOVERY_CORE__SAFETY_CHECKS_HPP_
