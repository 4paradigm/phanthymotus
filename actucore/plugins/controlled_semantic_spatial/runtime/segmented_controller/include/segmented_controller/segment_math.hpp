#ifndef SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_
#define SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace segmented_controller
{

struct SegmentProgress
{
  double along{0.0};
  double lateral{0.0};
  double length{0.0};
  double remaining{0.0};
};

inline double normalize_angle(double value)
{
  return std::atan2(std::sin(value), std::cos(value));
}

inline bool same_goal_endpoint(
  double old_x, double old_y, double old_yaw,
  double new_x, double new_y, double new_yaw,
  double position_epsilon, double yaw_epsilon)
{
  return std::hypot(new_x - old_x, new_y - old_y) <= position_epsilon &&
         std::abs(normalize_angle(new_yaw - old_yaw)) <= yaw_epsilon;
}

inline SegmentProgress segment_progress(
  double start_x, double start_y, double end_x, double end_y,
  double current_x, double current_y)
{
  const double dx = end_x - start_x;
  const double dy = end_y - start_y;
  const double length = std::hypot(dx, dy);
  if (length <= 1e-9) {
    return {0.0, std::hypot(current_x - end_x, current_y - end_y), 0.0, 0.0};
  }
  const double ux = dx / length;
  const double uy = dy / length;
  const double rx = current_x - start_x;
  const double ry = current_y - start_y;
  const double along = rx * ux + ry * uy;
  const double lateral = std::abs(rx * uy - ry * ux);
  return {along, lateral, length, length - along};
}

inline bool segment_reached(
  const SegmentProgress & progress, double distance_to_end, double tolerance)
{
  return distance_to_end <= tolerance ||
         (progress.along >= progress.length && progress.lateral <= tolerance);
}

inline bool segment_crossed(const SegmentProgress & progress)
{
  return progress.length > 1e-9 && progress.along >= progress.length;
}

inline void observe_stopped_cycle(bool stopped, int & stopped_cycles)
{
  stopped_cycles = stopped ? stopped_cycles + 1 : 0;
}

// Check the straight motion commanded along the current body heading. This
// geometrical preview is not a measured braking-distance or dynamics guarantee.
template<typename IsPoseSafe>
bool forward_motion_safe(
  double x, double y, double yaw, double distance, double step, IsPoseSafe is_pose_safe)
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw) ||
    !std::isfinite(distance) || distance < 0.0 || !std::isfinite(step) || step <= 0.0)
  {
    return false;
  }
  const double raw_samples = std::ceil(distance / step);
  // Bound work done while the costmap mutex is held, and fail closed on overflow.
  if (!std::isfinite(raw_samples) || raw_samples > 100000.0) {
    return false;
  }
  const auto samples = static_cast<std::size_t>(std::max(1.0, raw_samples));
  const double dx = std::cos(yaw);
  const double dy = std::sin(yaw);
  for (std::size_t sample = 0; sample <= samples; ++sample) {
    const double traveled = distance * static_cast<double>(sample) / static_cast<double>(samples);
    const double sample_x = x + traveled * dx;
    const double sample_y = y + traveled * dy;
    if (!std::isfinite(sample_x) || !std::isfinite(sample_y) ||
      !is_pose_safe(sample_x, sample_y, yaw))
    {
      return false;
    }
  }
  return true;
}

}  // namespace segmented_controller

#endif  // SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_
