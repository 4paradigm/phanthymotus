#ifndef SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_
#define SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_

#include <cmath>

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

}  // namespace segmented_controller

#endif  // SEGMENTED_CONTROLLER__SEGMENT_MATH_HPP_
