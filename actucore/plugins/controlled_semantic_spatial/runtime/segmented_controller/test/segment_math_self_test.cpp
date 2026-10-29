// Keep test assertions active even when this target is built in Release mode.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "segmented_controller/segment_math.hpp"

int main()
{
  using segmented_controller::segment_progress;
  using segmented_controller::segment_crossed;
  using segmented_controller::segment_reached;
  using segmented_controller::same_goal_endpoint;
  using segmented_controller::odometry_fresh;
  using segmented_controller::observe_stopped_sample;
  using segmented_controller::velocity_covariance_valid;

  std::array<double, 36> covariance{};
  assert(!velocity_covariance_valid(covariance));  // Legacy all-zero raw twist.
  for (std::size_t i = 0; i < 6; ++i) {
    covariance[i * 7] = 1.0;
  }
  assert(velocity_covariance_valid(covariance));
  covariance[35] = -1.0;
  assert(!velocity_covariance_valid(covariance));
  covariance[35] = std::numeric_limits<double>::quiet_NaN();
  assert(!velocity_covariance_valid(covariance));

  assert(odometry_fresh(0.2, 0.4, 0.6));
  assert(!odometry_fresh(-0.001, 0.1, 0.6));
  assert(!odometry_fresh(0.61, 0.1, 0.6));
  assert(!odometry_fresh(0.1, 0.61, 0.6));
  assert(!odometry_fresh(0.1, 0.1, 0.0));
  assert(!odometry_fresh(std::numeric_limits<double>::quiet_NaN(), 0.1, 0.6));
  std::int64_t stamp = 10;
  int stopped = 0;
  observe_stopped_sample(true, 10, stamp, stopped);
  assert(stopped == 0);  // A pre-stop sample cannot prove a stop.
  observe_stopped_sample(true, 11, stamp, stopped);
  observe_stopped_sample(true, 11, stamp, stopped);
  assert(stopped == 1);  // One sample cannot count twice at 20 Hz.
  observe_stopped_sample(false, 12, stamp, stopped);
  assert(stopped == 0);
  observe_stopped_sample(true, 13, stamp, stopped);
  observe_stopped_sample(true, 14, stamp, stopped);
  assert(stopped == 2);

  assert(same_goal_endpoint(1.0, 2.0, 3.13, 1.0005, 1.9995, -3.13, 0.001, 0.03));
  assert(!same_goal_endpoint(1.0, 2.0, 0.0, 1.01, 2.0, 0.0, 0.001, 0.001));
  assert(!same_goal_endpoint(1.0, 2.0, 0.0, 1.0, 2.0, 0.01, 0.001, 0.001));

  const auto before = segment_progress(0.0, 0.0, 1.0, 0.0, 0.7, 0.0);
  assert(std::abs(before.remaining - 0.3) < 1e-9);
  assert(!segment_reached(before, 0.3, 0.18));

  const auto near = segment_progress(0.0, 0.0, 1.0, 0.0, 0.85, 0.0);
  assert(segment_reached(near, 0.15, 0.18));

  const auto crossed = segment_progress(0.0, 0.0, 1.0, 0.0, 1.05, 0.05);
  assert(segment_crossed(crossed));
  assert(segment_reached(crossed, std::hypot(0.05, 0.05), 0.18));

  const auto missed = segment_progress(0.0, 0.0, 1.0, 0.0, 1.05, 0.4);
  assert(segment_crossed(missed));
  assert(!segment_reached(missed, std::hypot(0.05, 0.4), 0.18));

  using segmented_controller::forward_motion_safe;
  const double yaw = 0.34;
  const double obstacle_x = 0.5 * std::cos(yaw);
  const double obstacle_y = 0.5 * std::sin(yaw);
  const auto obstacle_free = [=](double x, double y, double) {
      return std::hypot(x - obstacle_x, y - obstacle_y) > 0.02;
    };
  // The old segment heading misses the obstacle, the commanded heading sees it.
  assert(forward_motion_safe(0.0, 0.0, 0.0, 1.0, 0.025, obstacle_free));
  assert(!forward_motion_safe(0.0, 0.0, yaw, 1.0, 0.025, obstacle_free));
  // The opposite yaw and a laterally offset starting pose must use their own
  // actual world-frame ray and preserve the current footprint orientation.
  const auto verify_pose = [=](double x, double y, double theta) {
      assert(std::abs(theta + yaw) < 1e-12);
      assert(std::abs((y - 1.0) + x * std::tan(yaw)) < 1e-12);
      return true;
    };
  assert(forward_motion_safe(0.0, 1.0, -yaw, 1.0, 0.025, verify_pose));
  int checks = 0;
  const auto fail_at_start = [&checks](double, double, double) {
      ++checks;
      return false;
    };
  assert(!forward_motion_safe(0.0, 0.0, 0.0, 1.0, 0.025, fail_at_start));
  assert(checks == 1);
  int total_checks = 0;
  const auto fail_at_end = [&total_checks](double x, double, double) {
      ++total_checks;
      return x < 1.0;
    };
  assert(!forward_motion_safe(0.0, 0.0, 0.0, 1.0, 0.3, fail_at_end));
  assert(total_checks == 5);  // Both endpoints and all three intermediate poses.
  const auto inside_map = [](double x, double, double) {return x <= 0.5;};
  assert(!forward_motion_safe(0.0, 0.0, 0.0, 1.0, 0.025, inside_map));
  const auto safe = [](double, double, double) {return true;};
  assert(forward_motion_safe(0.0, 0.0, 0.0, 0.0, 0.025, safe));
  assert(!forward_motion_safe(0.0, 0.0, 0.0, -1.0, 0.025, safe));
  assert(!forward_motion_safe(0.0, 0.0, 0.0, 1.0, 0.0, safe));
  assert(!forward_motion_safe(0.0, 0.0, 0.0, 1.0, 1e-12, safe));
  assert(!forward_motion_safe(0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0, 0.025, safe));
  assert(!forward_motion_safe(0.0, 0.0, 0.0, std::numeric_limits<double>::infinity(), 0.025, safe));
  return 0;
}
