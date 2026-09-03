#include <cassert>
#include <cmath>

#include "g1_segmented_controller/segment_math.hpp"

int main()
{
  using g1_segmented_controller::segment_progress;
  using g1_segmented_controller::segment_crossed;
  using g1_segmented_controller::segment_reached;
  using g1_segmented_controller::same_goal_endpoint;

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
  return 0;
}
