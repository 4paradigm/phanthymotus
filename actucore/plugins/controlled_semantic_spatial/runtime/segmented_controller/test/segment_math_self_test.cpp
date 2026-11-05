// Keep test assertions active even when this target is built in Release mode.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>

#include "segmented_controller/segment_math.hpp"

int main()
{
  using segmented_controller::segment_progress;
  using segmented_controller::segment_crossed;
  using segmented_controller::segment_reached;
  using segmented_controller::same_goal_endpoint;
  using segmented_controller::observe_stopped_cycle;

  int stopped = 0;
  observe_stopped_cycle(true, stopped);
  assert(stopped == 1);
  observe_stopped_cycle(false, stopped);
  assert(stopped == 0);  // Motion breaks the consecutive sequence.
  observe_stopped_cycle(true, stopped);
  observe_stopped_cycle(true, stopped);
  assert(stopped == 2);  // Two cycles may use the same velocity sample.
  observe_stopped_cycle(false, stopped);
  assert(stopped == 0);

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
