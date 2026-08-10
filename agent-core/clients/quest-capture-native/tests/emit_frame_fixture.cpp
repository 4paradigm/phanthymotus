#include "motus/quest_capture/frame_v1.hpp"

#include <iostream>

namespace {

using motus::quest_capture::BuildFrameV1;
using motus::quest_capture::ControllerSample;
using motus::quest_capture::FrameConfiguration;
using motus::quest_capture::FrameMode;
using motus::quest_capture::FrameSample;
using motus::quest_capture::FrameState;
using motus::quest_capture::PoseSample;
using motus::quest_capture::SerializeFrameV1;

PoseSample Pose(double x, double y, double z) {
  return PoseSample{
      .valid = true,
      .emulated = false,
      .position = {x, y, z},
      .orientation = {0.0, 0.0, 0.0, 1.0},
  };
}

ControllerSample Controller(bool pressed) {
  return ControllerSample{
      .active = true,
      .xr_standard = true,
      .correct_handedness = true,
      .tracked_pointer = true,
      .has_grip_space = true,
      .squeeze_pressed = pressed,
      .axes = {0.0, 0.0, 0.0, 0.0},
      .buttons = {0.0, pressed ? 1.0 : 0.0},
  };
}

FrameSample Sample(bool pressed, std::int64_t monotonic_ns) {
  return FrameSample{
      .head = Pose(0.0, 1.6, 0.0),
      .left_controller = Pose(-0.2, 1.2, -0.3),
      .right_controller = Pose(0.2, 1.2, -0.3),
      .left_input = Controller(pressed),
      .right_input = Controller(pressed),
      .distinct_input_sources = true,
      .monotonic_ns = monotonic_ns,
  };
}

}  // namespace

int main() {
  const FrameConfiguration configuration{
      .mode = FrameMode::kLive,
      .base_twist = std::nullopt,
  };
  const auto released = BuildFrameV1(FrameState{}, Sample(false, 1'000'000), configuration);
  const auto active = BuildFrameV1(released.state, Sample(true, 2'000'000), configuration);
  std::cout << SerializeFrameV1(active.frame) << '\n';
  return 0;
}
