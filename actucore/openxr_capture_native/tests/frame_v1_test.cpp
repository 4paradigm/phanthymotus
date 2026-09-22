#include "motus/openxr_capture/frame_v1.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using motus::openxr_capture::AxisBinding;
using motus::openxr_capture::BaseTwistBinding;
using motus::openxr_capture::BuildFrameV1;
using motus::openxr_capture::ControllerSample;
using motus::openxr_capture::FrameConfiguration;
using motus::openxr_capture::FrameMode;
using motus::openxr_capture::FrameSample;
using motus::openxr_capture::FrameState;
using motus::openxr_capture::MarkDeadmanReleased;
using motus::openxr_capture::PoseSample;
using motus::openxr_capture::SerializeFrameV1;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "frame_v1_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

void CheckNear(double actual, double expected, const std::string& message) {
  if (std::abs(actual - expected) > 1e-12) {
    Fail(message);
  }
}

PoseSample TrackedPose(double x = 0.0, double y = 1.5, double z = 0.0) {
  return PoseSample{
      .valid = true,
      .emulated = false,
      .position = {x, y, z},
      .orientation = {0.0, 0.0, 0.0, 1.0},
  };
}

ControllerSample Controller(bool pressed, double squeeze_value, double x = 0.0, double y = 0.0) {
  return ControllerSample{
      .active = true,
      .xr_standard = true,
      .correct_handedness = true,
      .tracked_pointer = true,
      .has_grip_space = true,
      .squeeze_pressed = pressed,
      .axes = {0.0, 0.0, x, y},
      .buttons = {0.0, squeeze_value},
  };
}

FrameSample Sample(bool pressed, double squeeze_value) {
  return FrameSample{
      .head = TrackedPose(0.0, 1.6, 0.0),
      .left_controller = TrackedPose(-0.2, 1.2, -0.3),
      .right_controller = TrackedPose(0.2, 1.2, -0.3),
      .left_input = Controller(pressed, squeeze_value),
      .right_input = Controller(pressed, squeeze_value),
      .distinct_input_sources = true,
      .monotonic_ns = 1'000'000,
  };
}

FrameConfiguration ArmOnly(FrameMode mode = FrameMode::kLive) {
  return FrameConfiguration{.mode = mode, .base_twist = std::nullopt};
}

FrameConfiguration WithBase() {
  return FrameConfiguration{
      .mode = FrameMode::kLive,
      .base_twist = BaseTwistBinding{
          .linear_x = AxisBinding{
              .hand = AxisBinding::Hand::kLeft,
              .axis = 3,
              .scale = 0.5,
              .deadzone = 0.2,
              .direction = -1,
          },
          .linear_y = AxisBinding{
              .hand = AxisBinding::Hand::kLeft,
              .axis = 2,
              .scale = 0.3,
              .deadzone = 0.2,
              .direction = -1,
          },
          .angular_z = AxisBinding{
              .hand = AxisBinding::Hand::kRight,
              .axis = 2,
              .scale = 0.6,
              .deadzone = 0.2,
              .direction = -1,
          },
      },
  };
}

void InitialHeldGripCannotArm() {
  const auto held = BuildFrameV1(FrameState{}, Sample(true, 1.0), ArmOnly());
  Check(!held.frame.deadman, "initial held squeeze must not arm");
  Check(held.state.rearm_required, "initial held squeeze must retain rearm latch");
  Check(held.frame.clutch_sequence == 0, "initial held squeeze must not advance clutch");
  Check(held.frame.sequence == 0 && held.state.next_sequence == 1, "sequence must advance once");
}

void ReleaseNeutralRegripArmsAndMapsXrStandardAxes() {
  FrameState state;
  auto released_sample = Sample(false, 0.0);
  const auto released = BuildFrameV1(state, released_sample, WithBase());
  Check(!released.frame.deadman, "release frame must be inactive");
  Check(!released.state.rearm_required, "neutral explicit release must clear rearm latch");

  auto pressed_sample = Sample(true, 1.0);
  pressed_sample.monotonic_ns = 2'000'000;
  const auto pressed = BuildFrameV1(released.state, pressed_sample, WithBase());
  Check(pressed.frame.deadman, "release-neutral-regrip must arm");
  Check(pressed.frame.clutch_sequence == 1, "first deliberate grip must increment clutch");
  Check(pressed.frame.base_twist.has_value(), "base-enabled frame must include base_twist");

  auto motion_sample = Sample(true, 1.0);
  motion_sample.monotonic_ns = 3'000'000;
  motion_sample.left_input.axes = {0.7, -0.6, -1.0, -1.0};
  motion_sample.right_input.axes = {-0.8, 0.9, -1.0, 0.5};
  const auto motion = BuildFrameV1(pressed.state, motion_sample, WithBase());
  Check(motion.frame.deadman, "motion after neutral engagement must remain armed");
  CheckNear(motion.frame.base_twist->linear[0], 0.5, "left axes[3] must drive linear_x");
  CheckNear(motion.frame.base_twist->linear[1], 0.3, "left axes[2] must drive linear_y");
  CheckNear(motion.frame.base_twist->angular[2], 0.6, "right axes[2] must drive angular_z");
}

void ReconnectAndTrackingLossRequireHigherClutch() {
  auto released = BuildFrameV1(FrameState{}, Sample(false, 0.0), ArmOnly());
  auto active_sample = Sample(true, 1.0);
  active_sample.monotonic_ns = 2'000'000;
  auto active = BuildFrameV1(released.state, active_sample, ArmOnly());
  Check(active.frame.deadman && active.frame.clutch_sequence == 1, "test precondition failed");

  const auto reconnect_state = MarkDeadmanReleased(active.state);
  auto held_sample = Sample(true, 1.0);
  held_sample.monotonic_ns = 3'000'000;
  auto held = BuildFrameV1(reconnect_state, held_sample, ArmOnly());
  Check(!held.frame.deadman && held.frame.clutch_sequence == 1,
        "holding squeeze across reconnect must stay inhibited");

  auto second_release_sample = Sample(false, 0.0);
  second_release_sample.monotonic_ns = 4'000'000;
  auto second_release = BuildFrameV1(held.state, second_release_sample, ArmOnly());
  auto second_grip_sample = Sample(true, 1.0);
  second_grip_sample.monotonic_ns = 5'000'000;
  auto second_grip = BuildFrameV1(second_release.state, second_grip_sample, ArmOnly());
  Check(second_grip.frame.deadman && second_grip.frame.clutch_sequence == 2,
        "reconnect must require and prove a higher clutch sequence");

  auto lost_tracking_sample = Sample(true, 1.0);
  lost_tracking_sample.left_controller.valid = false;
  lost_tracking_sample.monotonic_ns = 6'000'000;
  auto lost = BuildFrameV1(second_grip.state, lost_tracking_sample, ArmOnly());
  Check(!lost.frame.deadman && lost.state.rearm_required,
        "tracking loss must release deadman and relatch rearm");
}

void MalformedInputFailsClosedAndMonotonicNeverRegresses() {
  auto malformed = Sample(false, 0.0);
  malformed.left_input.buttons = {0.0};
  malformed.head.orientation = {0.0, 0.0, 0.0, 2.0};
  malformed.monotonic_ns = -4;
  const auto result = BuildFrameV1(
      FrameState{.last_monotonic_ns = 41}, malformed, ArmOnly(FrameMode::kShadow));
  Check(!result.frame.deadman && result.state.rearm_required,
        "malformed input must fail closed");
  Check(!result.frame.head.has_value() && !result.frame.tracking.head,
        "invalid pose must be projected as null and untracked");
  Check(result.frame.client_monotonic_ns == 42,
        "invalid or regressing monotonic sample must advance from watermark");
}

void JsonHasExactPublicEnvelopeAndNoAuthority() {
  auto released = BuildFrameV1(FrameState{}, Sample(false, 0.0), ArmOnly(FrameMode::kShadow));
  const std::string json = SerializeFrameV1(released.frame);
  Check(json.find("\"schema_version\":1") != std::string::npos, "schema_version missing");
  Check(json.find("\"mode\":\"shadow\"") != std::string::npos, "mode missing");
  Check(json.find("\"controllers\":{\"left\":{\"axes\":[0,0,0,0],\"buttons\":[0,0]}") !=
            std::string::npos,
        "xr-standard controller slots must be serialized exactly");
  Check(json.find("session_id") == std::string::npos, "frame must not contain session authority");
  Check(json.find("fence") == std::string::npos, "frame must not contain fence authority");
  Check(json.find("token") == std::string::npos, "frame must not contain credentials");
  Check(json.find("base_twist") == std::string::npos,
        "arm-only capabilities must omit base_twist");
}

void SafeIntegerExhaustionThrows() {
  bool sequence_threw = false;
  try {
    static_cast<void>(BuildFrameV1(
        FrameState{.next_sequence = motus::openxr_capture::kMaxSafeWireInteger},
        Sample(false, 0.0),
        ArmOnly()));
  } catch (const std::range_error&) {
    sequence_threw = true;
  }
  Check(sequence_threw, "exhausted sequence must throw instead of wrapping");

  bool monotonic_threw = false;
  try {
    static_cast<void>(BuildFrameV1(
        FrameState{.last_monotonic_ns = motus::openxr_capture::kMaxSafeWireInteger},
        Sample(false, 0.0),
        ArmOnly()));
  } catch (const std::range_error&) {
    monotonic_threw = true;
  }
  Check(monotonic_threw, "exhausted monotonic watermark must throw instead of wrapping");
}

void TriggerMustBeNeutralBeforeRearm() {
  auto sample=Sample(false,0.0);
  sample.left_input.buttons[0]=0.6;
  auto held=BuildFrameV1(FrameState{},sample,ArmOnly());
  Check(held.state.rearm_required,"non-neutral trigger must block rearm");
  auto neutral=BuildFrameV1(held.state,Sample(false,0.0),ArmOnly());
  auto armed=BuildFrameV1(neutral.state,Sample(true,1.0),ArmOnly());
  Check(armed.frame.deadman,"neutral trigger then regrip enables");
  sample=Sample(true,1.0);sample.right_input.buttons[0]=0.42;
  auto motion=BuildFrameV1(armed.state,sample,ArmOnly());
  Check(motion.frame.deadman,"trigger motion must not release arms");
  CheckNear(motion.frame.right_input.buttons[0],0.42,"continuous trigger preserved");
}

}  // namespace

int main() {
  TriggerMustBeNeutralBeforeRearm();
  InitialHeldGripCannotArm();
  ReleaseNeutralRegripArmsAndMapsXrStandardAxes();
  ReconnectAndTrackingLossRequireHigherClutch();
  MalformedInputFailsClosedAndMonotonicNeverRegresses();
  JsonHasExactPublicEnvelopeAndNoAuthority();
  SafeIntegerExhaustionThrows();
  std::cout << "openxr-capture frame_v1: all host tests passed\n";
  return 0;
}
