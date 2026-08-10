#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace motus::quest_capture {

inline constexpr std::int64_t kMaxSafeWireInteger = 9'007'199'254'740'991LL;

enum class FrameMode {
  kShadow,
  kLive,
};

struct PoseSample {
  bool valid{false};
  bool emulated{false};
  std::array<double, 3> position{};
  std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};
};

struct ControllerSample {
  bool active{false};
  bool xr_standard{false};
  bool correct_handedness{false};
  bool tracked_pointer{false};
  bool has_grip_space{false};
  bool squeeze_pressed{false};
  std::vector<double> axes;
  std::vector<double> buttons;
};

struct AxisBinding {
  enum class Hand {
    kLeft,
    kRight,
  };

  Hand hand{Hand::kLeft};
  std::size_t axis{0};
  double scale{0.0};
  double deadzone{0.0};
  int direction{1};
};

struct BaseTwistBinding {
  AxisBinding linear_x;
  AxisBinding linear_y;
  AxisBinding angular_z;
};

struct FrameConfiguration {
  FrameMode mode{FrameMode::kShadow};
  std::optional<BaseTwistBinding> base_twist;
};

struct FrameSample {
  PoseSample head;
  PoseSample left_controller;
  PoseSample right_controller;
  ControllerSample left_input;
  ControllerSample right_input;
  bool distinct_input_sources{true};
  std::optional<std::int64_t> monotonic_ns;
};

struct FrameState {
  std::int64_t next_sequence{0};
  std::int64_t clutch_sequence{0};
  bool deadman_active{false};
  bool rearm_required{true};
  std::int64_t last_monotonic_ns{0};
};

struct WirePose {
  std::array<double, 3> position{};
  std::array<double, 4> orientation{};
};

struct WireController {
  std::vector<double> axes;
  std::vector<double> buttons;
};

struct TrackingState {
  bool head{false};
  bool left_controller{false};
  bool right_controller{false};
};

struct BaseTwist {
  std::array<double, 3> linear{};
  std::array<double, 3> angular{};
};

struct FrameV1 {
  std::int64_t sequence{0};
  std::int64_t client_monotonic_ns{0};
  FrameMode mode{FrameMode::kShadow};
  bool deadman{false};
  std::int64_t clutch_sequence{0};
  TrackingState tracking;
  std::optional<WirePose> head;
  std::optional<WirePose> left_controller;
  std::optional<WirePose> right_controller;
  WireController left_input;
  WireController right_input;
  std::optional<BaseTwist> base_twist;
};

struct FrameResult {
  FrameV1 frame;
  FrameState state;
};

// Validate and reduce one native OpenXR sample into the public
// motus.teleop.rtc-frame.v1 contract. Session authority is deliberately absent.
FrameResult BuildFrameV1(
    const FrameState& previous,
    const FrameSample& sample,
    const FrameConfiguration& configuration);

// Preserve sequence/clutch watermarks across RTC reconnects while forcing a
// physical release-neutral-regrip before motion can resume.
FrameState MarkDeadmanReleased(const FrameState& previous);

// Compact JSON suitable for the unordered teleop-pose DataChannel.
std::string SerializeFrameV1(const FrameV1& frame);

}  // namespace motus::quest_capture
