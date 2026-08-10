#include "motus/quest_capture/frame_v1.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace motus::quest_capture {
namespace {

constexpr std::size_t kMaxAxes = 8;
constexpr std::size_t kMaxButtons = 16;
constexpr double kMaxPositionMetres = 100.0;
constexpr double kQuaternionNormMin = 0.5;
constexpr double kQuaternionNormMax = 1.5;
constexpr double kQuaternionComponentLimit = 1.000001;

enum class SqueezeState {
  kInvalid,
  kReleased,
  kTransition,
  kPressed,
};

struct NormalizedController {
  WireController wire;
  bool valid{false};
  SqueezeState squeeze{SqueezeState::kInvalid};
};

void RequireCounter(std::int64_t value, std::string_view name, bool allow_maximum = true) {
  const auto maximum = allow_maximum ? kMaxSafeWireInteger : kMaxSafeWireInteger - 1;
  if (value < 0 || value > maximum) {
    throw std::range_error(std::string(name) + " is outside the safe wire integer range");
  }
}

bool IsFinite(double value) {
  return std::isfinite(value);
}

double Clamp(double value, double minimum, double maximum) {
  return std::min(maximum, std::max(minimum, value));
}

std::optional<WirePose> NormalizePose(const PoseSample& sample) {
  if (!sample.valid || sample.emulated) {
    return std::nullopt;
  }
  for (double value : sample.position) {
    if (!IsFinite(value) || std::abs(value) > kMaxPositionMetres) {
      return std::nullopt;
    }
  }
  double norm_squared = 0.0;
  for (double value : sample.orientation) {
    if (!IsFinite(value) || std::abs(value) > kQuaternionComponentLimit) {
      return std::nullopt;
    }
    norm_squared += value * value;
  }
  const double norm = std::sqrt(norm_squared);
  if (!IsFinite(norm) || norm < kQuaternionNormMin || norm > kQuaternionNormMax) {
    return std::nullopt;
  }

  WirePose pose;
  pose.position = sample.position;
  double normalized_norm_squared = 0.0;
  for (std::size_t index = 0; index < pose.orientation.size(); ++index) {
    pose.orientation[index] = sample.orientation[index] / norm;
    normalized_norm_squared += pose.orientation[index] * pose.orientation[index];
  }
  const double normalized_norm = std::sqrt(normalized_norm_squared);
  if (!IsFinite(normalized_norm) || normalized_norm == 0.0) {
    return std::nullopt;
  }
  for (double& value : pose.orientation) {
    value = Clamp(value / normalized_norm, -1.0, 1.0);
    if (!IsFinite(value)) {
      return std::nullopt;
    }
  }
  return pose;
}

NormalizedController NormalizeController(const ControllerSample& sample) {
  NormalizedController result;
  result.valid = sample.active && sample.xr_standard && sample.correct_handedness &&
      sample.tracked_pointer && sample.has_grip_space && sample.axes.size() <= kMaxAxes &&
      sample.buttons.size() >= 2 && sample.buttons.size() <= kMaxButtons;

  const auto axis_count = std::min(sample.axes.size(), kMaxAxes);
  result.wire.axes.reserve(axis_count);
  for (std::size_t index = 0; index < axis_count; ++index) {
    const double raw = sample.axes[index];
    result.wire.axes.push_back(IsFinite(raw) ? Clamp(raw, -1.0, 1.0) : 0.0);
    result.valid = result.valid && IsFinite(raw) && raw >= -1.0 && raw <= 1.0;
  }

  const auto button_count = std::min(sample.buttons.size(), kMaxButtons);
  result.wire.buttons.reserve(button_count);
  for (std::size_t index = 0; index < button_count; ++index) {
    const double raw = sample.buttons[index];
    result.wire.buttons.push_back(IsFinite(raw) ? Clamp(raw, 0.0, 1.0) : 0.0);
    result.valid = result.valid && IsFinite(raw) && raw >= 0.0 && raw <= 1.0;
  }

  if (sample.buttons.size() < 2 || !IsFinite(sample.buttons[1]) || sample.buttons[1] < 0.0 ||
      sample.buttons[1] > 1.0) {
    result.squeeze = SqueezeState::kInvalid;
  } else if (sample.squeeze_pressed && sample.buttons[1] >= 0.75) {
    result.squeeze = SqueezeState::kPressed;
  } else if (!sample.squeeze_pressed && sample.buttons[1] < 0.75) {
    result.squeeze = SqueezeState::kReleased;
  } else {
    result.squeeze = SqueezeState::kTransition;
  }
  return result;
}

void ValidateBinding(const AxisBinding& binding) {
  if (binding.axis >= kMaxAxes || !IsFinite(binding.scale) || binding.scale < 0.0 ||
      binding.scale > 10.0 || !IsFinite(binding.deadzone) || binding.deadzone < 0.0 ||
      binding.deadzone > 0.95 || (binding.direction != -1 && binding.direction != 1)) {
    throw std::invalid_argument("base_twist axis binding is invalid");
  }
}

const std::vector<double>& AxesFor(
    const AxisBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  return binding.hand == AxisBinding::Hand::kLeft ? left.wire.axes : right.wire.axes;
}

std::optional<double> BoundAxis(
    const AxisBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  const auto& axes = AxesFor(binding, left, right);
  if (binding.axis >= axes.size()) {
    return std::nullopt;
  }
  return axes[binding.axis];
}

std::optional<double> RemapAxis(double value, double deadzone) {
  if (!IsFinite(value) || value < -1.0 || value > 1.0) {
    return std::nullopt;
  }
  const double magnitude = std::abs(value);
  if (magnitude <= deadzone) {
    return 0.0;
  }
  return std::copysign((magnitude - deadzone) / (1.0 - deadzone), value);
}

bool BoundInputsAvailable(
    const BaseTwistBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  return BoundAxis(binding.linear_x, left, right).has_value() &&
      BoundAxis(binding.linear_y, left, right).has_value() &&
      BoundAxis(binding.angular_z, left, right).has_value();
}

bool BoundInputsNeutral(
    const BaseTwistBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  const auto neutral = [&](const AxisBinding& axis) {
    const auto value = BoundAxis(axis, left, right);
    const auto remapped = value ? RemapAxis(*value, axis.deadzone) : std::nullopt;
    return remapped.has_value() && *remapped == 0.0;
  };
  return neutral(binding.linear_x) && neutral(binding.linear_y) && neutral(binding.angular_z);
}

double ScaledAxis(
    const AxisBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  const auto value = BoundAxis(binding, left, right);
  const auto remapped = value ? RemapAxis(*value, binding.deadzone) : std::nullopt;
  if (!remapped || *remapped == 0.0) {
    return 0.0;
  }
  return *remapped * binding.scale * static_cast<double>(binding.direction);
}

BaseTwist BuildBaseTwist(
    bool deadman,
    bool inputs_valid,
    const BaseTwistBinding& binding,
    const NormalizedController& left,
    const NormalizedController& right) {
  BaseTwist twist;
  if (!deadman || !inputs_valid) {
    return twist;
  }
  twist.linear = {
      ScaledAxis(binding.linear_x, left, right),
      ScaledAxis(binding.linear_y, left, right),
      0.0,
  };
  twist.angular = {0.0, 0.0, ScaledAxis(binding.angular_z, left, right)};
  return twist;
}

std::int64_t NextMonotonicNanoseconds(std::int64_t previous, std::optional<std::int64_t> value) {
  if (value && *value >= 0 && *value <= kMaxSafeWireInteger && *value > previous) {
    return *value;
  }
  if (previous >= kMaxSafeWireInteger) {
    throw std::range_error("client_monotonic_ns exhausted the safe wire integer range");
  }
  return previous + 1;
}

std::string_view ModeName(FrameMode mode) {
  switch (mode) {
    case FrameMode::kShadow:
      return "shadow";
    case FrameMode::kLive:
      return "live";
  }
  throw std::invalid_argument("unknown frame mode");
}

void AppendInteger(std::string& output, std::int64_t value) {
  std::array<char, 32> buffer{};
  const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  if (result.ec != std::errc{}) {
    throw std::runtime_error("failed to serialize integer");
  }
  output.append(buffer.data(), result.ptr);
}

void AppendNumber(std::string& output, double value) {
  if (!IsFinite(value)) {
    throw std::runtime_error("refusing to serialize a non-finite number");
  }
  if (value == 0.0) {
    output.push_back('0');
    return;
  }
  std::array<char, 64> buffer{};
  const auto result = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::general);
  if (result.ec != std::errc{}) {
    throw std::runtime_error("failed to serialize number");
  }
  output.append(buffer.data(), result.ptr);
}

template <std::size_t Size>
void AppendArray(std::string& output, const std::array<double, Size>& values) {
  output.push_back('[');
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output.push_back(',');
    }
    AppendNumber(output, values[index]);
  }
  output.push_back(']');
}

void AppendVector(std::string& output, const std::vector<double>& values) {
  output.push_back('[');
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output.push_back(',');
    }
    AppendNumber(output, values[index]);
  }
  output.push_back(']');
}

void AppendPose(std::string& output, const std::optional<WirePose>& pose) {
  if (!pose) {
    output.append("null");
    return;
  }
  output.append("{\"position\":");
  AppendArray(output, pose->position);
  output.append(",\"orientation\":");
  AppendArray(output, pose->orientation);
  output.push_back('}');
}

void AppendController(std::string& output, const WireController& controller) {
  output.append("{\"axes\":");
  AppendVector(output, controller.axes);
  output.append(",\"buttons\":");
  AppendVector(output, controller.buttons);
  output.push_back('}');
}

}  // namespace

FrameResult BuildFrameV1(
    const FrameState& previous,
    const FrameSample& sample,
    const FrameConfiguration& configuration) {
  RequireCounter(previous.next_sequence, "next_sequence", false);
  RequireCounter(previous.clutch_sequence, "clutch_sequence");
  RequireCounter(previous.last_monotonic_ns, "last_monotonic_ns");
  if (configuration.base_twist) {
    ValidateBinding(configuration.base_twist->linear_x);
    ValidateBinding(configuration.base_twist->linear_y);
    ValidateBinding(configuration.base_twist->angular_z);
  }

  FrameResult result;
  result.frame.sequence = previous.next_sequence;
  result.frame.client_monotonic_ns =
      NextMonotonicNanoseconds(previous.last_monotonic_ns, sample.monotonic_ns);
  result.frame.mode = configuration.mode;
  result.frame.head = NormalizePose(sample.head);
  result.frame.left_controller = NormalizePose(sample.left_controller);
  result.frame.right_controller = NormalizePose(sample.right_controller);
  result.frame.tracking = {
      result.frame.head.has_value(),
      result.frame.left_controller.has_value(),
      result.frame.right_controller.has_value(),
  };

  const auto left = NormalizeController(sample.left_input);
  const auto right = NormalizeController(sample.right_input);
  result.frame.left_input = left.wire;
  result.frame.right_input = right.wire;

  const bool all_tracked = result.frame.tracking.head &&
      result.frame.tracking.left_controller && result.frame.tracking.right_controller;
  const bool squeeze_requested = sample.distinct_input_sources &&
      left.squeeze == SqueezeState::kPressed && right.squeeze == SqueezeState::kPressed;
  const bool base_inputs_available = !configuration.base_twist ||
      BoundInputsAvailable(*configuration.base_twist, left, right);
  const bool inputs_valid = left.valid && right.valid && sample.distinct_input_sources &&
      base_inputs_available;
  const bool explicit_release_observed = inputs_valid &&
      left.squeeze != SqueezeState::kInvalid && right.squeeze != SqueezeState::kInvalid &&
      (left.squeeze == SqueezeState::kReleased || right.squeeze == SqueezeState::kReleased);
  const bool motion_inputs_neutral = !configuration.base_twist ||
      BoundInputsNeutral(*configuration.base_twist, left, right);
  const bool non_neutral_engagement_attempt =
      !previous.deadman_active && squeeze_requested && !motion_inputs_neutral;

  bool rearm_required = previous.rearm_required;
  if (!all_tracked || !inputs_valid) {
    rearm_required = true;
  } else if (explicit_release_observed) {
    rearm_required = !motion_inputs_neutral;
  } else if ((previous.deadman_active && !squeeze_requested) ||
             non_neutral_engagement_attempt) {
    rearm_required = true;
  }

  result.frame.deadman =
      all_tracked && inputs_valid && squeeze_requested && !rearm_required;
  result.frame.clutch_sequence = previous.clutch_sequence;
  if (result.frame.deadman && !previous.deadman_active) {
    if (result.frame.clutch_sequence >= kMaxSafeWireInteger) {
      throw std::range_error("clutch_sequence exhausted the safe wire integer range");
    }
    ++result.frame.clutch_sequence;
  }
  if (configuration.base_twist) {
    result.frame.base_twist = BuildBaseTwist(
        result.frame.deadman,
        inputs_valid,
        *configuration.base_twist,
        left,
        right);
  }

  result.state = {
      previous.next_sequence + 1,
      result.frame.clutch_sequence,
      result.frame.deadman,
      rearm_required,
      result.frame.client_monotonic_ns,
  };
  return result;
}

FrameState MarkDeadmanReleased(const FrameState& previous) {
  RequireCounter(previous.next_sequence, "next_sequence");
  RequireCounter(previous.clutch_sequence, "clutch_sequence");
  RequireCounter(previous.last_monotonic_ns, "last_monotonic_ns");
  FrameState released = previous;
  released.deadman_active = false;
  released.rearm_required = true;
  return released;
}

std::string SerializeFrameV1(const FrameV1& frame) {
  RequireCounter(frame.sequence, "sequence");
  RequireCounter(frame.client_monotonic_ns, "client_monotonic_ns");
  RequireCounter(frame.clutch_sequence, "clutch_sequence");

  std::string output;
  output.reserve(1024);
  output.append("{\"schema_version\":1,\"sequence\":");
  AppendInteger(output, frame.sequence);
  output.append(",\"client_monotonic_ns\":");
  AppendInteger(output, frame.client_monotonic_ns);
  output.append(",\"mode\":\"");
  output.append(ModeName(frame.mode));
  output.append("\",\"deadman\":");
  output.append(frame.deadman ? "true" : "false");
  output.append(",\"clutch_sequence\":");
  AppendInteger(output, frame.clutch_sequence);
  output.append(",\"tracking\":{\"head\":");
  output.append(frame.tracking.head ? "true" : "false");
  output.append(",\"left_controller\":");
  output.append(frame.tracking.left_controller ? "true" : "false");
  output.append(",\"right_controller\":");
  output.append(frame.tracking.right_controller ? "true" : "false");
  output.append("},\"head\":");
  AppendPose(output, frame.head);
  output.append(",\"left_controller\":");
  AppendPose(output, frame.left_controller);
  output.append(",\"right_controller\":");
  AppendPose(output, frame.right_controller);
  output.append(",\"controllers\":{\"left\":");
  AppendController(output, frame.left_input);
  output.append(",\"right\":");
  AppendController(output, frame.right_input);
  output.push_back('}');
  if (frame.base_twist) {
    output.append(",\"base_twist\":{\"linear\":");
    AppendArray(output, frame.base_twist->linear);
    output.append(",\"angular\":");
    AppendArray(output, frame.base_twist->angular);
    output.push_back('}');
  }
  output.push_back('}');
  return output;
}

}  // namespace motus::quest_capture
