#include "motus/quest_capture/capture_wire.hpp"

#include <cmath>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace motus::quest_capture {
namespace {

using Json = nlohmann::json;

const std::regex kUuidV4(
    "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$");
const std::regex kSafeId("^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$");
const std::regex kAppVersion("^[A-Za-z0-9][A-Za-z0-9_.+-]{0,31}$");
const std::regex kDigest("^[0-9a-f]{64}$");

[[noreturn]] void Invalid(const std::string& field) {
  throw std::invalid_argument("invalid capture wire field: " + field);
}

void ExactKeys(
    const Json& value,
    std::initializer_list<std::string_view> expected,
    const std::string& field) {
  if (!value.is_object() || value.size() != expected.size()) {
    Invalid(field);
  }
  for (const auto key : expected) {
    if (!value.contains(std::string(key))) {
      Invalid(field);
    }
  }
}

std::string StringMatching(
    const Json& value,
    const std::regex& pattern,
    const std::string& field) {
  if (!value.is_string()) {
    Invalid(field);
  }
  const auto result = value.get<std::string>();
  if (!std::regex_match(result, pattern)) {
    Invalid(field);
  }
  return result;
}

std::string BoundedString(
    const Json& value,
    std::size_t minimum,
    std::size_t maximum,
    const std::string& field) {
  if (!value.is_string()) {
    Invalid(field);
  }
  auto result = value.get<std::string>();
  if (result.size() < minimum || result.size() > maximum ||
      result.find('\0') != std::string::npos) {
    Invalid(field);
  }
  return result;
}

std::int64_t SafeInteger(
    const Json& value,
    std::int64_t minimum,
    std::int64_t maximum,
    const std::string& field) {
  if (!value.is_number_integer()) {
    Invalid(field);
  }
  const auto result = value.get<std::int64_t>();
  if (result < minimum || result > maximum) {
    Invalid(field);
  }
  return result;
}

double FiniteNumber(
    const Json& value,
    double minimum,
    double maximum,
    const std::string& field) {
  if (!value.is_number()) {
    Invalid(field);
  }
  const auto result = value.get<double>();
  if (!std::isfinite(result) || result < minimum || result > maximum) {
    Invalid(field);
  }
  return result;
}

bool StrictBoolean(const Json& value, const std::string& field) {
  if (!value.is_boolean()) {
    Invalid(field);
  }
  return value.get<bool>();
}

Json ParseStrict(const std::string& payload) {
  if (payload.empty() || payload.size() > kMaxCaptureWireBytes ||
      payload.find('\0') != std::string::npos) {
    Invalid("message");
  }
  bool duplicate = false;
  std::vector<std::unordered_set<std::string>> object_keys;
  const auto callback = [&duplicate, &object_keys](
                            int,
                            Json::parse_event_t event,
                            Json& parsed) {
    if (event == Json::parse_event_t::object_start) {
      object_keys.emplace_back();
    } else if (event == Json::parse_event_t::key) {
      if (object_keys.empty() ||
          !object_keys.back().insert(parsed.get<std::string>()).second) {
        duplicate = true;
      }
    } else if (event == Json::parse_event_t::object_end) {
      if (object_keys.empty()) {
        duplicate = true;
      } else {
        object_keys.pop_back();
      }
    }
    return true;
  };
  Json result;
  try {
    result = Json::parse(payload, callback, true, true);
  } catch (const Json::exception&) {
    Invalid("message");
  }
  if (duplicate || !object_keys.empty() || !result.is_object()) {
    Invalid("message");
  }
  return result;
}

std::string PresenceName(PresenceState state) {
  switch (state) {
    case PresenceState::kBrowserReady:
      return "browser_ready";
    case PresenceState::kError:
      return "error";
    case PresenceState::kRtcConnecting:
      return "rtc_connecting";
    case PresenceState::kStreaming:
      return "streaming";
    case PresenceState::kXrEnded:
      return "xr_ended";
    case PresenceState::kXrStandby:
      return "xr_standby";
  }
  Invalid("presence.state");
}

PresenceState ParsePresence(const Json& value) {
  if (!value.is_string()) {
    Invalid("presence.state");
  }
  const auto state = value.get<std::string>();
  if (state == "browser_ready") {
    return PresenceState::kBrowserReady;
  }
  if (state == "error") {
    return PresenceState::kError;
  }
  if (state == "rtc_connecting") {
    return PresenceState::kRtcConnecting;
  }
  if (state == "streaming") {
    return PresenceState::kStreaming;
  }
  if (state == "xr_ended") {
    return PresenceState::kXrEnded;
  }
  if (state == "xr_standby") {
    return PresenceState::kXrStandby;
  }
  Invalid("presence.state");
}

AxisBinding ParseAxisBinding(const Json& value, const std::string& field) {
  ExactKeys(value, {"hand", "axis", "scale", "deadzone", "direction"}, field);
  AxisBinding binding;
  const auto hand = BoundedString(value.at("hand"), 4, 5, field + ".hand");
  if (hand == "left") {
    binding.hand = AxisBinding::Hand::kLeft;
  } else if (hand == "right") {
    binding.hand = AxisBinding::Hand::kRight;
  } else {
    Invalid(field + ".hand");
  }
  binding.axis = static_cast<std::size_t>(SafeInteger(
      value.at("axis"), 0, 7, field + ".axis"));
  binding.scale = FiniteNumber(value.at("scale"), 0.0, 10.0, field + ".scale");
  binding.deadzone = FiniteNumber(
      value.at("deadzone"), 0.0, 0.95, field + ".deadzone");
  binding.direction = static_cast<int>(SafeInteger(
      value.at("direction"), -1, 1, field + ".direction"));
  if (binding.direction == 0) {
    Invalid(field + ".direction");
  }
  return binding;
}

FrameConfiguration ParseCapabilities(
    const Json& value,
    const std::string& profile_id,
    FrameMode mode,
    const Json& top_effectors) {
  ExactKeys(
      value,
      {"profile_id", "input_bindings", "outputs", "effectors"},
      "assignment.capabilities");
  if (StringMatching(value.at("profile_id"), kSafeId, "capabilities.profile_id") !=
      profile_id) {
    Invalid("capabilities.profile_id");
  }
  const auto& inputs = value.at("input_bindings");
  const auto& outputs = value.at("outputs");
  const auto& effectors = value.at("effectors");
  if (!inputs.is_object() || inputs.size() > 16 || !outputs.is_object() ||
      outputs.empty() || outputs.size() > 16 || !effectors.is_array() ||
      effectors.size() > 16 || effectors != top_effectors) {
    Invalid("capabilities");
  }

  std::set<std::string> enabled_outputs;
  for (const auto& [name, output] : outputs.items()) {
    if (!std::regex_match(name, kSafeId) || !output.is_object() ||
        output.empty() || output.size() > 2 || !output.contains("enabled")) {
      Invalid("capabilities.outputs");
    }
    for (const auto& [key, ignored] : output.items()) {
      static_cast<void>(ignored);
      if (key != "enabled" && key != "joint_count") {
        Invalid("capabilities.outputs");
      }
    }
    if (output.contains("joint_count")) {
      static_cast<void>(SafeInteger(
          output.at("joint_count"), 0, 128, "capabilities.outputs.joint_count"));
    }
    if (StrictBoolean(output.at("enabled"), "capabilities.outputs.enabled")) {
      enabled_outputs.insert(name);
    }
  }

  std::set<std::string> declared_effectors;
  for (const auto& effector : effectors) {
    declared_effectors.insert(StringMatching(
        effector, kSafeId, "capabilities.effectors"));
  }
  if (declared_effectors.size() != effectors.size() ||
      declared_effectors != enabled_outputs) {
    Invalid("capabilities.effectors");
  }

  for (const auto& [name, binding] : inputs.items()) {
    if (!std::regex_match(name, kSafeId)) {
      Invalid("capabilities.input_bindings");
    }
    if (name == "base_twist") {
      continue;
    }
    ExactKeys(binding, {"required", "role"}, "capabilities.input_bindings.role");
    static_cast<void>(StrictBoolean(
        binding.at("required"), "capabilities.input_bindings.required"));
    static_cast<void>(StringMatching(
        binding.at("role"), kSafeId, "capabilities.input_bindings.role"));
  }

  const bool base_enabled = enabled_outputs.contains("base");
  const bool has_base_binding = inputs.contains("base_twist");
  if (base_enabled != has_base_binding) {
    Invalid("capabilities.input_bindings.base_twist");
  }
  FrameConfiguration configuration{.mode = mode};
  if (has_base_binding) {
    const auto& base = inputs.at("base_twist");
    ExactKeys(base, {"linear_x", "linear_y", "angular_z"}, "base_twist");
    configuration.base_twist = BaseTwistBinding{
        .linear_x = ParseAxisBinding(base.at("linear_x"), "base_twist.linear_x"),
        .linear_y = ParseAxisBinding(base.at("linear_y"), "base_twist.linear_y"),
        .angular_z = ParseAxisBinding(base.at("angular_z"), "base_twist.angular_z"),
    };
  }
  return configuration;
}

CaptureAssignmentV1 ParseAssignment(const Json& value) {
  ExactKeys(
      value,
      {"id", "generation", "session_id", "mode", "profile_id",
       "capability_digest", "capabilities", "effectors", "state",
       "created_at", "updated_at", "failure_code"},
      "assignment");
  if (value.at("state") != "issued" || !value.at("failure_code").is_null()) {
    Invalid("assignment.state");
  }
  static_cast<void>(FiniteNumber(
      value.at("created_at"), 0.0, 1.0e13, "assignment.created_at"));
  static_cast<void>(FiniteNumber(
      value.at("updated_at"), 0.0, 1.0e13, "assignment.updated_at"));
  const auto mode_name = BoundedString(value.at("mode"), 4, 6, "assignment.mode");
  FrameMode mode;
  if (mode_name == "shadow") {
    mode = FrameMode::kShadow;
  } else if (mode_name == "live") {
    mode = FrameMode::kLive;
  } else {
    Invalid("assignment.mode");
  }
  const auto profile_id = StringMatching(
      value.at("profile_id"), kSafeId, "assignment.profile_id");
  if (!value.at("effectors").is_array()) {
    Invalid("assignment.effectors");
  }
  const auto configuration = ParseCapabilities(
      value.at("capabilities"), profile_id, mode, value.at("effectors"));
  return CaptureAssignmentV1{
      .id = StringMatching(value.at("id"), kUuidV4, "assignment.id"),
      .generation = SafeInteger(
          value.at("generation"), 1, kMaxSafeWireInteger, "assignment.generation"),
      .session_id = StringMatching(
          value.at("session_id"), kUuidV4, "assignment.session_id"),
      .mode = mode,
      .profile_id = profile_id,
      .capability_digest = StringMatching(
          value.at("capability_digest"), kDigest, "assignment.capability_digest"),
      .frame_configuration = configuration,
  };
}

void ValidateAuthenticationInputs(
    const std::string& app_version,
    const std::string& id,
    const std::string& secret,
    const std::string& id_field) {
  if (!std::regex_match(app_version, kAppVersion) ||
      !std::regex_match(id, kUuidV4) || secret.size() < 32 || secret.size() > 128 ||
      secret.find('\0') != std::string::npos) {
    Invalid(id_field);
  }
}

}  // namespace

std::string SerializePairAuthentication(
    const PairingBootstrap& pairing,
    const std::string& app_version) {
  ValidateAuthenticationInputs(
      app_version, pairing.pairing_id, pairing.pairing_code, "pairing");
  return Json{
      {"type", "pair"},
      {"pairing_id", pairing.pairing_id},
      {"pairing_code", pairing.pairing_code},
      {"capture_protocol", kCaptureProtocol},
      {"frame_protocol", kFrameProtocol},
      {"client_kind", kNativeClientKind},
      {"app_version", app_version},
  }.dump();
}

std::string SerializeCredentialAuthentication(
    const CaptureIdentity& identity,
    const std::string& app_version) {
  ValidateAuthenticationInputs(
      app_version,
      identity.capture_id,
      identity.capture_credential,
      "capture_identity");
  return Json{
      {"type", "credential"},
      {"capture_id", identity.capture_id},
      {"capture_credential", identity.capture_credential},
      {"capture_protocol", kCaptureProtocol},
      {"frame_protocol", kFrameProtocol},
      {"client_kind", kNativeClientKind},
      {"app_version", app_version},
  }.dump();
}

std::string SerializeCapturePresence(
    PresenceState state,
    const std::string& assignment_id) {
  const bool assignment_bound = state == PresenceState::kError ||
      state == PresenceState::kRtcConnecting || state == PresenceState::kStreaming;
  if (assignment_bound != !assignment_id.empty() ||
      (!assignment_id.empty() && !std::regex_match(assignment_id, kUuidV4))) {
    Invalid("presence.assignment_id");
  }
  return Json{
      {"type", "presence"},
      {"state", PresenceName(state)},
      {"assignment_id", assignment_id.empty() ? Json(nullptr) : Json(assignment_id)},
  }.dump();
}

std::string SerializeCaptureSignalingOffer(
    const std::string& assignment_id,
    const std::string& offer_sdp) {
  if (!std::regex_match(assignment_id, kUuidV4) || offer_sdp.empty() ||
      offer_sdp.size() > kMaxSignalingSdpBytes ||
      offer_sdp.find('\0') != std::string::npos) {
    Invalid("signaling_offer");
  }
  const auto payload = Json{
      {"type", "signaling_offer"},
      {"assignment_id", assignment_id},
      {"offer", {{"type", "offer"}, {"sdp", offer_sdp}}},
  }.dump();
  if (payload.size() > kMaxCaptureWireBytes) {
    Invalid("signaling_offer");
  }
  return payload;
}

CaptureServerMessage ParseCaptureServerMessage(
    const std::string& payload,
    const std::optional<CaptureIdentity>& existing_identity) {
  const auto message = ParseStrict(payload);
  if (!message.contains("type") || !message.at("type").is_string()) {
    Invalid("type");
  }
  const auto type = message.at("type").get<std::string>();
  if (type == "paired") {
    ExactKeys(
        message,
        {"type", "capture_id", "capture_credential", "capture_protocol",
         "frame_protocol", "presence_interval_ms", "presence_timeout_ms"},
        "paired");
    if (message.at("capture_protocol") != kCaptureProtocol ||
        message.at("frame_protocol") != kFrameProtocol) {
      Invalid("paired.protocol");
    }
    return CaptureAuthenticatedMessage{
        .identity = CaptureIdentity{
            .capture_id = StringMatching(
                message.at("capture_id"), kUuidV4, "paired.capture_id"),
            .capture_credential = BoundedString(
                message.at("capture_credential"), 32, 128, "paired.credential"),
        },
        .fresh_credential = true,
        .presence_interval_ms = SafeInteger(
            message.at("presence_interval_ms"), 250, 10'000, "presence_interval_ms"),
        .presence_timeout_ms = SafeInteger(
            message.at("presence_timeout_ms"), 1'000, 30'000, "presence_timeout_ms"),
    };
  }
  if (type == "connected") {
    ExactKeys(
        message,
        {"type", "capture_id", "capture_protocol", "frame_protocol",
         "presence_interval_ms", "presence_timeout_ms"},
        "connected");
    if (!existing_identity || message.at("capture_protocol") != kCaptureProtocol ||
        message.at("frame_protocol") != kFrameProtocol ||
        StringMatching(message.at("capture_id"), kUuidV4, "connected.capture_id") !=
            existing_identity->capture_id) {
      Invalid("connected.identity");
    }
    return CaptureAuthenticatedMessage{
        .identity = *existing_identity,
        .fresh_credential = false,
        .presence_interval_ms = SafeInteger(
            message.at("presence_interval_ms"), 250, 10'000, "presence_interval_ms"),
        .presence_timeout_ms = SafeInteger(
            message.at("presence_timeout_ms"), 1'000, 30'000, "presence_timeout_ms"),
    };
  }
  if (type == "assignment") {
    ExactKeys(message, {"type", "assignment"}, "assignment_message");
    return CaptureAssignmentMessage{.assignment = ParseAssignment(message.at("assignment"))};
  }
  if (type == "signaling_answer") {
    ExactKeys(
        message,
        {"type", "assignment_id", "answer"},
        "signaling_answer");
    const auto& answer = message.at("answer");
    ExactKeys(answer, {"type", "sdp"}, "signaling_answer.answer");
    if (answer.at("type") != "answer") {
      Invalid("signaling_answer.answer.type");
    }
    return CaptureSignalingAnswerMessage{
        .assignment_id = StringMatching(
            message.at("assignment_id"), kUuidV4, "signaling_answer.assignment_id"),
        .answer_sdp = BoundedString(
            answer.at("sdp"), 1, kMaxSignalingSdpBytes, "signaling_answer.sdp"),
    };
  }
  if (type == "assignment_revoked") {
    ExactKeys(
        message,
        {"type", "assignment_id", "reason"},
        "assignment_revoked");
    return CaptureAssignmentRevokedMessage{
        .assignment_id = StringMatching(
            message.at("assignment_id"), kUuidV4, "assignment_revoked.assignment_id"),
        .reason = BoundedString(message.at("reason"), 1, 128, "assignment_revoked.reason"),
    };
  }
  if (type == "capture_revoked" || type == "capture_stale") {
    ExactKeys(message, {"type", "reason"}, type);
    return CaptureTerminalMessage{
        .type = type,
        .reason = BoundedString(message.at("reason"), 1, 128, type + ".reason"),
    };
  }
  if (type == "presence_ack") {
    ExactKeys(message, {"type", "state"}, "presence_ack");
    return CapturePresenceAcknowledgedMessage{
        .state = ParsePresence(message.at("state")),
    };
  }
  if (type == "error") {
    ExactKeys(message, {"type", "code"}, "error");
    return CaptureErrorMessage{
        .code = StringMatching(message.at("code"), kSafeId, "error.code"),
    };
  }
  Invalid("type");
}

}  // namespace motus::quest_capture
