#include "motus/quest_capture/capture_wire.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using motus::quest_capture::CaptureAssignmentMessage;
using motus::quest_capture::CaptureAuthenticatedMessage;
using motus::quest_capture::CaptureIdentity;
using motus::quest_capture::CaptureSignalingAnswerMessage;
using motus::quest_capture::FrameMode;
using motus::quest_capture::PairingBootstrap;
using motus::quest_capture::ParseCaptureServerMessage;
using motus::quest_capture::PresenceState;
using motus::quest_capture::SerializeCapturePresence;
using motus::quest_capture::SerializeCaptureSignalingOffer;
using motus::quest_capture::SerializeCredentialAuthentication;
using motus::quest_capture::SerializePairAuthentication;

constexpr char kCaptureId[] = "3f32ff2f-ed55-44d1-8ee3-87f3576c1a6f";
constexpr char kAssignmentId[] = "441b5d6c-ee37-419c-bd00-c4ed0c1962a8";
constexpr char kSessionId[] = "7ad3de66-64f2-4d47-89a2-c8da2940eb97";

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "capture_wire_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

template <typename Callback>
void CheckInvalid(Callback callback, const std::string& message) {
  try {
    callback();
  } catch (const std::invalid_argument&) {
    return;
  }
  Fail(message);
}

std::string AssignmentMessage(bool base = false) {
  const std::string input_bindings = base
      ? R"({"base_twist":{"linear_x":{"hand":"left","axis":3,"scale":0.5,"deadzone":0.2,"direction":-1},"linear_y":{"hand":"left","axis":2,"scale":0.3,"deadzone":0.2,"direction":-1},"angular_z":{"hand":"right","axis":2,"scale":0.6,"deadzone":0.2,"direction":-1}}})"
      : "{}";
  const std::string outputs = base
      ? R"({"dual_arm":{"enabled":true,"joint_count":10},"base":{"enabled":true}})"
      : R"({"dual_arm":{"enabled":true,"joint_count":10}})";
  const std::string effectors = base ? R"(["dual_arm","base"])" : R"(["dual_arm"])";
  return std::string(R"({"type":"assignment","assignment":{"id":")") +
      kAssignmentId + R"(","generation":1,"session_id":")" + kSessionId +
      R"(","mode":"live","profile_id":"unitree_g1_23_dual_arm_controller_v1","capability_digest":")" +
      std::string(64, 'a') + R"(","capabilities":{"profile_id":"unitree_g1_23_dual_arm_controller_v1","input_bindings":)" +
      input_bindings + R"(,"outputs":)" + outputs + R"(,"effectors":)" + effectors +
      R"(},"effectors":)" + effectors +
      R"(,"state":"issued","created_at":1000,"updated_at":1001,"failure_code":null}})";
}

void AuthenticationNeverMovesSecretsIntoUrls() {
  const PairingBootstrap pairing{
      .pairing_id = "f922333d-88f7-42a9-b140-56fbf7f670f3",
      .pairing_code = std::string(43, 'p'),
  };
  const auto pair = SerializePairAuthentication(pairing, "0.1.0");
  Check(pair.find("motus.teleop.capture.v1") != std::string::npos,
        "pair auth must bind capture protocol");
  Check(pair.find("native_openxr") != std::string::npos,
        "pair auth must identify native OpenXR client");

  const CaptureIdentity identity{
      .capture_id = kCaptureId,
      .capture_credential = std::string(43, 'c'),
  };
  const auto credential = SerializeCredentialAuthentication(identity, "0.1.0");
  Check(credential.find(identity.capture_credential) != std::string::npos,
        "credential belongs only in the in-band first message");
}

void AuthAndAssignmentAreStrictlyProjected() {
  const auto paired = ParseCaptureServerMessage(
      std::string(R"({"type":"paired","capture_id":")") + kCaptureId +
      R"(","capture_credential":")" + std::string(43, 'c') +
      R"(","capture_protocol":"motus.teleop.capture.v1","frame_protocol":"motus.teleop.rtc-frame.v1","presence_interval_ms":2000,"presence_timeout_ms":5000})");
  const auto& authenticated = std::get<CaptureAuthenticatedMessage>(paired);
  Check(authenticated.fresh_credential &&
            authenticated.identity.capture_id == kCaptureId &&
            authenticated.presence_interval_ms == 2000,
        "paired response must expose exact persisted identity and timers");

  const auto connected = ParseCaptureServerMessage(
      std::string(R"({"type":"connected","capture_id":")") + kCaptureId +
      R"(","capture_protocol":"motus.teleop.capture.v1","frame_protocol":"motus.teleop.rtc-frame.v1","presence_interval_ms":2000,"presence_timeout_ms":5000})",
      authenticated.identity);
  Check(!std::get<CaptureAuthenticatedMessage>(connected).fresh_credential,
        "reconnect acknowledgement must reuse rather than echo credential");

  const auto assignment = std::get<CaptureAssignmentMessage>(
      ParseCaptureServerMessage(AssignmentMessage())).assignment;
  Check(assignment.id == kAssignmentId && assignment.session_id == kSessionId &&
            assignment.mode == FrameMode::kLive &&
            !assignment.frame_configuration.base_twist.has_value(),
        "arm-only assignment must map to exact native frame configuration");

  const auto base_assignment = std::get<CaptureAssignmentMessage>(
      ParseCaptureServerMessage(AssignmentMessage(true))).assignment;
  Check(base_assignment.frame_configuration.base_twist.has_value() &&
            base_assignment.frame_configuration.base_twist->linear_x.axis == 3,
        "descriptor-provided base mapping must survive strict projection");
}

void SignalingAndPresenceUseExactPublicEnvelopes() {
  const auto standby = SerializeCapturePresence(PresenceState::kXrStandby, {});
  Check(standby == R"({"assignment_id":null,"state":"xr_standby","type":"presence"})",
        "standby presence must be explicitly unbound");
  const auto streaming = SerializeCapturePresence(
      PresenceState::kStreaming, kAssignmentId);
  Check(streaming.find(kAssignmentId) != std::string::npos,
        "streaming presence must bind assignment");
  CheckInvalid(
      [] { static_cast<void>(SerializeCapturePresence(PresenceState::kStreaming, {})); },
      "assignment-bound presence without id must fail closed");

  const auto offer = SerializeCaptureSignalingOffer(kAssignmentId, "v=0\r\na=offer");
  Check(offer.find("signaling_offer") != std::string::npos &&
            offer.find("pairing") == std::string::npos,
        "SDP message must not contain enrollment authority");
  const auto answer = ParseCaptureServerMessage(
      std::string(R"({"type":"signaling_answer","assignment_id":")") +
      kAssignmentId + R"(","answer":{"type":"answer","sdp":"v=0\r\na=answer"}})");
  Check(std::get<CaptureSignalingAnswerMessage>(answer).answer_sdp == "v=0\r\na=answer",
        "answer SDP must parse for matching state-machine binding");
}

void AmbiguousOrContradictoryJsonFailsClosed() {
  CheckInvalid(
      [] {
        static_cast<void>(ParseCaptureServerMessage(
            R"({"type":"error","type":"presence_ack","code":"bad"})"));
      },
      "duplicate JSON keys must be rejected");
  CheckInvalid(
      [] {
        static_cast<void>(ParseCaptureServerMessage(
            R"({"type":"error","code":"bad","extra":true})"));
      },
      "unknown fields must be rejected");

  auto contradictory = AssignmentMessage();
  const auto marker = contradictory.find(R"("effectors":["dual_arm"],"state")");
  Check(marker != std::string::npos, "test fixture marker missing");
  contradictory.replace(
      marker,
      std::string(R"("effectors":["dual_arm"])" ).size(),
      R"("effectors":[])" );
  CheckInvalid(
      [&contradictory] { static_cast<void>(ParseCaptureServerMessage(contradictory)); },
      "top-level and capability effectors must agree");
}

}  // namespace

int main() {
  AuthenticationNeverMovesSecretsIntoUrls();
  AuthAndAssignmentAreStrictlyProjected();
  SignalingAndPresenceUseExactPublicEnvelopes();
  AmbiguousOrContradictoryJsonFailsClosed();
  std::cout << "quest-capture wire: all host tests passed\n";
  return 0;
}
