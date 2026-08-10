#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

#include "motus/quest_capture/capture_session.hpp"

namespace motus::quest_capture {

inline constexpr std::size_t kMaxCaptureWireBytes = 128 * 1024;
inline constexpr std::size_t kMaxSignalingSdpBytes = 120 * 1024;
inline constexpr std::size_t kMaxRtcMessageBytes = 64 * 1024;

struct CaptureAuthenticatedMessage {
  CaptureIdentity identity;
  bool fresh_credential{false};
  std::int64_t presence_interval_ms{0};
  std::int64_t presence_timeout_ms{0};
};

struct CaptureAssignmentMessage {
  CaptureAssignmentV1 assignment;
};

struct CaptureSignalingAnswerMessage {
  std::string assignment_id;
  std::string answer_sdp;
};

struct CaptureAssignmentRevokedMessage {
  std::string assignment_id;
  std::string reason;
};

struct CaptureTerminalMessage {
  std::string type;
  std::string reason;
};

struct CapturePresenceAcknowledgedMessage {
  PresenceState state{PresenceState::kXrStandby};
};

struct CaptureErrorMessage {
  std::string code;
};

using CaptureServerMessage = std::variant<
    CaptureAuthenticatedMessage,
    CaptureAssignmentMessage,
    CaptureSignalingAnswerMessage,
    CaptureAssignmentRevokedMessage,
    CaptureTerminalMessage,
    CapturePresenceAcknowledgedMessage,
    CaptureErrorMessage>;

std::string SerializePairAuthentication(
    const PairingBootstrap& pairing,
    const std::string& app_version);

std::string SerializeCredentialAuthentication(
    const CaptureIdentity& identity,
    const std::string& app_version);

std::string SerializeCapturePresence(
    PresenceState state,
    const std::string& assignment_id);

std::string SerializeCaptureSignalingOffer(
    const std::string& assignment_id,
    const std::string& offer_sdp);

// existing_identity is required for the credential reconnect acknowledgement,
// because Core deliberately does not echo the long-lived credential.
CaptureServerMessage ParseCaptureServerMessage(
    const std::string& payload,
    const std::optional<CaptureIdentity>& existing_identity = std::nullopt);

}  // namespace motus::quest_capture
