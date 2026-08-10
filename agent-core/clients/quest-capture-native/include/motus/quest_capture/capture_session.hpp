#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "motus/quest_capture/frame_v1.hpp"

namespace motus::quest_capture {

inline constexpr char kCaptureProtocol[] = "motus.teleop.capture.v1";
inline constexpr char kFrameProtocol[] = "motus.teleop.rtc-frame.v1";
inline constexpr char kNativeClientKind[] = "native_openxr";

enum class CaptureLinkState {
  kOffline,
  kAuthenticating,
  kStandby,
  kNegotiating,
  kStreaming,
  kFaulted,
};

enum class PresenceState {
  kBrowserReady,
  kError,
  kRtcConnecting,
  kStreaming,
  kXrEnded,
  kXrStandby,
};

enum class CaptureActionKind {
  kSendPairAuthentication,
  kSendCredentialAuthentication,
  kPersistCredential,
  kSendPresence,
  kCreateRtcOffer,
  kSendSignalingOffer,
  kApplyRtcAnswer,
  kCloseRtc,
  kScheduleReconnect,
  kReportFault,
};

struct CaptureIdentity {
  std::string capture_id;
  std::string capture_credential;
};

struct PairingBootstrap {
  std::string pairing_id;
  std::string pairing_code;
};

struct CaptureAssignmentV1 {
  std::string id;
  std::int64_t generation{0};
  std::string session_id;
  FrameMode mode{FrameMode::kShadow};
  std::string profile_id;
  std::string capability_digest;
  FrameConfiguration frame_configuration;
};

struct CaptureAction {
  CaptureActionKind kind{CaptureActionKind::kReportFault};
  std::int64_t connection_epoch{0};
  std::optional<PresenceState> presence;
  std::string assignment_id;
  std::string value;
};

struct CaptureSessionState {
  CaptureLinkState link{CaptureLinkState::kOffline};
  std::int64_t connection_epoch{0};
  std::optional<CaptureIdentity> identity;
  std::optional<PairingBootstrap> pairing;
  std::optional<CaptureAssignmentV1> assignment;
  std::int64_t highest_assignment_generation{0};
  bool authenticated{false};
  bool xr_focused{false};
  bool control_open{false};
  bool pose_open{false};
  bool offer_sent{false};
  bool answer_applied{false};
  std::string frame_session_id;
  FrameState frame_state;
};

struct CaptureTransition {
  CaptureSessionState state;
  std::vector<CaptureAction> actions;
};

// Start one WSS connection. Authentication is always sent in-band and the
// returned epoch must be supplied to all callbacks so stale sockets are inert.
CaptureTransition BeginCaptureConnection(const CaptureSessionState& previous);

// Apply the exact paired/connected server acknowledgement. A freshly paired
// credential is persisted before it can be used for a later reconnect.
CaptureTransition CaptureAuthenticated(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const CaptureIdentity& identity,
    bool persist_credential);

// The native Activity calls this from the OpenXR session state machine. Merely
// becoming focused never creates robot authority; it only advertises standby.
CaptureTransition CaptureXrFocused(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch);

// A PC-authorized assignment is the only event that may start RTC signaling.
// In particular there is no native action for Acquire or confirm-live.
CaptureTransition CaptureAssigned(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const CaptureAssignmentV1& assignment);

CaptureTransition CaptureOfferCreated(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& offer_sdp);

CaptureTransition CaptureAnswerReceived(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& answer_sdp);

CaptureTransition CaptureChannelState(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    bool control_open,
    bool pose_open);

// Presence timers use this to renew only capture presence. This is not and
// must never become the Core robot-session heartbeat.
CaptureTransition CapturePresenceTick(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch);

// These loss paths close the Quest-to-Driver peer locally before reconnecting
// or notifying Core. The frame deadman is relatched immediately.
CaptureTransition CaptureTransportLost(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& reason);

CaptureTransition CaptureRtcFailed(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& reason);

CaptureTransition CaptureXrEnded(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& reason);

CaptureTransition CaptureAssignmentRevoked(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& reason);

}  // namespace motus::quest_capture
