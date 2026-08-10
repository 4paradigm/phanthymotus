#include "motus/quest_capture/capture_session.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using motus::quest_capture::BeginCaptureConnection;
using motus::quest_capture::CaptureActionKind;
using motus::quest_capture::CaptureAnswerReceived;
using motus::quest_capture::CaptureAssigned;
using motus::quest_capture::CaptureAssignmentRevoked;
using motus::quest_capture::CaptureAssignmentV1;
using motus::quest_capture::CaptureAuthenticated;
using motus::quest_capture::CaptureChannelState;
using motus::quest_capture::CaptureIdentity;
using motus::quest_capture::CaptureLinkState;
using motus::quest_capture::CaptureOfferCreated;
using motus::quest_capture::CapturePresenceTick;
using motus::quest_capture::CaptureRtcFailed;
using motus::quest_capture::CaptureSessionState;
using motus::quest_capture::CaptureTransportLost;
using motus::quest_capture::CaptureXrEnded;
using motus::quest_capture::CaptureXrFocused;
using motus::quest_capture::FrameConfiguration;
using motus::quest_capture::FrameMode;
using motus::quest_capture::PairingBootstrap;
using motus::quest_capture::PresenceState;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "capture_session_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

CaptureAssignmentV1 Assignment(
    std::int64_t generation = 1,
    std::string session_id = "session-a") {
  return CaptureAssignmentV1{
      .id = "assignment-a",
      .generation = generation,
      .session_id = std::move(session_id),
      .mode = FrameMode::kLive,
      .profile_id = "unitree_g1_23_dual_arm_controller_v1",
      .capability_digest = std::string(64, 'a'),
      .frame_configuration = FrameConfiguration{
          .mode = FrameMode::kLive,
          .base_twist = std::nullopt,
      },
  };
}

CaptureSessionState PairedStandby() {
  CaptureSessionState initial{
      .pairing = PairingBootstrap{
          .pairing_id = "pairing-a",
          .pairing_code = std::string(43, 'p'),
      },
  };
  auto begun = BeginCaptureConnection(initial);
  Check(begun.actions.size() == 1 &&
            begun.actions[0].kind == CaptureActionKind::kSendPairAuthentication,
        "first connection must authenticate with the one-time pairing code");
  auto paired = CaptureAuthenticated(
      begun.state,
      begun.state.connection_epoch,
      CaptureIdentity{
          .capture_id = "capture-a",
          .capture_credential = std::string(43, 'c'),
      },
      true);
  Check(paired.state.identity.has_value() && !paired.state.pairing.has_value(),
        "pairing must become a persistent capture identity");
  Check(paired.actions.size() == 2 &&
            paired.actions[0].kind == CaptureActionKind::kPersistCredential &&
            paired.actions[1].presence == PresenceState::kBrowserReady,
        "fresh credential must persist and immediately start unbound presence");
  return CaptureXrFocused(paired.state, paired.state.connection_epoch).state;
}

CaptureSessionState Streaming() {
  auto standby = PairedStandby();
  auto assigned = CaptureAssigned(
      standby,
      standby.connection_epoch,
      Assignment());
  Check(assigned.state.link == CaptureLinkState::kNegotiating,
        "PC assignment must start negotiation");
  Check(assigned.actions.size() == 2 &&
            assigned.actions[0].presence == PresenceState::kRtcConnecting &&
            assigned.actions[1].kind == CaptureActionKind::kCreateRtcOffer,
        "assignment must emit connecting presence and exactly one RTC start action");
  const auto offered = CaptureOfferCreated(
      assigned.state,
      assigned.state.connection_epoch,
      "assignment-a",
      "v=0\r\na=offer");
  Check(offered.actions.size() == 1 &&
            offered.actions[0].kind == CaptureActionKind::kSendSignalingOffer,
        "gathered SDP must be sent exactly once over capture WSS");
  const auto duplicate_offer = CaptureOfferCreated(
      offered.state,
      offered.state.connection_epoch,
      "assignment-a",
      "v=0\r\na=duplicate");
  Check(duplicate_offer.actions.empty(),
        "duplicate gathering callback must not consume the one-shot Core offer twice");
  auto answered = CaptureAnswerReceived(
      offered.state,
      offered.state.connection_epoch,
      "assignment-a",
      "v=0\r\na=answer");
  Check(answered.state.answer_applied &&
            answered.actions[0].kind == CaptureActionKind::kApplyRtcAnswer,
        "matching answer must be applied");
  auto streaming = CaptureChannelState(
      answered.state,
      answered.state.connection_epoch,
      "assignment-a",
      true,
      true);
  Check(streaming.state.link == CaptureLinkState::kStreaming,
        "both exact data channels must enter streaming");
  Check(streaming.actions.size() == 1 &&
            streaming.actions[0].presence == PresenceState::kStreaming,
        "streaming presence must bind the active assignment");
  return streaming.state;
}

void PcAssignmentIsTheOnlyRtcAuthorityTrigger() {
  const auto standby = PairedStandby();
  Check(standby.link == CaptureLinkState::kStandby && standby.xr_focused,
        "focused native OpenXR must settle in standby");
  const auto tick = CapturePresenceTick(standby, standby.connection_epoch);
  Check(tick.actions.size() == 1 &&
            tick.actions[0].presence == PresenceState::kXrStandby,
        "standby heartbeat must not create RTC or robot authority");
  for (const auto& action : tick.actions) {
    Check(action.kind != CaptureActionKind::kCreateRtcOffer,
          "presence alone must never start Driver RTC");
  }

  auto not_focused = standby;
  not_focused.xr_focused = false;
  const auto background_tick = CapturePresenceTick(
      not_focused, not_focused.connection_epoch);
  Check(background_tick.actions.size() == 1 &&
            background_tick.actions[0].presence == PresenceState::kBrowserReady,
        "authenticated native app must remain observable while XR is not focused");
}

void SocketLossClosesRtcLocallyAndStaleCallbacksAreInert() {
  auto streaming = Streaming();
  streaming.frame_state.deadman_active = true;
  streaming.frame_state.rearm_required = false;
  const auto old_epoch = streaming.connection_epoch;
  const auto lost = CaptureTransportLost(streaming, old_epoch, "wss_disconnected");
  Check(lost.state.link == CaptureLinkState::kOffline && !lost.state.assignment,
        "WSS loss must synchronously discard the local assignment");
  Check(!lost.state.frame_state.deadman_active && lost.state.frame_state.rearm_required,
        "WSS loss must relatch physical deadman before reconnect");
  Check(lost.actions.size() == 2 &&
            lost.actions[0].kind == CaptureActionKind::kCloseRtc &&
            lost.actions[1].kind == CaptureActionKind::kScheduleReconnect,
        "WSS loss must close Driver RTC before reconnecting");

  auto reconnecting = BeginCaptureConnection(lost.state);
  Check(reconnecting.actions[0].kind ==
            CaptureActionKind::kSendCredentialAuthentication,
        "reconnect must use persisted credential, never replay pairing");
  const auto stale = CaptureAnswerReceived(
      reconnecting.state,
      old_epoch,
      "assignment-a",
      "stale-answer");
  Check(stale.actions.empty() &&
            stale.state.connection_epoch == reconnecting.state.connection_epoch,
        "events from an old socket epoch must be ignored");
}

void RtcAndXrLossFailClosed() {
  auto streaming = Streaming();
  const auto failed = CaptureRtcFailed(
      streaming,
      streaming.connection_epoch,
      "assignment-a",
      "pose_channel_closed");
  Check(failed.state.link == CaptureLinkState::kFaulted,
        "RTC failure must latch a local fault");
  Check(failed.actions.size() == 3 &&
            failed.actions[0].kind == CaptureActionKind::kCloseRtc &&
            failed.actions[1].presence == PresenceState::kError,
        "RTC failure must close peer before reporting assignment-bound error");

  streaming = Streaming();
  const auto ended = CaptureXrEnded(
      streaming,
      streaming.connection_epoch,
      "openxr_not_focused");
  Check(!ended.state.assignment && !ended.state.xr_focused,
        "OpenXR focus loss must revoke local streaming state");
  Check(ended.actions.size() == 2 &&
            ended.actions[0].kind == CaptureActionKind::kCloseRtc &&
            ended.actions[1].presence == PresenceState::kXrEnded &&
            ended.actions[1].assignment_id.empty(),
        "XR end must close RTC and use terminal unbound presence contract");
}

void GenerationAndSessionWatermarksPreventReplay() {
  auto streaming = Streaming();
  streaming.frame_state = {
      .next_sequence = 41,
      .clutch_sequence = 3,
      .deadman_active = true,
      .rearm_required = false,
      .last_monotonic_ns = 99,
  };
  const auto revoked = CaptureAssignmentRevoked(
      streaming,
      streaming.connection_epoch,
      "assignment-a",
      "operator_pause");
  Check(revoked.state.frame_state.next_sequence == 41 &&
            revoked.state.frame_state.clutch_sequence == 3 &&
            revoked.state.frame_state.rearm_required,
        "same-session retry must retain sequence watermarks and require regrip");

  auto retry = Assignment(2);
  retry.id = "assignment-b";
  const auto retried = CaptureAssigned(
      revoked.state,
      revoked.state.connection_epoch,
      retry);
  Check(retried.state.frame_state.next_sequence == 41 &&
            retried.state.frame_state.clutch_sequence == 3,
        "same session must not rewind Driver sequence");

  const auto stale = CaptureAssignmentRevoked(
      retried.state,
      retried.state.connection_epoch,
      "assignment-b",
      "retry_failed");
  auto replay = Assignment(1);
  replay.id = "assignment-replay";
  const auto rejected = CaptureAssigned(
      stale.state,
      stale.state.connection_epoch,
      replay);
  Check(rejected.state.link == CaptureLinkState::kFaulted,
        "old assignment generation must fail closed");

  auto next_session_state = stale.state;
  auto next = Assignment(3, "session-b");
  next.id = "assignment-c";
  const auto next_session = CaptureAssigned(
      next_session_state,
      next_session_state.connection_epoch,
      next);
  Check(next_session.state.frame_state.next_sequence == 0 &&
            next_session.state.frame_state.clutch_sequence == 0,
        "a new Core session gets fresh RTC sequence watermarks");
}

}  // namespace

int main() {
  PcAssignmentIsTheOnlyRtcAuthorityTrigger();
  SocketLossClosesRtcLocallyAndStaleCallbacksAreInert();
  RtcAndXrLossFailClosed();
  GenerationAndSessionWatermarksPreventReplay();
  std::cout << "quest-capture session: all host tests passed\n";
  return 0;
}
