#include "motus/quest_capture/capture_session.hpp"

#include <stdexcept>
#include <utility>

namespace motus::quest_capture {
namespace {

bool CurrentEpoch(const CaptureSessionState& state, std::int64_t epoch) {
  return epoch > 0 && epoch == state.connection_epoch;
}

void RequireText(const std::string& value, const char* name) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(name) + " is required");
  }
}

CaptureTransition Unchanged(const CaptureSessionState& previous) {
  return CaptureTransition{.state = previous};
}

CaptureAction Action(
    CaptureActionKind kind,
    std::int64_t epoch,
    std::string assignment_id = {},
    std::string value = {}) {
  return CaptureAction{
      .kind = kind,
      .connection_epoch = epoch,
      .assignment_id = std::move(assignment_id),
      .value = std::move(value),
  };
}

CaptureAction Presence(
    std::int64_t epoch,
    PresenceState presence,
    std::string assignment_id = {}) {
  return CaptureAction{
      .kind = CaptureActionKind::kSendPresence,
      .connection_epoch = epoch,
      .presence = presence,
      .assignment_id = std::move(assignment_id),
  };
}

void ReleaseLocalMotion(CaptureSessionState& state) {
  state.frame_state = MarkDeadmanReleased(state.frame_state);
  state.control_open = false;
  state.pose_open = false;
  state.offer_sent = false;
  state.answer_applied = false;
}

void ClearAssignment(CaptureSessionState& state) {
  state.assignment.reset();
  ReleaseLocalMotion(state);
}

bool AssignmentMatches(
    const CaptureSessionState& state,
    const std::string& assignment_id) {
  return state.assignment && state.assignment->id == assignment_id;
}

CaptureTransition Faulted(
    const CaptureSessionState& previous,
    const std::string& code) {
  CaptureTransition transition{.state = previous};
  ReleaseLocalMotion(transition.state);
  transition.state.link = CaptureLinkState::kFaulted;
  transition.actions.push_back(Action(
      CaptureActionKind::kReportFault,
      previous.connection_epoch,
      previous.assignment ? previous.assignment->id : std::string{},
      code));
  return transition;
}

}  // namespace

CaptureTransition BeginCaptureConnection(const CaptureSessionState& previous) {
  CaptureTransition transition{.state = previous};
  if (previous.connection_epoch >= kMaxSafeWireInteger) {
    throw std::range_error("connection_epoch exhausted the safe wire integer range");
  }
  if (!previous.identity && !previous.pairing) {
    return Faulted(previous, "capture_enrollment_missing");
  }
  if (previous.link != CaptureLinkState::kOffline &&
      previous.link != CaptureLinkState::kFaulted) {
    transition.actions.push_back(Action(
        CaptureActionKind::kCloseRtc,
        previous.connection_epoch,
        previous.assignment ? previous.assignment->id : std::string{},
        "connection_replaced"));
  }
  ClearAssignment(transition.state);
  ++transition.state.connection_epoch;
  transition.state.authenticated = false;
  transition.state.link = CaptureLinkState::kAuthenticating;
  const auto kind = transition.state.identity
      ? CaptureActionKind::kSendCredentialAuthentication
      : CaptureActionKind::kSendPairAuthentication;
  transition.actions.push_back(Action(kind, transition.state.connection_epoch));
  return transition;
}

CaptureTransition CaptureAuthenticated(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const CaptureIdentity& identity,
    bool persist_credential) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      previous.link != CaptureLinkState::kAuthenticating) {
    return Unchanged(previous);
  }
  RequireText(identity.capture_id, "capture_id");
  RequireText(identity.capture_credential, "capture_credential");
  if (previous.identity && previous.identity->capture_id != identity.capture_id) {
    return Faulted(previous, "capture_identity_mismatch");
  }

  CaptureTransition transition{.state = previous};
  transition.state.identity = identity;
  transition.state.pairing.reset();
  transition.state.authenticated = true;
  transition.state.link = CaptureLinkState::kStandby;
  if (persist_credential) {
    transition.actions.push_back(Action(
        CaptureActionKind::kPersistCredential,
        connection_epoch,
        {},
        identity.capture_id));
  }
  transition.actions.push_back(Presence(
      connection_epoch,
      transition.state.xr_focused
          ? PresenceState::kXrStandby
          : PresenceState::kBrowserReady));
  return transition;
}

CaptureTransition CaptureXrFocused(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch) {
  if (!CurrentEpoch(previous, connection_epoch)) {
    return Unchanged(previous);
  }
  CaptureTransition transition{.state = previous};
  transition.state.xr_focused = true;
  if (previous.authenticated && previous.link == CaptureLinkState::kStandby) {
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kXrStandby));
  }
  return transition;
}

CaptureTransition CaptureAssigned(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const CaptureAssignmentV1& assignment) {
  if (!CurrentEpoch(previous, connection_epoch) || !previous.authenticated) {
    return Unchanged(previous);
  }
  RequireText(assignment.id, "assignment.id");
  RequireText(assignment.session_id, "assignment.session_id");
  RequireText(assignment.profile_id, "assignment.profile_id");
  RequireText(assignment.capability_digest, "assignment.capability_digest");
  if (assignment.generation <= previous.highest_assignment_generation) {
    return Faulted(previous, "capture_assignment_stale");
  }
  if (previous.link != CaptureLinkState::kStandby || !previous.xr_focused ||
      previous.assignment) {
    return Faulted(previous, "capture_assignment_unexpected");
  }
  if (assignment.frame_configuration.mode != assignment.mode) {
    return Faulted(previous, "capture_assignment_mode_mismatch");
  }

  CaptureTransition transition{.state = previous};
  transition.state.assignment = assignment;
  transition.state.highest_assignment_generation = assignment.generation;
  transition.state.link = CaptureLinkState::kNegotiating;
  ReleaseLocalMotion(transition.state);
  if (transition.state.frame_session_id != assignment.session_id) {
    transition.state.frame_session_id = assignment.session_id;
    transition.state.frame_state = FrameState{};
  }
  transition.actions.push_back(Presence(
      connection_epoch,
      PresenceState::kRtcConnecting,
      assignment.id));
  transition.actions.push_back(Action(
      CaptureActionKind::kCreateRtcOffer,
      connection_epoch,
      assignment.id));
  return transition;
}

CaptureTransition CaptureOfferCreated(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& offer_sdp) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      previous.link != CaptureLinkState::kNegotiating ||
      !AssignmentMatches(previous, assignment_id) || previous.offer_sent) {
    return Unchanged(previous);
  }
  RequireText(offer_sdp, "offer_sdp");
  CaptureTransition transition{.state = previous};
  transition.state.offer_sent = true;
  transition.actions.push_back(Action(
      CaptureActionKind::kSendSignalingOffer,
      connection_epoch,
      assignment_id,
      offer_sdp));
  return transition;
}

CaptureTransition CaptureAnswerReceived(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& answer_sdp) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      previous.link != CaptureLinkState::kNegotiating ||
      !AssignmentMatches(previous, assignment_id) || !previous.offer_sent ||
      previous.answer_applied) {
    return Unchanged(previous);
  }
  RequireText(answer_sdp, "answer_sdp");
  CaptureTransition transition{.state = previous};
  transition.state.answer_applied = true;
  transition.actions.push_back(Action(
      CaptureActionKind::kApplyRtcAnswer,
      connection_epoch,
      assignment_id,
      answer_sdp));
  return transition;
}

CaptureTransition CaptureChannelState(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    bool control_open,
    bool pose_open) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      !AssignmentMatches(previous, assignment_id) ||
      (previous.link != CaptureLinkState::kNegotiating &&
       previous.link != CaptureLinkState::kStreaming)) {
    return Unchanged(previous);
  }
  CaptureTransition transition{.state = previous};
  transition.state.control_open = control_open;
  transition.state.pose_open = pose_open;
  if (control_open && pose_open && previous.answer_applied) {
    transition.state.link = CaptureLinkState::kStreaming;
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kStreaming,
        assignment_id));
  } else if (previous.link == CaptureLinkState::kStreaming) {
    return CaptureRtcFailed(
        previous,
        connection_epoch,
        assignment_id,
        "capture_data_channel_lost");
  }
  return transition;
}

CaptureTransition CapturePresenceTick(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch) {
  if (!CurrentEpoch(previous, connection_epoch) || !previous.authenticated) {
    return Unchanged(previous);
  }
  CaptureTransition transition{.state = previous};
  if (previous.link == CaptureLinkState::kStreaming && previous.assignment) {
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kStreaming,
        previous.assignment->id));
  } else if (previous.link == CaptureLinkState::kNegotiating && previous.assignment) {
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kRtcConnecting,
        previous.assignment->id));
  } else if (previous.link == CaptureLinkState::kStandby) {
    transition.actions.push_back(Presence(
        connection_epoch,
        previous.xr_focused
            ? PresenceState::kXrStandby
            : PresenceState::kBrowserReady));
  }
  return transition;
}

CaptureTransition CaptureTransportLost(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& reason) {
  if (!CurrentEpoch(previous, connection_epoch)) {
    return Unchanged(previous);
  }
  RequireText(reason, "reason");
  CaptureTransition transition{.state = previous};
  const std::string assignment_id = previous.assignment
      ? previous.assignment->id
      : std::string{};
  ClearAssignment(transition.state);
  transition.state.authenticated = false;
  transition.state.link = CaptureLinkState::kOffline;
  transition.actions.push_back(Action(
      CaptureActionKind::kCloseRtc,
      connection_epoch,
      assignment_id,
      reason));
  transition.actions.push_back(Action(
      CaptureActionKind::kScheduleReconnect,
      connection_epoch,
      {},
      reason));
  return transition;
}

CaptureTransition CaptureRtcFailed(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& reason) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      !AssignmentMatches(previous, assignment_id)) {
    return Unchanged(previous);
  }
  RequireText(reason, "reason");
  CaptureTransition transition{.state = previous};
  ReleaseLocalMotion(transition.state);
  transition.state.link = CaptureLinkState::kFaulted;
  transition.actions.push_back(Action(
      CaptureActionKind::kCloseRtc,
      connection_epoch,
      assignment_id,
      reason));
  transition.actions.push_back(Presence(
      connection_epoch,
      PresenceState::kError,
      assignment_id));
  transition.actions.push_back(Action(
      CaptureActionKind::kReportFault,
      connection_epoch,
      assignment_id,
      reason));
  return transition;
}

CaptureTransition CaptureXrEnded(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& reason) {
  if (!CurrentEpoch(previous, connection_epoch)) {
    return Unchanged(previous);
  }
  RequireText(reason, "reason");
  CaptureTransition transition{.state = previous};
  const std::string assignment_id = previous.assignment
      ? previous.assignment->id
      : std::string{};
  ClearAssignment(transition.state);
  transition.state.xr_focused = false;
  transition.state.link = previous.authenticated
      ? CaptureLinkState::kStandby
      : CaptureLinkState::kOffline;
  transition.actions.push_back(Action(
      CaptureActionKind::kCloseRtc,
      connection_epoch,
      assignment_id,
      reason));
  if (previous.authenticated) {
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kXrEnded));
  }
  return transition;
}

CaptureTransition CaptureAssignmentRevoked(
    const CaptureSessionState& previous,
    std::int64_t connection_epoch,
    const std::string& assignment_id,
    const std::string& reason) {
  if (!CurrentEpoch(previous, connection_epoch) ||
      !AssignmentMatches(previous, assignment_id)) {
    return Unchanged(previous);
  }
  RequireText(reason, "reason");
  CaptureTransition transition{.state = previous};
  ClearAssignment(transition.state);
  transition.state.link = previous.authenticated
      ? CaptureLinkState::kStandby
      : CaptureLinkState::kOffline;
  transition.actions.push_back(Action(
      CaptureActionKind::kCloseRtc,
      connection_epoch,
      assignment_id,
      reason));
  if (previous.authenticated && previous.xr_focused) {
    transition.actions.push_back(Presence(
        connection_epoch,
        PresenceState::kXrStandby));
  }
  return transition;
}

}  // namespace motus::quest_capture
