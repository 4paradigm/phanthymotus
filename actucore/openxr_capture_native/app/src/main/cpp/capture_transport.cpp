#include "capture_transport.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>

#include "motus/openxr_capture/capture_wire.hpp"

namespace motus::openxr_capture {
namespace {

constexpr std::size_t kPoseBackpressureBytes = 16 * 1024;
constexpr auto kPeerPingInterval = std::chrono::seconds(1);
constexpr auto kMaximumReconnectDelay = std::chrono::milliseconds(5'000);
constexpr auto kRtcNegotiationTimeout = std::chrono::seconds(10);

bool SafeWebSocketUrl(const std::string& value) {
  return value.size() >= 8 && value.size() <= 2'048 && value.starts_with("wss://") &&
      value.find('?') == std::string::npos && value.find('#') == std::string::npos &&
      value.find('@') == std::string::npos &&
      value.ends_with("/ws/teleop-capture");
}

}  // namespace

CaptureTransport::CaptureTransport(
    CaptureTransportConfiguration configuration,
    IdentitySink identity_sink,
    LogSink log_sink)
    : configuration_(std::move(configuration)),
      identity_sink_(std::move(identity_sink)),
      log_sink_(std::move(log_sink)) {
  if (!SafeWebSocketUrl(configuration_.websocket_url) ||
      configuration_.app_version.empty() ||
      configuration_.ca_certificate_pem.size() > 32 * 1024 ||
      configuration_.ca_certificate_pem.find("-----BEGIN CERTIFICATE-----") ==
          std::string::npos ||
      (!configuration_.identity && !configuration_.pairing)) {
    throw std::invalid_argument("capture transport configuration is invalid");
  }
  state_.identity = configuration_.identity;
  state_.pairing = configuration_.pairing;
}

CaptureTransport::~CaptureTransport() {
  Shutdown();
}

void CaptureTransport::Start() {
  if (started_ || shutting_down_) {
    return;
  }
  if (weak_from_this().expired()) {
    throw std::logic_error("capture transport must have shared ownership");
  }
  started_ = true;
  rtc::InitLogger(rtc::LogLevel::Warning);
  ConnectNow();
}

void CaptureTransport::Shutdown() {
  if (shutting_down_.exchange(true)) {
    return;
  }
  CloseRtc("capture_shutdown");
  CloseWebSocket();
  {
    std::lock_guard lock(event_mutex_);
    events_.clear();
  }
  static_cast<void>(rtc::Cleanup().wait_for(std::chrono::seconds(5)));
}

void CaptureTransport::Tick() {
  if (!started_ || shutting_down_) {
    return;
  }
  std::deque<Event> pending;
  {
    std::lock_guard lock(event_mutex_);
    pending.swap(events_);
  }
  for (const auto& event : pending) {
    HandleEvent(event);
  }

  const auto now = std::chrono::steady_clock::now();
  if (state_.link == CaptureLinkState::kOffline && now >= next_reconnect_) {
    ConnectNow();
    return;
  }
  if (state_.authenticated && now >= next_presence_) {
    Apply(CapturePresenceTick(state_, state_.connection_epoch));
    next_presence_ = now + presence_interval_;
  }
  if (
      state_.link == CaptureLinkState::kNegotiating && state_.assignment &&
      now >= rtc_negotiation_deadline_) {
    Apply(CaptureRtcFailed(
        state_,
        state_.connection_epoch,
        state_.assignment->id,
        "rtc_negotiation_timeout"));
    return;
  }
  if (streaming() && now >= next_ping_) {
    SendPeerPing();
    next_ping_ = now + kPeerPingInterval;
  }
}

void CaptureTransport::SetXrFocused(bool focused) {
  if (!started_ || shutting_down_ || focused == state_.xr_focused) {
    return;
  }
  if (focused) {
    Apply(CaptureXrFocused(state_, state_.connection_epoch));
  } else {
    Apply(CaptureXrEnded(
        state_, state_.connection_epoch, "openxr_focus_lost"));
  }
}

void CaptureTransport::SendFrame(const FrameSample& sample) {
  if (!streaming() || !state_.assignment || !pose_ || !pose_->isOpen()) {
    return;
  }
  if (pose_->bufferedAmount() > kPoseBackpressureBytes) {
    Apply(CaptureRtcFailed(
        state_,
        state_.connection_epoch,
        state_.assignment->id,
        "pose_backpressure"));
    return;
  }
  try {
    auto result = BuildFrameV1(
        state_.frame_state,
        sample,
        state_.assignment->frame_configuration);
    state_.frame_state = result.state;
    if (!pose_->send(SerializeFrameV1(result.frame))) {
      Apply(CaptureRtcFailed(
          state_,
          state_.connection_epoch,
          state_.assignment->id,
          "pose_send_buffered"));
    }
  } catch (const std::exception& error) {
    Apply(CaptureRtcFailed(
        state_,
        state_.connection_epoch,
        state_.assignment->id,
        std::string("pose_frame_invalid:") + error.what()));
  }
}

void CaptureTransport::ConnectNow() {
  if (shutting_down_) {
    return;
  }
  Apply(BeginCaptureConnection(state_));
  const auto epoch = state_.connection_epoch;
  rtc::WebSocket::Configuration websocket_configuration;
  websocket_configuration.disableTlsVerification = false;
  websocket_configuration.connectionTimeout = std::chrono::seconds(5);
  websocket_configuration.pingInterval = std::chrono::seconds(2);
  websocket_configuration.maxOutstandingPings = 2;
  websocket_configuration.maxMessageSize = kMaxCaptureWireBytes;
  // libdatachannel's Mbed TLS backend does not import Android's system trust
  // store. Pin the deployment CA content explicitly and keep verification on.
  websocket_configuration.caCertificatePemFile =
      configuration_.ca_certificate_pem;
  websocket_ = std::make_shared<rtc::WebSocket>(websocket_configuration);
  const std::weak_ptr<CaptureTransport> weak_self = weak_from_this();
  websocket_->onOpen([weak_self, epoch] {
    const auto self = weak_self.lock();
    if (!self) {
      return;
    }
    self->Enqueue(Event{
        .kind = EventKind::kWebSocketOpen,
        .connection_epoch = epoch,
    });
  });
  websocket_->onMessage([weak_self, epoch](rtc::message_variant message) {
    const auto self = weak_self.lock();
    if (!self) {
      return;
    }
    if (!std::holds_alternative<std::string>(message)) {
      self->Enqueue(Event{
          .kind = EventKind::kWebSocketError,
          .connection_epoch = epoch,
          .value = "capture_binary_message_rejected",
      });
      return;
    }
    self->Enqueue(Event{
        .kind = EventKind::kWebSocketMessage,
        .connection_epoch = epoch,
        .value = std::get<std::string>(std::move(message)),
    });
  });
  websocket_->onClosed([weak_self, epoch] {
    const auto self = weak_self.lock();
    if (!self) {
      return;
    }
    self->Enqueue(Event{
        .kind = EventKind::kWebSocketClosed,
        .connection_epoch = epoch,
        .value = "capture_wss_closed",
    });
  });
  websocket_->onError([weak_self, epoch](std::string error) {
    const auto self = weak_self.lock();
    if (!self) {
      return;
    }
    self->Enqueue(Event{
        .kind = EventKind::kWebSocketError,
        .connection_epoch = epoch,
        .value = std::move(error),
    });
  });
  websocket_->open(configuration_.websocket_url);
}

void CaptureTransport::Enqueue(Event event) {
  std::lock_guard lock(event_mutex_);
  if (!shutting_down_) {
    events_.push_back(std::move(event));
  }
}

void CaptureTransport::HandleEvent(const Event& event) {
  if (event.connection_epoch != state_.connection_epoch) {
    return;
  }
  switch (event.kind) {
    case EventKind::kWebSocketOpen:
      if (pending_authentication_) {
        const auto action = *pending_authentication_;
        pending_authentication_.reset();
        ApplyAction(action);
      }
      break;
    case EventKind::kWebSocketMessage:
      HandleServerMessage(event.value);
      break;
    case EventKind::kWebSocketClosed:
    case EventKind::kWebSocketError:
      Apply(CaptureTransportLost(
          state_,
          state_.connection_epoch,
          event.value.empty() ? "capture_wss_lost" : event.value));
      break;
    case EventKind::kRtcOfferReady:
      Apply(CaptureOfferCreated(
          state_,
          state_.connection_epoch,
          event.assignment_id,
          event.value));
      break;
    case EventKind::kRtcStateFailed:
      Apply(CaptureRtcFailed(
          state_,
          state_.connection_epoch,
          event.assignment_id,
          event.value));
      break;
    case EventKind::kControlOpened:
    case EventKind::kControlClosed:
    case EventKind::kPoseOpened:
    case EventKind::kPoseClosed: {
      const bool control_open = event.kind == EventKind::kControlOpened
          ? true
          : event.kind == EventKind::kControlClosed ? false : state_.control_open;
      const bool pose_open = event.kind == EventKind::kPoseOpened
          ? true
          : event.kind == EventKind::kPoseClosed ? false : state_.pose_open;
      Apply(CaptureChannelState(
          state_,
          state_.connection_epoch,
          event.assignment_id,
          control_open,
          pose_open));
      break;
    }
  }
}

void CaptureTransport::SendOperatorCommand(const std::string& action) {
  if(!operator_enabled() || (action!="start" && action!="finish" && action!="stop"))return;
  const auto id=operator_connection_+":"+std::to_string(++operator_sequence_);
  SendWebSocket(nlohmann::json{{"type","operator_command"},{"connection_id",operator_connection_},
    {"request_id",id},{"action",action}}.dump());
  operator_state_="pending";operator_error_.clear();
}

void CaptureTransport::HandleServerMessage(const std::string& payload) {
  try {
    auto wire = nlohmann::json::parse(payload);
    if(wire.contains("operator_control")){
      operator_connection_=wire.at("operator_control").value("connection_id",std::string{});
      wire.erase("operator_control");
    }
    if(wire.value("type",std::string{})=="operator_result"){
      if(state_.authenticated){operator_state_=wire.value("state",std::string{});operator_error_=wire.value("error",std::string{});}
      return;
    }
    if (wire.value("type", std::string{}) == "visualization") {
      if (state_.authenticated) {
        try {
          const auto& visual=wire.at("visualization");
          if(visual.contains("operator")){
            const auto& op=visual.at("operator");operator_state_=op.value("state",std::string{});
            operator_mode_=op.value("mode",std::string{});
            operator_error_=op.contains("error") && op["error"].is_string()?op["error"].get<std::string>():"";
          }
          UpdateIkVisualization(visualization_, visual);
        }
        catch (const std::exception&) { Log("ik_visualization_invalid"); }
      }
      return; // Display updates never enter authority or heartbeat state machines.
    }
    if (wire.value("type", std::string{}) == "presence_ack" && wire.contains("visualization")) {
      try { UpdateIkVisualization(visualization_, wire.at("visualization")); }
      catch (const std::exception&) { visualization_ = {}; Log("ik_visualization_invalid"); }
      wire.erase("visualization");
    }
    const auto message = ParseCaptureServerMessage(wire.dump(), state_.identity);
    std::visit([this](const auto& item) {
      using Message = std::decay_t<decltype(item)>;
      if constexpr (std::is_same_v<Message, CaptureAuthenticatedMessage>) {
        if (item.presence_timeout_ms <= item.presence_interval_ms) {
          throw std::invalid_argument("presence timeout must exceed interval");
        }
        presence_interval_ = std::chrono::milliseconds(item.presence_interval_ms);
        next_presence_ = std::chrono::steady_clock::now() + presence_interval_;
        reconnect_delay_ = std::chrono::milliseconds(250);
        Apply(CaptureAuthenticated(
            state_,
            state_.connection_epoch,
            item.identity,
            item.fresh_credential));
      } else if constexpr (std::is_same_v<Message, CaptureAssignmentMessage>) {
        Apply(CaptureAssigned(
            state_, state_.connection_epoch, item.assignment));
      } else if constexpr (std::is_same_v<Message, CaptureSignalingAnswerMessage>) {
        Apply(CaptureAnswerReceived(
            state_,
            state_.connection_epoch,
            item.assignment_id,
            item.answer_sdp));
      } else if constexpr (std::is_same_v<Message, CaptureAssignmentRevokedMessage>) {
        Apply(CaptureAssignmentRevoked(
            state_,
            state_.connection_epoch,
            item.assignment_id,
            item.reason));
      } else if constexpr (std::is_same_v<Message, CaptureTerminalMessage>) {
        Apply(CaptureTransportLost(
            state_, state_.connection_epoch, item.reason));
        if (item.type == "capture_revoked") {
          next_reconnect_ = std::chrono::steady_clock::time_point::max();
        }
      } else if constexpr (std::is_same_v<Message, CapturePresenceAcknowledgedMessage>) {
        static_cast<void>(item);
      } else if constexpr (std::is_same_v<Message, CaptureErrorMessage>) {
        // The Driver intentionally closes WSS after a signaling error. Close
        // locally now so headset-to-Driver RTC cannot outlive that decision.
        Apply(CaptureTransportLost(
            state_, state_.connection_epoch, item.code));
        if (CaptureErrorStopsReconnect(item.code)) {
          next_reconnect_ = std::chrono::steady_clock::time_point::max();
        }
        CloseWebSocket();
      }
    }, message);
  } catch (const std::exception& error) {
    Apply(CaptureTransportLost(
        state_,
        state_.connection_epoch,
        std::string("capture_wire_rejected:") + error.what()));
    CloseWebSocket();
  }
}

void CaptureTransport::Apply(CaptureTransition transition) {
  state_ = std::move(transition.state);
  for (const auto& action : transition.actions) {
    ApplyAction(action);
  }
}

void CaptureTransport::ApplyAction(const CaptureAction& action) {
  if (action.connection_epoch != state_.connection_epoch &&
      action.kind != CaptureActionKind::kCloseRtc) {
    return;
  }
  switch (action.kind) {
    case CaptureActionKind::kSendPairAuthentication:
    case CaptureActionKind::kSendCredentialAuthentication:
      if (!websocket_ || !websocket_->isOpen()) {
        pending_authentication_ = action;
        return;
      }
      if (action.kind == CaptureActionKind::kSendPairAuthentication) {
        if (!state_.pairing) {
          throw std::logic_error("pair authentication requested without pairing");
        }
        SendWebSocket(SerializePairAuthentication(
            *state_.pairing, configuration_.app_version));
      } else {
        if (!state_.identity) {
          throw std::logic_error("credential authentication requested without identity");
        }
        SendWebSocket(SerializeCredentialAuthentication(
            *state_.identity, configuration_.app_version));
      }
      break;
    case CaptureActionKind::kPersistCredential:
      if (state_.identity && identity_sink_) {
        try {
          identity_sink_(*state_.identity);
          configuration_.identity = state_.identity;
          configuration_.pairing.reset();
        } catch (...) {
          // The server has already consumed the one-time pairing and enabled
          // this credential.  Continuing only in memory would look healthy
          // until the next app restart.  Forget both identities and disable
          // reconnect so the failure is immediately visible and requires an
          // explicit new pairing.
          credential_persistence_failed_ = true;
          state_.identity.reset();
          state_.pairing.reset();
          configuration_.identity.reset();
          configuration_.pairing.reset();
          throw;
        }
      }
      break;
    case CaptureActionKind::kSendPresence:
      if (!action.presence) {
        throw std::logic_error("presence action is missing state");
      }
      SendWebSocket(SerializeCapturePresence(
          *action.presence, action.assignment_id));
      break;
    case CaptureActionKind::kCreateRtcOffer:
      StartRtc(action.assignment_id);
      break;
    case CaptureActionKind::kSendSignalingOffer:
      SendWebSocket(SerializeCaptureSignalingOffer(
          action.assignment_id, action.value));
      break;
    case CaptureActionKind::kApplyRtcAnswer:
      if (!peer_) {
        throw std::logic_error("RTC answer received without peer");
      }
      peer_->setRemoteDescription(rtc::Description(action.value, "answer"));
      break;
    case CaptureActionKind::kCloseRtc:
      CloseRtc(action.value);
      break;
    case CaptureActionKind::kScheduleReconnect:
      ScheduleReconnect(action.value);
      break;
    case CaptureActionKind::kReportFault:
      Log(std::string("capture fault: ") + action.value);
      break;
  }
}

void CaptureTransport::StartRtc(const std::string& assignment_id) {
  CloseRtc("rtc_replaced");
  rtc_negotiation_deadline_ =
      std::chrono::steady_clock::now() + kRtcNegotiationTimeout;
  const auto epoch = state_.connection_epoch;
  rtc::Configuration peer_configuration;
  peer_configuration.disableAutoNegotiation = true;
  peer_configuration.disableFingerprintVerification = false;
  peer_configuration.maxMessageSize = kMaxRtcMessageBytes;
  peer_ = std::make_shared<rtc::PeerConnection>(peer_configuration);
  const std::weak_ptr<rtc::PeerConnection> weak_peer = peer_;
  const std::weak_ptr<CaptureTransport> weak_self = weak_from_this();

  rtc::DataChannelInit control_init;
  control_init.reliability.unordered = false;
  control_ = peer_->createDataChannel("teleop-control", control_init);
  rtc::DataChannelInit pose_init;
  pose_init.reliability.unordered = true;
  pose_init.reliability.maxRetransmits = 0;
  pose_ = peer_->createDataChannel("teleop-pose", pose_init);

  peer_->onGatheringStateChange([weak_self, epoch, assignment_id, weak_peer](
                                     rtc::PeerConnection::GatheringState state) {
    const auto self = weak_self.lock();
    const auto peer = weak_peer.lock();
    if (state != rtc::PeerConnection::GatheringState::Complete || !self || !peer) {
      return;
    }
    const auto description = peer->localDescription();
    if (!description || description->type() != rtc::Description::Type::Offer) {
      self->Enqueue(Event{
          .kind = EventKind::kRtcStateFailed,
          .connection_epoch = epoch,
          .assignment_id = assignment_id,
          .value = "rtc_local_offer_missing",
      });
      return;
    }
    self->Enqueue(Event{
        .kind = EventKind::kRtcOfferReady,
        .connection_epoch = epoch,
        .assignment_id = assignment_id,
        .value = std::string(*description),
    });
  });
  peer_->onStateChange([weak_self, epoch, assignment_id](
                           rtc::PeerConnection::State state) {
    if (state == rtc::PeerConnection::State::Disconnected ||
        state == rtc::PeerConnection::State::Failed ||
        state == rtc::PeerConnection::State::Closed) {
      const auto self = weak_self.lock();
      if (!self) {
        return;
      }
      self->Enqueue(Event{
          .kind = EventKind::kRtcStateFailed,
          .connection_epoch = epoch,
          .assignment_id = assignment_id,
          .value = "rtc_peer_lost",
      });
    }
  });
  const auto channel_callbacks = [weak_self, epoch, assignment_id](
                                     const std::shared_ptr<rtc::DataChannel>& channel,
                                     EventKind opened,
                                     EventKind closed) {
    channel->onOpen([weak_self, epoch, assignment_id, opened] {
      const auto self = weak_self.lock();
      if (!self) {
        return;
      }
      self->Enqueue(Event{
          .kind = opened,
          .connection_epoch = epoch,
          .assignment_id = assignment_id,
      });
    });
    channel->onClosed([weak_self, epoch, assignment_id, closed] {
      const auto self = weak_self.lock();
      if (!self) {
        return;
      }
      self->Enqueue(Event{
          .kind = closed,
          .connection_epoch = epoch,
          .assignment_id = assignment_id,
      });
    });
    channel->onError([weak_self, epoch, assignment_id, closed](std::string error) {
      const auto self = weak_self.lock();
      if (!self) {
        return;
      }
      self->Enqueue(Event{
          .kind = closed,
          .connection_epoch = epoch,
          .assignment_id = assignment_id,
          .value = std::move(error),
      });
    });
  };
  channel_callbacks(
      control_, EventKind::kControlOpened, EventKind::kControlClosed);
  channel_callbacks(pose_, EventKind::kPoseOpened, EventKind::kPoseClosed);
  control_->onMessage([weak_self](rtc::message_variant message) {
    const auto self = weak_self.lock();
    if (self) {
      self->HandleControlMessage(message);
    }
  });
  peer_->setLocalDescription(rtc::Description::Type::Offer);
}

void CaptureTransport::CloseRtc(const std::string& reason) {
  rtc_negotiation_deadline_ = std::chrono::steady_clock::time_point::max();
  if (pose_) {
    pose_->resetCallbacks();
    pose_->close();
    pose_.reset();
  }
  if (control_) {
    control_->resetCallbacks();
    control_->close();
    control_.reset();
  }
  if (peer_) {
    peer_->resetCallbacks();
    peer_->close();
    peer_.reset();
  }
  if (!reason.empty()) {
    Log(std::string("RTC closed: ") + reason);
  }
}

void CaptureTransport::CloseWebSocket() {
  visualization_ = {};
  operator_connection_.clear();operator_state_="disconnected";operator_error_.clear();
  pending_authentication_.reset();
  if (websocket_) {
    websocket_->resetCallbacks();
    websocket_->close();
    websocket_.reset();
  }
}

void CaptureTransport::SendWebSocket(const std::string& payload) {
  if (!websocket_ || !websocket_->isOpen() || !websocket_->send(payload)) {
    throw std::runtime_error("capture WSS send failed");
  }
}

void CaptureTransport::ScheduleReconnect(const std::string& reason) {
  CloseWebSocket();
  if (credential_persistence_failed_) {
    next_reconnect_ = std::chrono::steady_clock::time_point::max();
    Log("capture reconnect disabled: credential persistence failed");
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  next_reconnect_ = now + reconnect_delay_;
  reconnect_delay_ = std::min(reconnect_delay_ * 2, kMaximumReconnectDelay);
  Log(std::string("capture reconnect scheduled: ") + reason);
}

void CaptureTransport::SendPeerPing() {
  if (!control_ || !control_->isOpen()) {
    return;
  }
  ++peer_ping_sequence_;
  const nlohmann::json payload = {
      {"type", "peer_ping"},
      {"request_id", "native-" + std::to_string(peer_ping_sequence_)},
  };
  if (!control_->send(payload.dump())) {
    Apply(CaptureRtcFailed(
        state_,
        state_.connection_epoch,
        state_.assignment ? state_.assignment->id : std::string{},
        "control_ping_buffered"));
  }
}

void CaptureTransport::HandleControlMessage(const rtc::message_variant& message) {
  if (!std::holds_alternative<std::string>(message)) {
    return;
  }
  try {
    const auto payload = nlohmann::json::parse(std::get<std::string>(message));
    if (payload.is_object() && payload.value("type", "") == "peer_pong") {
      return;
    }
    if (payload.is_object() && payload.contains("error")) {
      Log("Driver RTC returned an error");
    }
  } catch (const nlohmann::json::exception&) {
    Log("Driver RTC returned malformed control JSON");
  }
}

void CaptureTransport::Log(const std::string& message) const {
  if (log_sink_) {
    log_sink_(message);
  }
}

}  // namespace motus::openxr_capture
