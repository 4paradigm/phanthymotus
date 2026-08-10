#pragma once

#include <rtc/rtc.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "motus/quest_capture/capture_session.hpp"

namespace motus::quest_capture {

struct CaptureTransportConfiguration {
  std::string websocket_url;
  std::string app_version;
  std::optional<CaptureIdentity> identity;
  std::optional<PairingBootstrap> pairing;
  std::string ca_certificate_pem;
};

class CaptureTransport final
    : public std::enable_shared_from_this<CaptureTransport> {
 public:
  using IdentitySink = std::function<void(const CaptureIdentity&)>;
  using LogSink = std::function<void(const std::string&)>;

  CaptureTransport(
      CaptureTransportConfiguration configuration,
      IdentitySink identity_sink,
      LogSink log_sink);
  ~CaptureTransport();

  CaptureTransport(const CaptureTransport&) = delete;
  CaptureTransport& operator=(const CaptureTransport&) = delete;

  void Start();
  void Shutdown();
  void Tick();
  void SetXrFocused(bool focused);
  void SendFrame(const FrameSample& sample);

  [[nodiscard]] CaptureLinkState link_state() const { return state_.link; }
  [[nodiscard]] bool streaming() const {
    return state_.link == CaptureLinkState::kStreaming;
  }
  [[nodiscard]] const std::optional<CaptureAssignmentV1>& assignment() const {
    return state_.assignment;
  }

 private:
  enum class EventKind {
    kWebSocketOpen,
    kWebSocketMessage,
    kWebSocketClosed,
    kWebSocketError,
    kRtcOfferReady,
    kRtcStateFailed,
    kControlOpened,
    kControlClosed,
    kPoseOpened,
    kPoseClosed,
  };

  struct Event {
    EventKind kind{EventKind::kWebSocketClosed};
    std::int64_t connection_epoch{0};
    std::string assignment_id;
    std::string value;
  };

  void ConnectNow();
  void Enqueue(Event event);
  void HandleEvent(const Event& event);
  void HandleServerMessage(const std::string& payload);
  void Apply(CaptureTransition transition);
  void ApplyAction(const CaptureAction& action);
  void StartRtc(const std::string& assignment_id);
  void CloseRtc(const std::string& reason);
  void CloseWebSocket();
  void SendWebSocket(const std::string& payload);
  void ScheduleReconnect(const std::string& reason);
  void SendPeerPing();
  void HandleControlMessage(const rtc::message_variant& message);
  void Log(const std::string& message) const;

  CaptureTransportConfiguration configuration_;
  IdentitySink identity_sink_;
  LogSink log_sink_;
  CaptureSessionState state_;

  mutable std::mutex event_mutex_;
  std::deque<Event> events_;
  std::shared_ptr<rtc::WebSocket> websocket_;
  std::shared_ptr<rtc::PeerConnection> peer_;
  std::shared_ptr<rtc::DataChannel> control_;
  std::shared_ptr<rtc::DataChannel> pose_;
  std::optional<CaptureAction> pending_authentication_;

  bool started_{false};
  std::atomic_bool shutting_down_{false};
  std::chrono::steady_clock::time_point next_presence_{};
  std::chrono::steady_clock::time_point next_reconnect_{};
  std::chrono::steady_clock::time_point next_ping_{};
  std::chrono::steady_clock::time_point rtc_negotiation_deadline_{
      std::chrono::steady_clock::time_point::max()};
  std::chrono::milliseconds presence_interval_{2'000};
  std::chrono::milliseconds reconnect_delay_{250};
  std::uint64_t peer_ping_sequence_{0};
};

}  // namespace motus::quest_capture
