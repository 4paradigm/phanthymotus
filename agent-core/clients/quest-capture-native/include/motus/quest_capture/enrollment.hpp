#pragma once

#include <optional>
#include <string>

#include "motus/quest_capture/capture_session.hpp"

namespace motus::quest_capture {

struct StoredCaptureEnrollment {
  std::string capture_id;
  std::string capture_credential;
  std::string websocket_url;
  std::string ca_certificate_pem;
};

struct CaptureLaunchBootstrap {
  std::string pairing_id;
  std::string pairing_code;
  std::string websocket_url;
  std::string ca_certificate_pem;
  bool transport_override_present{false};
};

struct CaptureEnrollmentSelection {
  std::optional<CaptureIdentity> identity;
  std::optional<PairingBootstrap> pairing;
  std::string websocket_url;
  std::string ca_certificate_pem;
};

// An exported NativeActivity may receive Intents from any local application.
// Existing bearer credentials are therefore inseparable from the WSS origin
// and trust material stored in the same successful pairing transaction.
CaptureEnrollmentSelection SelectCaptureEnrollment(
    const StoredCaptureEnrollment& stored,
    const CaptureLaunchBootstrap& launch);

}  // namespace motus::quest_capture
