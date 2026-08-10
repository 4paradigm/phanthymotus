#include "motus/quest_capture/enrollment.hpp"

#include <stdexcept>

namespace motus::quest_capture {

CaptureEnrollmentSelection SelectCaptureEnrollment(
    const StoredCaptureEnrollment& stored,
    const CaptureLaunchBootstrap& launch) {
  const bool pairing_present =
      !launch.pairing_id.empty() || !launch.pairing_code.empty();
  if (pairing_present) {
    if (
        launch.pairing_id.empty() || launch.pairing_code.empty() ||
        launch.websocket_url.empty() || launch.ca_certificate_pem.empty()) {
      throw std::invalid_argument("capture pairing bootstrap is incomplete");
    }
    return CaptureEnrollmentSelection{
        .identity = std::nullopt,
        .pairing = PairingBootstrap{
            .pairing_id = launch.pairing_id,
            .pairing_code = launch.pairing_code,
        },
        .websocket_url = launch.websocket_url,
        .ca_certificate_pem = launch.ca_certificate_pem,
    };
  }

  if (launch.transport_override_present) {
    throw std::invalid_argument("stored capture transport cannot be overridden");
  }
  if (
      stored.capture_id.empty() || stored.capture_credential.empty() ||
      stored.websocket_url.empty() || stored.ca_certificate_pem.empty()) {
    throw std::invalid_argument("stored capture enrollment is incomplete");
  }
  return CaptureEnrollmentSelection{
      .identity = CaptureIdentity{
          .capture_id = stored.capture_id,
          .capture_credential = stored.capture_credential,
      },
      .pairing = std::nullopt,
      .websocket_url = stored.websocket_url,
      .ca_certificate_pem = stored.ca_certificate_pem,
  };
}

}  // namespace motus::quest_capture
