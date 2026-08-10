#include "motus/quest_capture/enrollment.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using motus::quest_capture::CaptureLaunchBootstrap;
using motus::quest_capture::SelectCaptureEnrollment;
using motus::quest_capture::StoredCaptureEnrollment;

[[noreturn]] void Fail(const std::string& message) {
  std::cerr << "enrollment_test: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, const std::string& message) {
  if (!condition) {
    Fail(message);
  }
}

template <typename Callback>
void CheckRejected(Callback callback, const std::string& message) {
  try {
    callback();
  } catch (const std::invalid_argument&) {
    return;
  }
  Fail(message);
}

StoredCaptureEnrollment Stored() {
  return StoredCaptureEnrollment{
      .capture_id = "capture-a",
      .capture_credential = "credential-a",
      .websocket_url = "wss://trusted.example/ws/teleop-capture",
      .ca_certificate_pem = "trusted-ca",
  };
}

void ResumeIsPinnedToTheStoredOrigin() {
  const auto selected = SelectCaptureEnrollment(Stored(), {});
  Check(selected.identity.has_value() && !selected.pairing.has_value(),
        "resume must select the stored identity");
  Check(selected.websocket_url == "wss://trusted.example/ws/teleop-capture" &&
            selected.ca_certificate_pem == "trusted-ca",
        "resume must atomically reuse the stored WSS origin and CA");

  CheckRejected(
      [] {
        static_cast<void>(SelectCaptureEnrollment(
            Stored(),
            CaptureLaunchBootstrap{
                .websocket_url = "wss://attacker.example/ws/teleop-capture",
                .ca_certificate_pem = "attacker-ca",
                .transport_override_present = true,
            }));
      },
      "an exported Activity must reject transport injection on resume");
}

void ExplicitCompletePairingMayReplaceTheEnrollment() {
  const auto selected = SelectCaptureEnrollment(
      Stored(),
      CaptureLaunchBootstrap{
          .pairing_id = "pairing-b",
          .pairing_code = "pairing-secret-b",
          .websocket_url = "wss://replacement.example/ws/teleop-capture",
          .ca_certificate_pem = "replacement-ca",
          .transport_override_present = true,
      });
  Check(!selected.identity.has_value() && selected.pairing.has_value(),
        "a complete new pairing must not send the old credential");
  Check(selected.websocket_url ==
            "wss://replacement.example/ws/teleop-capture",
        "a complete new pairing may bind a replacement origin");

  CheckRejected(
      [] {
        static_cast<void>(SelectCaptureEnrollment(
            Stored(),
            CaptureLaunchBootstrap{
                .pairing_id = "pairing-only",
                .websocket_url = "wss://replacement.example/ws/teleop-capture",
                .ca_certificate_pem = "replacement-ca",
                .transport_override_present = true,
            }));
      },
      "partial pairing bootstrap must fail closed");
}

void IncompleteStoredEnrollmentFailsClosed() {
  auto stored = Stored();
  stored.capture_credential.clear();
  CheckRejected(
      [&stored] {
        static_cast<void>(SelectCaptureEnrollment(stored, {}));
      },
      "identity without its credential and transport binding must be unusable");
}

}  // namespace

int main() {
  ResumeIsPinnedToTheStoredOrigin();
  ExplicitCompletePairingMayReplaceTheEnrollment();
  IncompleteStoredEnrollmentFailsClosed();
  std::cout << "quest-capture enrollment: all host tests passed\n";
  return 0;
}
