#include "runtime_profile.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using motus::quest_capture::FloorReferenceSpace;
using motus::quest_capture::HeadsetTarget;
using motus::quest_capture::IsInteractionProfileAllowed;
using motus::quest_capture::SelectFloorReferenceSpace;
using motus::quest_capture::SelectRuntime;

constexpr std::string_view kLocalFloorExtension = "XR_EXT_local_floor";
constexpr std::string_view kPico4Extension = "XR_BD_controller_interaction";
constexpr std::string_view kPicoUltraExtension =
    "XR_BD_ultra_controller_interaction";
constexpr std::string_view kOculusTouchProfile =
    "/interaction_profiles/oculus/touch_controller";
constexpr std::string_view kPico4Profile =
    "/interaction_profiles/bytedance/pico4_controller";
constexpr std::string_view kPicoUltraProfile =
    "/interaction_profiles/bytedance/pico_ultra_controller_bd";

[[noreturn]] void Fail(std::string_view message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void Check(bool condition, std::string_view message) {
  if (!condition) {
    Fail(message);
  }
}

template <typename Function>
void CheckThrows(Function&& function, std::string_view message) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  Fail(message);
}

void TestMetaSelection() {
  const auto selection = SelectRuntime(
      motus::quest_capture::MetaRuntimeProfile(),
      {kLocalFloorExtension, kPico4Extension, kPicoUltraExtension});
  Check(selection.local_floor_extension_enabled,
        "Meta selection should enable available local-floor support");
  Check(selection.optional_extensions.size() == 1,
        "Meta selection must not enable PICO controller extensions");
  Check(IsInteractionProfileAllowed(selection, kOculusTouchProfile),
        "Meta selection should allow Oculus Touch");
  Check(!IsInteractionProfileAllowed(selection, kPico4Profile),
        "Meta selection must reject PICO 4 controllers");
  Check(!IsInteractionProfileAllowed(selection, kPicoUltraProfile),
        "Meta selection must reject PICO Ultra controllers");
}

void TestPicoSelection() {
  const auto both = SelectRuntime(
      motus::quest_capture::PicoRuntimeProfile(),
      {kLocalFloorExtension, kPico4Extension, kPicoUltraExtension});
  Check(both.local_floor_extension_enabled,
        "PICO selection should enable available local-floor support");
  Check(both.optional_extensions.size() == 3,
        "PICO selection should enable both available controller extensions");
  Check(IsInteractionProfileAllowed(both, kPico4Profile),
        "PICO selection should allow PICO 4 when its extension is enabled");
  Check(IsInteractionProfileAllowed(both, kPicoUltraProfile),
        "PICO selection should allow PICO Ultra when its extension is enabled");
  Check(!IsInteractionProfileAllowed(both, kOculusTouchProfile),
        "PICO selection must reject Oculus Touch");

  const auto pico4_only = SelectRuntime(
      motus::quest_capture::PicoRuntimeProfile(), {kPico4Extension});
  Check(IsInteractionProfileAllowed(pico4_only, kPico4Profile),
        "PICO 4-only runtime should select PICO 4");
  Check(!IsInteractionProfileAllowed(pico4_only, kPicoUltraProfile),
        "PICO 4-only runtime must not allow PICO Ultra");

  const auto ultra_only = SelectRuntime(
      motus::quest_capture::PicoRuntimeProfile(), {kPicoUltraExtension});
  Check(!IsInteractionProfileAllowed(ultra_only, kPico4Profile),
        "PICO Ultra-only runtime must not allow PICO 4");
  Check(IsInteractionProfileAllowed(ultra_only, kPicoUltraProfile),
        "PICO Ultra-only runtime should select PICO Ultra");

  CheckThrows(
      [] {
        static_cast<void>(SelectRuntime(
            motus::quest_capture::PicoRuntimeProfile(),
            {kLocalFloorExtension}));
      },
      "PICO runtime without either controller extension must fail closed");
}

void TestFloorSelection() {
  Check(
      SelectFloorReferenceSpace(true, true, true) ==
          FloorReferenceSpace::kLocalFloor,
      "local-floor should take priority over stage");
  Check(
      SelectFloorReferenceSpace(false, true, true) ==
          FloorReferenceSpace::kStage,
      "an unenabled local-floor extension must not be used");
  Check(
      SelectFloorReferenceSpace(true, false, true) ==
          FloorReferenceSpace::kStage,
      "stage should be the floor-relative fallback");
  CheckThrows(
      [] {
        static_cast<void>(SelectFloorReferenceSpace(false, false, false));
      },
      "absence of local-floor and stage must fail closed");
  CheckThrows(
      [] {
        static_cast<void>(SelectFloorReferenceSpace(true, false, false));
      },
      "enumerated local-floor space is required before use");
}

void TestCompiledTarget() {
#if defined(TEST_EXPECT_META)
  Check(
      motus::quest_capture::CompiledRuntimeProfile().target ==
          HeadsetTarget::kMeta,
      "Meta compile macro should select the Meta runtime profile");
#elif defined(TEST_EXPECT_PICO)
  Check(
      motus::quest_capture::CompiledRuntimeProfile().target ==
          HeadsetTarget::kPico,
      "PICO compile macro should select the PICO runtime profile");
#else
#error "runtime_profile_test requires TEST_EXPECT_META or TEST_EXPECT_PICO"
#endif
}

}  // namespace

int main() {
  TestMetaSelection();
  TestPicoSelection();
  TestFloorSelection();
  TestCompiledTarget();
  std::cout << "runtime_profile_test: PASS\n";
  return 0;
}
