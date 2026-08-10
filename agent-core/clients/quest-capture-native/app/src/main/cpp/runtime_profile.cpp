#include "runtime_profile.hpp"

#include <algorithm>
#include <stdexcept>

namespace motus::quest_capture {
namespace {

constexpr std::string_view kLocalFloorExtension = "XR_EXT_local_floor";
constexpr std::string_view kPico4ControllerExtension =
    "XR_BD_controller_interaction";
constexpr std::string_view kPicoUltraControllerExtension =
    "XR_BD_ultra_controller_interaction";
constexpr std::string_view kOculusTouchProfile =
    "/interaction_profiles/oculus/touch_controller";
constexpr std::string_view kPico4ControllerProfile =
    "/interaction_profiles/bytedance/pico4_controller";
constexpr std::string_view kPicoUltraControllerProfile =
    "/interaction_profiles/bytedance/pico_ultra_controller_bd";

bool Contains(
    const std::vector<std::string_view>& values,
    std::string_view value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

const RuntimeProfile& MetaRuntimeProfile() {
  static const RuntimeProfile profile{
      .target = HeadsetTarget::kMeta,
      .application_name = "PhanthyMotus Meta Capture",
      .interaction_profiles = {
          {
              .extension_name = {},
              .interaction_profile_path = kOculusTouchProfile,
          },
      },
      .require_any_interaction_extension = false,
  };
  return profile;
}

const RuntimeProfile& PicoRuntimeProfile() {
  static const RuntimeProfile profile{
      .target = HeadsetTarget::kPico,
      .application_name = "PhanthyMotus PICO Capture",
      .interaction_profiles = {
          {
              .extension_name = kPico4ControllerExtension,
              .interaction_profile_path = kPico4ControllerProfile,
          },
          {
              .extension_name = kPicoUltraControllerExtension,
              .interaction_profile_path = kPicoUltraControllerProfile,
          },
      },
      .require_any_interaction_extension = true,
  };
  return profile;
}

const RuntimeProfile& CompiledRuntimeProfile() {
#if defined(MOTUS_CAPTURE_HEADSET_META) && defined(MOTUS_CAPTURE_HEADSET_PICO)
#error "Only one MOTUS capture headset target may be compiled"
#elif defined(MOTUS_CAPTURE_HEADSET_META)
  return MetaRuntimeProfile();
#elif defined(MOTUS_CAPTURE_HEADSET_PICO)
  return PicoRuntimeProfile();
#else
#error "MOTUS_CAPTURE_HEADSET_META or MOTUS_CAPTURE_HEADSET_PICO is required"
#endif
}

RuntimeSelection SelectRuntime(
    const RuntimeProfile& profile,
    const std::vector<std::string_view>& available_extensions) {
  RuntimeSelection selection{
      .optional_extensions = {},
      .interaction_profile_paths = {},
      .local_floor_extension_enabled =
          Contains(available_extensions, kLocalFloorExtension),
  };
  if (selection.local_floor_extension_enabled) {
    selection.optional_extensions.push_back(kLocalFloorExtension);
  }

  bool interaction_extension_enabled = false;
  for (const auto& candidate : profile.interaction_profiles) {
    if (candidate.extension_name.empty()) {
      selection.interaction_profile_paths.push_back(
          candidate.interaction_profile_path);
      continue;
    }
    if (!Contains(available_extensions, candidate.extension_name)) {
      continue;
    }
    selection.optional_extensions.push_back(candidate.extension_name);
    selection.interaction_profile_paths.push_back(
        candidate.interaction_profile_path);
    interaction_extension_enabled = true;
  }

  if (profile.require_any_interaction_extension &&
      !interaction_extension_enabled) {
    throw std::runtime_error(
        "no supported PICO controller interaction extension is available");
  }
  if (selection.interaction_profile_paths.empty()) {
    throw std::runtime_error("no allowed controller interaction profile selected");
  }
  return selection;
}

FloorReferenceSpace SelectFloorReferenceSpace(
    bool local_floor_extension_enabled,
    bool local_floor_space_available,
    bool stage_space_available) {
  if (local_floor_extension_enabled && local_floor_space_available) {
    return FloorReferenceSpace::kLocalFloor;
  }
  if (stage_space_available) {
    return FloorReferenceSpace::kStage;
  }
  throw std::runtime_error(
      "no floor-relative OpenXR reference space is available");
}

bool IsInteractionProfileAllowed(
    const RuntimeSelection& selection,
    std::string_view interaction_profile_path) {
  return Contains(
      selection.interaction_profile_paths, interaction_profile_path);
}

}  // namespace motus::quest_capture
