#pragma once

#include <string_view>
#include <vector>

namespace motus::quest_capture {

enum class HeadsetTarget {
  kMeta,
  kPico,
};

struct InteractionProfileRequirement {
  // Empty for interaction profiles which are part of OpenXR core.
  std::string_view extension_name;
  std::string_view interaction_profile_path;
};

struct RuntimeProfile {
  HeadsetTarget target;
  std::string_view application_name;
  std::vector<InteractionProfileRequirement> interaction_profiles;
  bool require_any_interaction_extension;
};

struct RuntimeSelection {
  // Optional extensions are enabled only when the runtime enumerates them.
  std::vector<std::string_view> optional_extensions;
  std::vector<std::string_view> interaction_profile_paths;
  bool local_floor_extension_enabled;
};

enum class FloorReferenceSpace {
  kLocalFloor,
  kStage,
};

[[nodiscard]] const RuntimeProfile& MetaRuntimeProfile();
[[nodiscard]] const RuntimeProfile& PicoRuntimeProfile();
[[nodiscard]] const RuntimeProfile& CompiledRuntimeProfile();

// Selects the optional extensions and interaction profiles which this build may
// use. PICO builds fail closed unless at least one supported PICO controller
// extension is present. Meta builds accept only the core Oculus Touch profile.
[[nodiscard]] RuntimeSelection SelectRuntime(
    const RuntimeProfile& profile,
    const std::vector<std::string_view>& available_extensions);

// Robot poses must always be expressed in a floor-relative reference space.
// LOCAL is deliberately not an accepted fallback because its origin is tied to
// the headset start/recenter pose rather than the physical floor.
[[nodiscard]] FloorReferenceSpace SelectFloorReferenceSpace(
    bool local_floor_extension_enabled,
    bool local_floor_space_available,
    bool stage_space_available);

[[nodiscard]] bool IsInteractionProfileAllowed(
    const RuntimeSelection& selection,
    std::string_view interaction_profile_path);

}  // namespace motus::quest_capture
