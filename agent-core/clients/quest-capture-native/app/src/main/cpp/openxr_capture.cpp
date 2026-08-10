// OpenXR Android lifecycle and graphics initialization follow Khronos hello_xr
// release 1.1.60 (Apache-2.0), reduced here to a capture-only GLES session.

#include "openxr_capture.hpp"

#include <GLES3/gl3.h>
#include <android_native_app_glue.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace motus::quest_capture {
namespace {

constexpr XrViewConfigurationType kViewConfiguration =
    XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

void CopyName(char* target, std::size_t size, std::string_view value) {
  if (value.size() >= size) {
    throw std::invalid_argument("OpenXR name is too long");
  }
  std::memset(target, 0, size);
  std::memcpy(target, value.data(), value.size());
}

bool SupportsExtension(
    const std::vector<XrExtensionProperties>& properties,
    const char* extension) {
  return std::any_of(properties.begin(), properties.end(), [&](const auto& item) {
    return std::strcmp(item.extensionName, extension) == 0;
  });
}

XrPosef IdentityPose() {
  XrPosef pose{};
  pose.orientation.w = 1.0F;
  return pose;
}

bool FullyTracked(XrSpaceLocationFlags flags) {
  constexpr XrSpaceLocationFlags required =
      XR_SPACE_LOCATION_POSITION_VALID_BIT |
      XR_SPACE_LOCATION_ORIENTATION_VALID_BIT |
      XR_SPACE_LOCATION_POSITION_TRACKED_BIT |
      XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
  return (flags & required) == required;
}

}  // namespace

OpenXrCapture::~OpenXrCapture() {
  Destroy();
}

void OpenXrCapture::Initialize(android_app* app) {
  if (app == nullptr || app->activity == nullptr) {
    throw std::invalid_argument("android_app is required");
  }
  monotonic_origin_ = std::chrono::steady_clock::now();
  InitializeLoader(app);
  CreateInstance(app);
  CreateEglContext();
  CreateSession();
  CreateSpaces();
  CreateActions();
  CreateSwapchains();
}

void OpenXrCapture::InitializeLoader(android_app* app) {
  PFN_xrInitializeLoaderKHR initialize_loader = nullptr;
  const XrResult get_result = xrGetInstanceProcAddr(
      XR_NULL_HANDLE,
      "xrInitializeLoaderKHR",
      reinterpret_cast<PFN_xrVoidFunction*>(&initialize_loader));
  if (XR_FAILED(get_result) || initialize_loader == nullptr) {
    throw std::runtime_error("xrInitializeLoaderKHR is unavailable");
  }
  XrLoaderInitInfoAndroidKHR loader_info{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
  loader_info.applicationVM = app->activity->vm;
  loader_info.applicationContext = app->activity->clazz;
  Check(
      initialize_loader(
          reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(&loader_info)),
      "xrInitializeLoaderKHR");
}

void OpenXrCapture::CreateInstance(android_app* app) {
  std::uint32_t extension_count = 0;
  Check(xrEnumerateInstanceExtensionProperties(
            nullptr, 0, &extension_count, nullptr),
        "xrEnumerateInstanceExtensionProperties(count)");
  std::vector<XrExtensionProperties> extensions(
      extension_count, XrExtensionProperties{XR_TYPE_EXTENSION_PROPERTIES});
  Check(xrEnumerateInstanceExtensionProperties(
            nullptr, extension_count, &extension_count, extensions.data()),
        "xrEnumerateInstanceExtensionProperties(values)");
  const std::array<const char*, 2> required_extensions = {
      XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
      XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME,
  };
  for (const char* extension : required_extensions) {
    if (!SupportsExtension(extensions, extension)) {
      throw std::runtime_error(std::string("required OpenXR extension missing: ") + extension);
    }
  }
  std::vector<std::string_view> available_extensions;
  available_extensions.reserve(extensions.size());
  for (const auto& extension : extensions) {
    available_extensions.emplace_back(extension.extensionName);
  }
  runtime_selection_ = SelectRuntime(
      CompiledRuntimeProfile(), available_extensions);
  std::vector<const char*> enabled_extensions(
      required_extensions.begin(), required_extensions.end());
  enabled_extensions.reserve(
      required_extensions.size() + runtime_selection_.optional_extensions.size());
  for (const std::string_view extension :
       runtime_selection_.optional_extensions) {
    enabled_extensions.push_back(extension.data());
  }

  XrInstanceCreateInfoAndroidKHR android_info{
      XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
  android_info.applicationVM = app->activity->vm;
  android_info.applicationActivity = app->activity->clazz;

  XrInstanceCreateInfo create_info{XR_TYPE_INSTANCE_CREATE_INFO};
  create_info.next = &android_info;
  CopyName(
      create_info.applicationInfo.applicationName,
      sizeof(create_info.applicationInfo.applicationName),
      CompiledRuntimeProfile().application_name);
  create_info.applicationInfo.applicationVersion = 1;
  CopyName(
      create_info.applicationInfo.engineName,
      sizeof(create_info.applicationInfo.engineName),
      "phanthymotus-native");
  create_info.applicationInfo.engineVersion = 1;
  // This client uses only OpenXR 1.0 core commands plus explicit extensions;
  // requesting 1.0 keeps it compatible with runtimes that have not promoted
  // their loader/runtime interface to 1.1.
  create_info.applicationInfo.apiVersion = XR_API_VERSION_1_0;
  create_info.enabledExtensionCount = enabled_extensions.size();
  create_info.enabledExtensionNames = enabled_extensions.data();
  Check(xrCreateInstance(&create_info, &instance_), "xrCreateInstance");

  XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
  system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  Check(xrGetSystem(instance_, &system_info, &system_id_), "xrGetSystem");
}

void OpenXrCapture::CreateEglContext() {
  PFN_xrGetOpenGLESGraphicsRequirementsKHR get_requirements = nullptr;
  Check(xrGetInstanceProcAddr(
            instance_,
            "xrGetOpenGLESGraphicsRequirementsKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&get_requirements)),
        "xrGetInstanceProcAddr(OpenGLES requirements)");
  if (get_requirements == nullptr) {
    throw std::runtime_error("OpenGL ES graphics requirements function is missing");
  }
  XrGraphicsRequirementsOpenGLESKHR requirements{
      XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
  Check(get_requirements(instance_, system_id_, &requirements),
        "xrGetOpenGLESGraphicsRequirementsKHR");
  constexpr XrVersion requested_gles = XR_MAKE_VERSION(3, 0, 0);
  if (requested_gles < requirements.minApiVersionSupported ||
      requested_gles > requirements.maxApiVersionSupported) {
    throw std::runtime_error("OpenGL ES 3.0 is outside runtime graphics requirements");
  }

  egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (egl_display_ == EGL_NO_DISPLAY || eglInitialize(egl_display_, nullptr, nullptr) != EGL_TRUE) {
    throw std::runtime_error("eglInitialize failed");
  }
  const EGLint config_attributes[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_DEPTH_SIZE, 0,
      EGL_NONE,
  };
  EGLint config_count = 0;
  if (eglChooseConfig(
          egl_display_, config_attributes, &egl_config_, 1, &config_count) != EGL_TRUE ||
      config_count != 1) {
    throw std::runtime_error("eglChooseConfig failed");
  }
  const EGLint context_attributes[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE,
  };
  egl_context_ = eglCreateContext(
      egl_display_, egl_config_, EGL_NO_CONTEXT, context_attributes);
  const EGLint surface_attributes[] = {
      EGL_WIDTH, 16,
      EGL_HEIGHT, 16,
      EGL_NONE,
  };
  egl_surface_ = eglCreatePbufferSurface(
      egl_display_, egl_config_, surface_attributes);
  if (egl_context_ == EGL_NO_CONTEXT || egl_surface_ == EGL_NO_SURFACE ||
      eglMakeCurrent(
          egl_display_, egl_surface_, egl_surface_, egl_context_) != EGL_TRUE) {
    throw std::runtime_error("OpenGL ES context creation failed");
  }
}

void OpenXrCapture::CreateSession() {
  XrGraphicsBindingOpenGLESAndroidKHR graphics_binding{
      XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
  graphics_binding.display = egl_display_;
  graphics_binding.config = egl_config_;
  graphics_binding.context = egl_context_;
  XrSessionCreateInfo create_info{XR_TYPE_SESSION_CREATE_INFO};
  create_info.next = &graphics_binding;
  create_info.systemId = system_id_;
  Check(xrCreateSession(instance_, &create_info, &session_), "xrCreateSession");
}

void OpenXrCapture::CreateSpaces() {
  std::uint32_t space_count = 0;
  Check(xrEnumerateReferenceSpaces(session_, 0, &space_count, nullptr),
        "xrEnumerateReferenceSpaces(count)");
  std::vector<XrReferenceSpaceType> spaces(space_count);
  Check(xrEnumerateReferenceSpaces(
            session_, space_count, &space_count, spaces.data()),
        "xrEnumerateReferenceSpaces(values)");
  const bool local_floor_available = std::find(
      spaces.begin(), spaces.end(), XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR_EXT) !=
      spaces.end();
  const bool stage_available = std::find(
      spaces.begin(), spaces.end(), XR_REFERENCE_SPACE_TYPE_STAGE) !=
      spaces.end();
  const FloorReferenceSpace selected_space = SelectFloorReferenceSpace(
      runtime_selection_.local_floor_extension_enabled,
      local_floor_available,
      stage_available);
  const XrReferenceSpaceType floor_space_type =
      selected_space == FloorReferenceSpace::kLocalFloor
      ? XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR_EXT
      : XR_REFERENCE_SPACE_TYPE_STAGE;
  XrReferenceSpaceCreateInfo floor_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  floor_info.referenceSpaceType = floor_space_type;
  floor_info.poseInReferenceSpace = IdentityPose();
  Check(xrCreateReferenceSpace(session_, &floor_info, &floor_space_),
        selected_space == FloorReferenceSpace::kLocalFloor
            ? "xrCreateReferenceSpace(local_floor)"
            : "xrCreateReferenceSpace(stage)");

  XrReferenceSpaceCreateInfo view_info{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  view_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  view_info.poseInReferenceSpace = IdentityPose();
  Check(xrCreateReferenceSpace(session_, &view_info, &view_space_),
        "xrCreateReferenceSpace(view)");
}

void OpenXrCapture::CreateActions() {
  Check(xrStringToPath(instance_, "/user/hand/left", &hand_paths_[0]),
        "xrStringToPath(left hand)");
  Check(xrStringToPath(instance_, "/user/hand/right", &hand_paths_[1]),
        "xrStringToPath(right hand)");

  XrActionSetCreateInfo set_info{XR_TYPE_ACTION_SET_CREATE_INFO};
  CopyName(set_info.actionSetName, sizeof(set_info.actionSetName), "teleop_capture");
  CopyName(
      set_info.localizedActionSetName,
      sizeof(set_info.localizedActionSetName),
      "Teleoperation capture");
  set_info.priority = 0;
  Check(xrCreateActionSet(instance_, &set_info, &action_set_), "xrCreateActionSet");

  const auto create_action = [this](
                                 XrActionType type,
                                 std::string_view name,
                                 std::string_view localized,
                                 XrAction* action) {
    XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
    info.actionType = type;
    CopyName(info.actionName, sizeof(info.actionName), name);
    CopyName(info.localizedActionName, sizeof(info.localizedActionName), localized);
    info.countSubactionPaths = hand_paths_.size();
    info.subactionPaths = hand_paths_.data();
    Check(xrCreateAction(action_set_, &info, action), "xrCreateAction");
  };
  create_action(
      XR_ACTION_TYPE_POSE_INPUT,
      "grip_pose",
      "Grip pose",
      &grip_pose_action_);
  create_action(
      XR_ACTION_TYPE_FLOAT_INPUT,
      "squeeze_value",
      "Squeeze value",
      &squeeze_action_);
  create_action(
      XR_ACTION_TYPE_VECTOR2F_INPUT,
      "thumbstick",
      "Thumbstick",
      &thumbstick_action_);

  std::vector<XrActionSuggestedBinding> bindings;
  bindings.reserve(6);
  for (std::size_t index = 0; index < hand_paths_.size(); ++index) {
    const char* hand = index == 0 ? "left" : "right";
    const auto binding_path = [&](const char* suffix) {
      XrPath path = XR_NULL_PATH;
      const std::string value = std::string("/user/hand/") + hand + suffix;
      Check(xrStringToPath(instance_, value.c_str(), &path), "xrStringToPath(binding)");
      return path;
    };
    bindings.push_back({grip_pose_action_, binding_path("/input/grip/pose")});
    bindings.push_back({squeeze_action_, binding_path("/input/squeeze/value")});
    bindings.push_back({thumbstick_action_, binding_path("/input/thumbstick")});
  }
  allowed_interaction_profiles_.clear();
  allowed_interaction_profiles_.reserve(
      runtime_selection_.interaction_profile_paths.size());
  for (const std::string_view interaction_profile :
       runtime_selection_.interaction_profile_paths) {
    XrPath profile_path = XR_NULL_PATH;
    Check(xrStringToPath(
              instance_, interaction_profile.data(), &profile_path),
          "xrStringToPath(interaction profile)");
    XrInteractionProfileSuggestedBinding suggested{
        XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggested.interactionProfile = profile_path;
    suggested.countSuggestedBindings = bindings.size();
    suggested.suggestedBindings = bindings.data();
    Check(xrSuggestInteractionProfileBindings(instance_, &suggested),
          "xrSuggestInteractionProfileBindings");
    allowed_interaction_profiles_.push_back(profile_path);
  }

  XrSessionActionSetsAttachInfo attach_info{
      XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
  attach_info.countActionSets = 1;
  attach_info.actionSets = &action_set_;
  Check(xrAttachSessionActionSets(session_, &attach_info),
        "xrAttachSessionActionSets");

  for (std::size_t index = 0; index < grip_spaces_.size(); ++index) {
    XrActionSpaceCreateInfo space_info{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    space_info.action = grip_pose_action_;
    space_info.subactionPath = hand_paths_[index];
    space_info.poseInActionSpace = IdentityPose();
    Check(xrCreateActionSpace(session_, &space_info, &grip_spaces_[index]),
          "xrCreateActionSpace");
  }
}

void OpenXrCapture::CreateSwapchains() {
  std::uint32_t view_count = 0;
  Check(xrEnumerateViewConfigurationViews(
            instance_, system_id_, kViewConfiguration, 0, &view_count, nullptr),
        "xrEnumerateViewConfigurationViews(count)");
  if (view_count == 0 || view_count > 4) {
    throw std::runtime_error("unexpected primary stereo view count");
  }
  view_configuration_.assign(
      view_count, XrViewConfigurationView{XR_TYPE_VIEW_CONFIGURATION_VIEW});
  Check(xrEnumerateViewConfigurationViews(
            instance_,
            system_id_,
            kViewConfiguration,
            view_count,
            &view_count,
            view_configuration_.data()),
        "xrEnumerateViewConfigurationViews(values)");

  std::uint32_t format_count = 0;
  Check(xrEnumerateSwapchainFormats(session_, 0, &format_count, nullptr),
        "xrEnumerateSwapchainFormats(count)");
  std::vector<std::int64_t> formats(format_count);
  Check(xrEnumerateSwapchainFormats(
            session_, format_count, &format_count, formats.data()),
        "xrEnumerateSwapchainFormats(values)");
  constexpr std::array<std::int64_t, 2> preferred = {
      GL_SRGB8_ALPHA8,
      GL_RGBA8,
  };
  for (const auto candidate : preferred) {
    if (std::find(formats.begin(), formats.end(), candidate) != formats.end()) {
      color_format_ = candidate;
      break;
    }
  }
  if (color_format_ == 0) {
    throw std::runtime_error("no supported RGBA OpenGL ES swapchain format");
  }

  swapchains_.reserve(view_count);
  for (const auto& view : view_configuration_) {
    XrSwapchainCreateInfo create_info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    create_info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
        XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    create_info.format = color_format_;
    // The capture renderer attaches a plain GL_TEXTURE_2D and does not create
    // a multisample resolve target. Khronos hello_xr's GLES backend likewise
    // advertises one supported sample for this path.
    create_info.sampleCount = 1;
    create_info.width = view.recommendedImageRectWidth;
    create_info.height = view.recommendedImageRectHeight;
    create_info.faceCount = 1;
    create_info.arraySize = 1;
    create_info.mipCount = 1;
    Swapchain swapchain;
    swapchain.width = static_cast<std::int32_t>(create_info.width);
    swapchain.height = static_cast<std::int32_t>(create_info.height);
    Check(
        xrCreateSwapchain(session_, &create_info, &swapchain.handle),
        "xrCreateSwapchain");
    std::uint32_t image_count = 0;
    Check(xrEnumerateSwapchainImages(
              swapchain.handle, 0, &image_count, nullptr),
          "xrEnumerateSwapchainImages(count)");
    swapchain.images.assign(
        image_count,
        XrSwapchainImageOpenGLESKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
    Check(xrEnumerateSwapchainImages(
              swapchain.handle,
              image_count,
              &image_count,
              reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchain.images.data())),
          "xrEnumerateSwapchainImages(values)");
    swapchains_.push_back(std::move(swapchain));
  }
}

void OpenXrCapture::PollEvents() {
  if (instance_ == XR_NULL_HANDLE) {
    return;
  }
  XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
  for (;;) {
    const XrResult result = xrPollEvent(instance_, &event);
    if (result == XR_EVENT_UNAVAILABLE) {
      break;
    }
    Check(result, "xrPollEvent");
    if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      const auto* changed =
          reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
      if (changed->session == session_) {
        HandleSessionState(changed->state);
      }
    } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
      exit_requested_ = true;
    } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
      const auto* changed =
          reinterpret_cast<const XrEventDataReferenceSpaceChangePending*>(&event);
      if (changed->session == session_) {
        // A recenter changes the robot-space transform discontinuously. Make
        // focused() false in this iteration so no post-reset pose is emitted,
        // then let the Activity close RTC/WSS and require a new PC session.
        session_state_ = XR_SESSION_STATE_UNKNOWN;
        exit_requested_ = true;
      }
    }
    event = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
  }
}

void OpenXrCapture::HandleSessionState(XrSessionState state) {
  session_state_ = state;
  if (state == XR_SESSION_STATE_READY && !session_running_) {
    XrSessionBeginInfo begin_info{XR_TYPE_SESSION_BEGIN_INFO};
    begin_info.primaryViewConfigurationType = kViewConfiguration;
    Check(xrBeginSession(session_, &begin_info), "xrBeginSession");
    session_running_ = true;
  } else if (state == XR_SESSION_STATE_STOPPING && session_running_) {
    session_running_ = false;
    Check(xrEndSession(session_), "xrEndSession");
  } else if (state == XR_SESSION_STATE_EXITING ||
             state == XR_SESSION_STATE_LOSS_PENDING) {
    exit_requested_ = true;
  }
}

bool OpenXrCapture::RenderFrame(FrameSample* sample) {
  if (!session_running_) {
    return false;
  }
  XrFrameWaitInfo wait_info{XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState frame_state{XR_TYPE_FRAME_STATE};
  Check(xrWaitFrame(session_, &wait_info, &frame_state), "xrWaitFrame");
  XrFrameBeginInfo begin_info{XR_TYPE_FRAME_BEGIN_INFO};
  Check(xrBeginFrame(session_, &begin_info), "xrBeginFrame");

  std::vector<XrCompositionLayerProjectionView> projection_views;
  if (frame_state.shouldRender == XR_TRUE) {
    RenderProjection(frame_state.predictedDisplayTime, &projection_views);
  }

  const bool sample_ready = focused() && sample != nullptr;
  if (sample_ready) {
    SyncActions(frame_state.predictedDisplayTime, sample);
  }

  XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  projection.space = floor_space_;
  projection.viewCount = projection_views.size();
  projection.views = projection_views.data();
  const XrCompositionLayerBaseHeader* layer =
      projection_views.empty()
      ? nullptr
      : reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);
  XrFrameEndInfo end_info{XR_TYPE_FRAME_END_INFO};
  end_info.displayTime = frame_state.predictedDisplayTime;
  end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  end_info.layerCount = layer == nullptr ? 0 : 1;
  end_info.layers = layer == nullptr ? nullptr : &layer;
  Check(xrEndFrame(session_, &end_info), "xrEndFrame");
  return sample_ready;
}

void OpenXrCapture::SyncActions(XrTime predicted_display_time, FrameSample* sample) {
  XrActiveActionSet active{action_set_, XR_NULL_PATH};
  XrActionsSyncInfo sync_info{XR_TYPE_ACTIONS_SYNC_INFO};
  sync_info.countActiveActionSets = 1;
  sync_info.activeActionSets = &active;
  Check(xrSyncActions(session_, &sync_info), "xrSyncActions");

  sample->head = LocatePose(view_space_, predicted_display_time, true);
  const auto fill_hand = [&](std::size_t index, PoseSample* pose, ControllerSample* input) {
    const bool interaction_profile_allowed =
        HasAllowedInteractionProfile(index);
    XrActionStateGetInfo get_info{XR_TYPE_ACTION_STATE_GET_INFO};
    get_info.subactionPath = hand_paths_[index];

    get_info.action = grip_pose_action_;
    XrActionStatePose pose_state{XR_TYPE_ACTION_STATE_POSE};
    Check(xrGetActionStatePose(session_, &get_info, &pose_state),
          "xrGetActionStatePose");
    *pose = LocatePose(
        grip_spaces_[index],
        predicted_display_time,
        interaction_profile_allowed && pose_state.isActive == XR_TRUE);

    get_info.action = squeeze_action_;
    XrActionStateFloat squeeze{XR_TYPE_ACTION_STATE_FLOAT};
    Check(xrGetActionStateFloat(session_, &get_info, &squeeze),
          "xrGetActionStateFloat");
    get_info.action = thumbstick_action_;
    XrActionStateVector2f thumbstick{XR_TYPE_ACTION_STATE_VECTOR2F};
    Check(xrGetActionStateVector2f(session_, &get_info, &thumbstick),
          "xrGetActionStateVector2f");

    const double squeeze_value = interaction_profile_allowed &&
            squeeze.isActive == XR_TRUE
        ? std::clamp(static_cast<double>(squeeze.currentState), 0.0, 1.0)
        : 0.0;
    const double axis_x = interaction_profile_allowed &&
            thumbstick.isActive == XR_TRUE
        ? std::clamp(static_cast<double>(thumbstick.currentState.x), -1.0, 1.0)
        : 0.0;
    const double axis_y = interaction_profile_allowed &&
            thumbstick.isActive == XR_TRUE
        ? std::clamp(static_cast<double>(thumbstick.currentState.y), -1.0, 1.0)
        : 0.0;
    *input = ControllerSample{
        .active = interaction_profile_allowed && pose->valid &&
            squeeze.isActive == XR_TRUE &&
            thumbstick.isActive == XR_TRUE,
        .xr_standard = true,
        .correct_handedness = interaction_profile_allowed,
        .tracked_pointer = pose->valid,
        .has_grip_space = pose->valid,
        .squeeze_pressed = squeeze_value >= 0.75,
        .axes = {0.0, 0.0, axis_x, axis_y},
        .buttons = {0.0, squeeze_value},
    };
  };
  fill_hand(0, &sample->left_controller, &sample->left_input);
  fill_hand(1, &sample->right_controller, &sample->right_input);
  sample->distinct_input_sources =
      sample->left_input.correct_handedness &&
      sample->right_input.correct_handedness;
  sample->monotonic_ns = MonotonicNanoseconds();
}

bool OpenXrCapture::HasAllowedInteractionProfile(
    std::size_t hand_index) const {
  if (hand_index >= hand_paths_.size()) {
    throw std::out_of_range("OpenXR hand index is out of range");
  }
  XrInteractionProfileState state{XR_TYPE_INTERACTION_PROFILE_STATE};
  Check(xrGetCurrentInteractionProfile(
            session_, hand_paths_[hand_index], &state),
        "xrGetCurrentInteractionProfile");
  if (state.interactionProfile == XR_NULL_PATH) {
    return false;
  }
  return std::find(
             allowed_interaction_profiles_.begin(),
             allowed_interaction_profiles_.end(),
             state.interactionProfile) != allowed_interaction_profiles_.end();
}

PoseSample OpenXrCapture::LocatePose(
    XrSpace space,
    XrTime predicted_display_time,
    bool active) const {
  if (!active || space == XR_NULL_HANDLE) {
    return {};
  }
  XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
  Check(xrLocateSpace(
            space, floor_space_, predicted_display_time, &location),
        "xrLocateSpace");
  if (!FullyTracked(location.locationFlags)) {
    return {};
  }
  return PoseSample{
      .valid = true,
      .emulated = false,
      .position = {
          location.pose.position.x,
          location.pose.position.y,
          location.pose.position.z,
      },
      .orientation = {
          location.pose.orientation.x,
          location.pose.orientation.y,
          location.pose.orientation.z,
          location.pose.orientation.w,
      },
  };
}

void OpenXrCapture::RenderProjection(
    XrTime predicted_display_time,
    std::vector<XrCompositionLayerProjectionView>* projection_views) {
  std::vector<XrView> views(
      view_configuration_.size(), XrView{XR_TYPE_VIEW});
  XrViewLocateInfo locate_info{XR_TYPE_VIEW_LOCATE_INFO};
  locate_info.viewConfigurationType = kViewConfiguration;
  locate_info.displayTime = predicted_display_time;
  locate_info.space = floor_space_;
  XrViewState view_state{XR_TYPE_VIEW_STATE};
  std::uint32_t view_count = 0;
  Check(xrLocateViews(
            session_,
            &locate_info,
            &view_state,
            views.size(),
            &view_count,
            views.data()),
        "xrLocateViews");
  constexpr XrViewStateFlags required =
      XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
  if ((view_state.viewStateFlags & required) != required ||
      view_count != swapchains_.size()) {
    return;
  }

  projection_views->assign(
      view_count, XrCompositionLayerProjectionView{
                      XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});
  for (std::size_t index = 0; index < swapchains_.size(); ++index) {
    auto& swapchain = swapchains_[index];
    std::uint32_t image_index = 0;
    XrSwapchainImageAcquireInfo acquire_info{
        XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    Check(xrAcquireSwapchainImage(
              swapchain.handle, &acquire_info, &image_index),
          "xrAcquireSwapchainImage");
    XrSwapchainImageWaitInfo wait_info{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait_info.timeout = XR_INFINITE_DURATION;
    Check(xrWaitSwapchainImage(swapchain.handle, &wait_info),
          "xrWaitSwapchainImage");

    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        swapchain.images.at(image_index).image,
        0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      glDeleteFramebuffers(1, &framebuffer);
      throw std::runtime_error("OpenXR swapchain framebuffer is incomplete");
    }
    glViewport(0, 0, swapchain.width, swapchain.height);
    const float streaming = focused() ? 0.08F : 0.02F;
    glClearColor(0.01F, streaming, 0.10F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer);

    XrSwapchainImageReleaseInfo release_info{
        XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    Check(xrReleaseSwapchainImage(swapchain.handle, &release_info),
          "xrReleaseSwapchainImage");
    auto& projection_view = projection_views->at(index);
    projection_view.pose = views[index].pose;
    projection_view.fov = views[index].fov;
    projection_view.subImage.swapchain = swapchain.handle;
    projection_view.subImage.imageRect.offset = {0, 0};
    projection_view.subImage.imageRect.extent = {
        swapchain.width,
        swapchain.height,
    };
    projection_view.subImage.imageArrayIndex = 0;
  }
}

std::int64_t OpenXrCapture::MonotonicNanoseconds() const {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - monotonic_origin_).count();
  if (elapsed < 0 || elapsed >= kMaxSafeWireInteger) {
    throw std::range_error("native monotonic clock exhausted wire range");
  }
  return elapsed + 1;
}

void OpenXrCapture::Check(XrResult result, const char* operation) const {
  if (XR_SUCCEEDED(result)) {
    return;
  }
  char result_name[XR_MAX_RESULT_STRING_SIZE] = {};
  if (instance_ != XR_NULL_HANDLE) {
    static_cast<void>(xrResultToString(instance_, result, result_name));
  }
  const std::string message = std::string(operation) + " failed: " +
      (result_name[0] == '\0' ? std::to_string(result) : result_name);
  throw std::runtime_error(message);
}

void OpenXrCapture::Destroy() noexcept {
  const auto destroy_space = [](XrSpace* space) {
    if (*space != XR_NULL_HANDLE) {
      static_cast<void>(xrDestroySpace(*space));
      *space = XR_NULL_HANDLE;
    }
  };
  for (auto& space : grip_spaces_) {
    destroy_space(&space);
  }
  destroy_space(&view_space_);
  destroy_space(&floor_space_);
  for (auto& swapchain : swapchains_) {
    if (swapchain.handle != XR_NULL_HANDLE) {
      static_cast<void>(xrDestroySwapchain(swapchain.handle));
      swapchain.handle = XR_NULL_HANDLE;
    }
  }
  swapchains_.clear();
  if (action_set_ != XR_NULL_HANDLE) {
    static_cast<void>(xrDestroyActionSet(action_set_));
    action_set_ = XR_NULL_HANDLE;
  }
  if (session_ != XR_NULL_HANDLE) {
    if (session_running_) {
      static_cast<void>(xrEndSession(session_));
    }
    static_cast<void>(xrDestroySession(session_));
    session_ = XR_NULL_HANDLE;
    session_running_ = false;
  }
  if (instance_ != XR_NULL_HANDLE) {
    static_cast<void>(xrDestroyInstance(instance_));
    instance_ = XR_NULL_HANDLE;
  }
  if (egl_display_ != EGL_NO_DISPLAY) {
    static_cast<void>(eglMakeCurrent(
        egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT));
    if (egl_surface_ != EGL_NO_SURFACE) {
      static_cast<void>(eglDestroySurface(egl_display_, egl_surface_));
      egl_surface_ = EGL_NO_SURFACE;
    }
    if (egl_context_ != EGL_NO_CONTEXT) {
      static_cast<void>(eglDestroyContext(egl_display_, egl_context_));
      egl_context_ = EGL_NO_CONTEXT;
    }
    static_cast<void>(eglTerminate(egl_display_));
    egl_display_ = EGL_NO_DISPLAY;
  }
}

}  // namespace motus::quest_capture
