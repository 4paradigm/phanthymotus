#pragma once

#include <EGL/egl.h>
#include <jni.h>
#include <openxr/openxr_platform.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "motus/openxr_capture/frame_v1.hpp"
#include "runtime_profile.hpp"
#include "ik_overlay_renderer.hpp"
#include "operator_panel.hpp"

struct android_app;

namespace motus::openxr_capture {

class OpenXrCapture final {
 public:
  OpenXrCapture() = default;
  ~OpenXrCapture();

  OpenXrCapture(const OpenXrCapture&) = delete;
  OpenXrCapture& operator=(const OpenXrCapture&) = delete;

  void Initialize(android_app* app);
  void PollEvents();
  void SetOperatorPanel(bool enabled,const std::string& state,const std::string& mode,const std::string& error) {
    panel_.enabled=enabled;panel_.state=state;panel_.mode=mode;panel_.error=error;
  }
  std::string TakeOperatorAction(){auto result=panel_.pending;panel_.pending.clear();return result;}
  void SetVisualization(const IkVisualization& value) { visualization_ = value; }

  // Drives xrWaitFrame/xrBeginFrame/xrEndFrame and returns a controller sample
  // only while the runtime grants FOCUSED input ownership.
  bool RenderFrame(FrameSample* sample);

  [[nodiscard]] bool session_running() const { return session_running_; }
  [[nodiscard]] bool focused() const { return session_state_ == XR_SESSION_STATE_FOCUSED && passthrough_ready_ && view_visible_; }
  [[nodiscard]] bool exit_requested() const { return exit_requested_; }

 private:
  struct Swapchain {
    XrSwapchain handle{XR_NULL_HANDLE};
    std::int32_t width{0};
    std::int32_t height{0};
    std::vector<XrSwapchainImageOpenGLESKHR> images;
  };

  void InitializeLoader(android_app* app);
  void CreateInstance(android_app* app);
  void CreateEglContext();
  void CreateSession();
  void CreateSpaces();
  void CreatePassthrough();
  void CreateActions();
  void CreateSwapchains();
  void Destroy() noexcept;

  void HandleSessionState(XrSessionState state);
  void SyncActions(XrTime predicted_display_time, FrameSample* sample);
  [[nodiscard]] bool HasAllowedInteractionProfile(std::size_t hand_index) const;
  PoseSample LocatePose(XrSpace space, XrTime predicted_display_time, bool active) const;
  void RenderProjection(
      XrTime predicted_display_time,
      std::vector<XrCompositionLayerProjectionView>* projection_views);
  std::int64_t MonotonicNanoseconds() const;

  void Check(XrResult result, const char* operation) const;

  IkVisualization visualization_;
  IkOverlayRenderer overlay_;
  OperatorPanel panel_;
  XrAction aim_action_{XR_NULL_HANDLE};
  std::array<XrSpace,2> aim_spaces_{XR_NULL_HANDLE,XR_NULL_HANDLE};
  XrInstance instance_{XR_NULL_HANDLE};
  XrSystemId system_id_{XR_NULL_SYSTEM_ID};
  XrSession session_{XR_NULL_HANDLE};
  XrSessionState session_state_{XR_SESSION_STATE_UNKNOWN};
  bool session_running_{false};
  bool exit_requested_{false};

  EGLDisplay egl_display_{EGL_NO_DISPLAY};
  EGLConfig egl_config_{nullptr};
  EGLContext egl_context_{EGL_NO_CONTEXT};
  EGLSurface egl_surface_{EGL_NO_SURFACE};

  XrPassthroughFB passthrough_{XR_NULL_HANDLE};
  XrPassthroughLayerFB passthrough_layer_{XR_NULL_HANDLE};
  PFN_xrDestroyPassthroughFB destroy_passthrough_{nullptr};
  PFN_xrDestroyPassthroughLayerFB destroy_passthrough_layer_{nullptr};
  bool passthrough_ready_{false};
  bool view_visible_{false};
  XrSpace floor_space_{XR_NULL_HANDLE};
  XrSpace view_space_{XR_NULL_HANDLE};
  XrActionSet action_set_{XR_NULL_HANDLE};
  XrAction grip_pose_action_{XR_NULL_HANDLE};
  XrAction squeeze_action_{XR_NULL_HANDLE};
  XrAction trigger_action_{XR_NULL_HANDLE};
  XrAction thumbstick_action_{XR_NULL_HANDLE};
  std::array<XrPath, 2> hand_paths_{};
  std::array<XrSpace, 2> grip_spaces_{XR_NULL_HANDLE, XR_NULL_HANDLE};
  RuntimeSelection runtime_selection_{};
  std::vector<XrPath> allowed_interaction_profiles_;

  std::vector<XrViewConfigurationView> view_configuration_;
  std::vector<Swapchain> swapchains_;
  std::int64_t color_format_{0};
  std::chrono::steady_clock::time_point monotonic_origin_{};
};

}  // namespace motus::openxr_capture
