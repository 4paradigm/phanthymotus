#include <android/log.h>
#include <android/native_activity.h>
#include <android/window.h>
#include <android_native_app_glue.h>
#include <jni.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "capture_transport.hpp"
#include "motus/quest_capture/enrollment.hpp"
#include "openxr_capture.hpp"

namespace motus::quest_capture {
namespace {

constexpr char kLogTag[] = "MotusOpenXrCapture";
constexpr char kPreferencesName[] = "motus_capture";
constexpr char kAppVersion[] = "0.2.0";

struct AndroidLifecycle {
  bool resumed{false};
  bool window_ready{false};
};

void Log(android_LogPriority priority, const std::string& message) {
  __android_log_write(priority, kLogTag, message.c_str());
}

void ThrowIfJavaException(JNIEnv* environment, const char* operation) {
  if (environment->ExceptionCheck() == JNI_FALSE) {
    return;
  }
  environment->ExceptionClear();
  throw std::runtime_error(std::string(operation) + " raised a Java exception");
}

std::string JavaString(JNIEnv* environment, jstring value) {
  if (value == nullptr) {
    return {};
  }
  const char* characters = environment->GetStringUTFChars(value, nullptr);
  ThrowIfJavaException(environment, "GetStringUTFChars");
  std::string result = characters == nullptr ? std::string{} : std::string(characters);
  if (characters != nullptr) {
    environment->ReleaseStringUTFChars(value, characters);
  }
  return result;
}

jstring NewJavaString(JNIEnv* environment, const std::string& value) {
  jstring result = environment->NewStringUTF(value.c_str());
  ThrowIfJavaException(environment, "NewStringUTF");
  return result;
}

std::string IntentExtra(
    JNIEnv* environment,
    jobject activity,
    const std::string& name) {
  jclass activity_class = environment->GetObjectClass(activity);
  jmethodID get_intent = environment->GetMethodID(
      activity_class, "getIntent", "()Landroid/content/Intent;");
  jobject intent = environment->CallObjectMethod(activity, get_intent);
  ThrowIfJavaException(environment, "Activity.getIntent");
  jclass intent_class = environment->GetObjectClass(intent);
  jmethodID get_string_extra = environment->GetMethodID(
      intent_class,
      "getStringExtra",
      "(Ljava/lang/String;)Ljava/lang/String;");
  jstring key = NewJavaString(environment, name);
  auto value = static_cast<jstring>(
      environment->CallObjectMethod(intent, get_string_extra, key));
  ThrowIfJavaException(environment, "Intent.getStringExtra");
  const std::string result = JavaString(environment, value);
  if (value != nullptr) {
    environment->DeleteLocalRef(value);
  }
  environment->DeleteLocalRef(key);
  environment->DeleteLocalRef(intent_class);
  environment->DeleteLocalRef(intent);
  environment->DeleteLocalRef(activity_class);
  return result;
}

jobject Preferences(JNIEnv* environment, jobject activity) {
  jclass activity_class = environment->GetObjectClass(activity);
  jmethodID get_preferences = environment->GetMethodID(
      activity_class,
      "getSharedPreferences",
      "(Ljava/lang/String;I)Landroid/content/SharedPreferences;");
  jstring name = NewJavaString(environment, kPreferencesName);
  jobject preferences = environment->CallObjectMethod(
      activity, get_preferences, name, 0);
  ThrowIfJavaException(environment, "Activity.getSharedPreferences");
  environment->DeleteLocalRef(name);
  environment->DeleteLocalRef(activity_class);
  return preferences;
}

std::string GetPreference(
    JNIEnv* environment,
    jobject activity,
    const std::string& key_name) {
  jobject preferences = Preferences(environment, activity);
  jclass preferences_class = environment->GetObjectClass(preferences);
  jmethodID get_string = environment->GetMethodID(
      preferences_class,
      "getString",
      "(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
  jstring key = NewJavaString(environment, key_name);
  jstring fallback = NewJavaString(environment, "");
  auto value = static_cast<jstring>(environment->CallObjectMethod(
      preferences, get_string, key, fallback));
  ThrowIfJavaException(environment, "SharedPreferences.getString");
  const std::string result = JavaString(environment, value);
  environment->DeleteLocalRef(value);
  environment->DeleteLocalRef(fallback);
  environment->DeleteLocalRef(key);
  environment->DeleteLocalRef(preferences_class);
  environment->DeleteLocalRef(preferences);
  return result;
}

void PutCaptureEnrollment(
    JNIEnv* environment,
    jobject activity,
    const CaptureIdentity& identity,
    const std::string& websocket_url,
    const std::string& ca_certificate_pem) {
  jobject preferences = Preferences(environment, activity);
  jclass preferences_class = environment->GetObjectClass(preferences);
  jmethodID edit = environment->GetMethodID(
      preferences_class, "edit", "()Landroid/content/SharedPreferences$Editor;");
  jobject editor = environment->CallObjectMethod(preferences, edit);
  ThrowIfJavaException(environment, "SharedPreferences.edit");
  jclass editor_class = environment->GetObjectClass(editor);
  jmethodID put_string = environment->GetMethodID(
      editor_class,
      "putString",
      "(Ljava/lang/String;Ljava/lang/String;)Landroid/content/SharedPreferences$Editor;");
  jmethodID commit = environment->GetMethodID(editor_class, "commit", "()Z");
  const auto put = [&](const char* key_text, const std::string& value_text) {
    jstring key = NewJavaString(environment, key_text);
    jstring value = NewJavaString(environment, value_text);
    static_cast<void>(environment->CallObjectMethod(editor, put_string, key, value));
    ThrowIfJavaException(environment, "SharedPreferences.Editor.putString");
    environment->DeleteLocalRef(value);
    environment->DeleteLocalRef(key);
  };
  // Commit identity and its pinned transport origin as one editor transaction.
  // A stored credential is never valid independently of these exact values.
  put("capture_id", identity.capture_id);
  put("capture_credential", identity.capture_credential);
  put("core_wss_url", websocket_url);
  put("ca_certificate_pem", ca_certificate_pem);
  const jboolean committed = environment->CallBooleanMethod(editor, commit);
  ThrowIfJavaException(environment, "SharedPreferences.Editor.commit");
  if (committed != JNI_TRUE) {
    throw std::runtime_error("capture preference commit failed");
  }
  environment->DeleteLocalRef(editor_class);
  environment->DeleteLocalRef(editor);
  environment->DeleteLocalRef(preferences_class);
  environment->DeleteLocalRef(preferences);
}

std::string Base64Decode(const std::string& encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  if (encoded.empty() || encoded.size() > 64 * 1024 || encoded.size() % 4 != 0) {
    throw std::invalid_argument("CA certificate base64 is invalid");
  }
  std::string result;
  result.reserve((encoded.size() / 4) * 3);
  std::uint32_t accumulator = 0;
  int bits = 0;
  bool padding = false;
  for (const char character : encoded) {
    if (character == '=') {
      padding = true;
      continue;
    }
    if (padding) {
      throw std::invalid_argument("CA certificate base64 padding is invalid");
    }
    const auto position = alphabet.find(character);
    if (position == std::string_view::npos) {
      throw std::invalid_argument("CA certificate base64 character is invalid");
    }
    accumulator = (accumulator << 6U) | static_cast<std::uint32_t>(position);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      result.push_back(static_cast<char>((accumulator >> bits) & 0xffU));
    }
  }
  if (result.find("-----BEGIN CERTIFICATE-----") == std::string::npos ||
      result.size() > 32 * 1024) {
    throw std::invalid_argument("CA certificate PEM is invalid");
  }
  return result;
}

CaptureTransportConfiguration LoadConfiguration(
    JNIEnv* environment,
    jobject activity) {
  CaptureTransportConfiguration configuration;
  configuration.app_version = kAppVersion;

  const auto pairing_id = IntentExtra(environment, activity, "pairing_id");
  const auto pairing_code = IntentExtra(environment, activity, "pairing_code");
  const auto launch_websocket_url = IntentExtra(
      environment, activity, "core_wss_url");
  const auto launch_ca_base64 = IntentExtra(
      environment, activity, "ca_certificate_base64");
  const bool pairing_present = !pairing_id.empty() || !pairing_code.empty();
  const CaptureLaunchBootstrap launch{
      .pairing_id = pairing_id,
      .pairing_code = pairing_code,
      .websocket_url = launch_websocket_url,
      .ca_certificate_pem = pairing_present && !launch_ca_base64.empty()
          ? Base64Decode(launch_ca_base64)
          : std::string{},
      .transport_override_present =
          !launch_websocket_url.empty() || !launch_ca_base64.empty(),
  };
  StoredCaptureEnrollment stored;
  if (!pairing_present) {
    stored = StoredCaptureEnrollment{
        .capture_id = GetPreference(environment, activity, "capture_id"),
        .capture_credential = GetPreference(
            environment, activity, "capture_credential"),
        .websocket_url = GetPreference(
            environment, activity, "core_wss_url"),
        .ca_certificate_pem = GetPreference(
            environment, activity, "ca_certificate_pem"),
    };
  }
  auto selected = SelectCaptureEnrollment(stored, launch);
  configuration.identity = std::move(selected.identity);
  configuration.pairing = std::move(selected.pairing);
  configuration.websocket_url = std::move(selected.websocket_url);
  configuration.ca_certificate_pem =
      std::move(selected.ca_certificate_pem);
  return configuration;
}

void HandleAppCommand(android_app* app, std::int32_t command) {
  auto* lifecycle = static_cast<AndroidLifecycle*>(app->userData);
  if (lifecycle == nullptr) {
    return;
  }
  switch (command) {
    case APP_CMD_RESUME:
      lifecycle->resumed = true;
      break;
    case APP_CMD_PAUSE:
    case APP_CMD_STOP:
      lifecycle->resumed = false;
      break;
    case APP_CMD_INIT_WINDOW:
      lifecycle->window_ready = true;
      break;
    case APP_CMD_TERM_WINDOW:
    case APP_CMD_DESTROY:
      lifecycle->window_ready = false;
      break;
    default:
      break;
  }
}

}  // namespace
}  // namespace motus::quest_capture

void android_main(android_app* app) {
  using namespace motus::quest_capture;
  JNIEnv* environment = nullptr;
  bool attached = false;
  try {
    if (app == nullptr || app->activity == nullptr || app->activity->vm == nullptr) {
      throw std::runtime_error("NativeActivity is unavailable");
    }
    const jint environment_result = app->activity->vm->GetEnv(
        reinterpret_cast<void**>(&environment), JNI_VERSION_1_6);
    if (environment_result == JNI_EDETACHED) {
      if (app->activity->vm->AttachCurrentThread(&environment, nullptr) != JNI_OK) {
        throw std::runtime_error("AttachCurrentThread failed");
      }
      attached = true;
    } else if (environment_result != JNI_OK || environment == nullptr) {
      throw std::runtime_error("JNI environment unavailable");
    }

    ANativeActivity_setWindowFlags(
        app->activity,
        AWINDOW_FLAG_KEEP_SCREEN_ON,
        0);
    AndroidLifecycle lifecycle;
    app->userData = &lifecycle;
    app->onAppCmd = HandleAppCommand;

    auto configuration = LoadConfiguration(
        environment, app->activity->clazz);
    // Construct the transport after OpenXR so stack unwinding always tears
    // down WSS/RTC before destroying the runtime and its swapchains.
    OpenXrCapture xr;
    xr.Initialize(app);
    const std::string enrollment_websocket_url = configuration.websocket_url;
    const std::string enrollment_ca_certificate_pem =
        configuration.ca_certificate_pem;
    auto transport = std::make_shared<CaptureTransport>(
        std::move(configuration),
        [environment,
         activity = app->activity->clazz,
         enrollment_websocket_url,
         enrollment_ca_certificate_pem](const CaptureIdentity& identity) {
          PutCaptureEnrollment(
              environment,
              activity,
              identity,
              enrollment_websocket_url,
              enrollment_ca_certificate_pem);
        },
        [](const std::string& message) { Log(ANDROID_LOG_INFO, message); });
    transport->Start();

    bool last_focused = false;
    while (app->destroyRequested == 0 && !xr.exit_requested()) {
      for (;;) {
        int events = 0;
        android_poll_source* source = nullptr;
        const int timeout_ms = xr.session_running() ? 0 : 50;
        if (ALooper_pollOnce(
                timeout_ms,
                nullptr,
                &events,
                reinterpret_cast<void**>(&source)) < 0) {
          break;
        }
        if (source != nullptr) {
          source->process(app, source);
        }
        if (timeout_ms != 0) {
          break;
        }
      }

      xr.PollEvents();
      const bool focused = lifecycle.resumed && xr.focused();
      if (focused != last_focused) {
        transport->SetXrFocused(focused);
        last_focused = focused;
      }
      transport->Tick();

      if (xr.session_running()) {
        FrameSample sample;
        if (xr.RenderFrame(&sample) && focused) {
          transport->SendFrame(sample);
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    transport->SetXrFocused(false);
    transport->Shutdown();
    ANativeActivity_finish(app->activity);
  } catch (const std::exception& error) {
    Log(ANDROID_LOG_ERROR, std::string("capture app stopped: ") + error.what());
    if (app != nullptr && app->activity != nullptr) {
      ANativeActivity_finish(app->activity);
    }
  } catch (...) {
    Log(ANDROID_LOG_ERROR, "capture app stopped: unknown error");
    if (app != nullptr && app->activity != nullptr) {
      ANativeActivity_finish(app->activity);
    }
  }
  if (attached && app != nullptr && app->activity != nullptr &&
      app->activity->vm != nullptr) {
    app->activity->vm->DetachCurrentThread();
  }
}
