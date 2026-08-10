cmake_minimum_required(VERSION 3.22.1)
project(motus_quest_capture LANGUAGES C CXX)

include(FetchContent)

set(MOTUS_CAPTURE_HEADSET "meta" CACHE STRING "OpenXR headset target: meta or pico")
set_property(CACHE MOTUS_CAPTURE_HEADSET PROPERTY STRINGS meta pico)
if(MOTUS_CAPTURE_HEADSET STREQUAL "meta")
    set(motus_capture_headset_definition MOTUS_CAPTURE_HEADSET_META=1)
elseif(MOTUS_CAPTURE_HEADSET STREQUAL "pico")
    set(motus_capture_headset_definition MOTUS_CAPTURE_HEADSET_PICO=1)
else()
    message(FATAL_ERROR "MOTUS_CAPTURE_HEADSET must be 'meta' or 'pico'")
endif()

# This is an Android application build, not an installable CMake package.  In
# particular, libdatachannel's install export cannot describe our in-tree
# static Mbed TLS target.  Suppress dependency install/export rules instead of
# weakening the TLS linkage or patching third-party sources.
set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(DISABLE_PACKAGE_CONFIG_AND_INSTALL ON CACHE BOOL "" FORCE)
set(
    MBEDTLS_USER_CONFIG_FILE
    "${CMAKE_CURRENT_LIST_DIR}/mbedtls_user_config.h"
    CACHE FILEPATH "" FORCE
)
FetchContent_Declare(
    mbedtls
    GIT_REPOSITORY https://github.com/Mbed-TLS/mbedtls.git
    GIT_TAG 068ff080b369adfac81509f9b57b2afabaf82dc5
    GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(mbedtls)
if(NOT TARGET MbedTLS::MbedTLS)
    add_library(MbedTLS::MbedTLS ALIAS mbedtls)
endif()

set(USE_MBEDTLS ON CACHE BOOL "" FORCE)
set(NO_MEDIA ON CACHE BOOL "" FORCE)
set(NO_EXAMPLES ON CACHE BOOL "" FORCE)
set(NO_TESTS ON CACHE BOOL "" FORCE)
set(WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_DEPS_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    libdatachannel
    GIT_REPOSITORY https://github.com/paullouisageneau/libdatachannel.git
    GIT_TAG c47f5d77c124c35c31ac8378ad613295a124d354
    GIT_SUBMODULES
        deps/json
        deps/libjuice
        deps/plog
        deps/usrsctp
    GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(libdatachannel)
if(NOT TARGET nlohmann_json::nlohmann_json)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(
        "${libdatachannel_SOURCE_DIR}/deps/json"
        "${libdatachannel_BINARY_DIR}/deps/json"
        EXCLUDE_FROM_ALL
    )
endif()

find_package(OpenXR REQUIRED CONFIG)
find_library(android-lib android REQUIRED)
find_library(egl-lib EGL REQUIRED)
find_library(gles-lib GLESv3 REQUIRED)
find_library(log-lib log REQUIRED)

set(project_root "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(native_app_glue "${ANDROID_NDK}/sources/android/native_app_glue")

add_library(android_native_app_glue STATIC "${native_app_glue}/android_native_app_glue.c")
target_include_directories(android_native_app_glue PUBLIC "${native_app_glue}")

add_library(motus_quest_capture SHARED
    android_main.cpp
    capture_transport.cpp
    openxr_capture.cpp
    runtime_profile.cpp
    "${project_root}/src/capture_session.cpp"
    "${project_root}/src/capture_wire.cpp"
    "${project_root}/src/enrollment.cpp"
    "${project_root}/src/frame_v1.cpp"
)
target_include_directories(motus_quest_capture PRIVATE
    "${project_root}/include"
)
target_compile_features(motus_quest_capture PRIVATE cxx_std_20)
target_compile_definitions(motus_quest_capture PRIVATE
    ${motus_capture_headset_definition}
    XR_USE_PLATFORM_ANDROID
    XR_USE_GRAPHICS_API_OPENGL_ES
)
target_compile_options(motus_quest_capture PRIVATE
    -Wall
    -Wextra
    -Werror
    -Wpedantic
    # OpenXR's C structs intentionally use aggregate initialization with the
    # mandatory type tag while every remaining member is zero-initialized.
    -Wno-missing-field-initializers
)
target_link_options(motus_quest_capture PRIVATE "-uANativeActivity_onCreate")
target_link_libraries(motus_quest_capture PRIVATE
    android_native_app_glue
    OpenXR::openxr_loader
    LibDataChannel::LibDataChannelStatic
    nlohmann_json::nlohmann_json
    ${android-lib}
    ${egl-lib}
    ${gles-lib}
    ${log-lib}
)
