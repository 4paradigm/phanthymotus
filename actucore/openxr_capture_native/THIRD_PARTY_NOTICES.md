# Third-party components

- Khronos OpenXR Android loader 1.1.60, Apache-2.0. The Android lifecycle,
  loader initialization, and session-state handling follow the public
  `hello_xr` sample from OpenXR-SDK-Source release 1.1.60.
- libdatachannel 0.24.3 (`c47f5d77...`), MPL-2.0, built without media support.
- Mbed TLS 3.6.7 (`068ff080...`), Apache-2.0 or GPL-2.0-or-later; this build
  consumes it under Apache-2.0.
- nlohmann/json is consumed at libdatachannel's pinned submodule revision and
  is MIT licensed.

All dependencies are pinned by immutable source revision or Maven version in
the build files. Their complete license texts remain available from their
respective distributions.
