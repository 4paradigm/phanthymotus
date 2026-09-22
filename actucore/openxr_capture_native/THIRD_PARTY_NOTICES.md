# Third-party components

- Khronos OpenXR Android loader 1.1.60, Apache-2.0. The Android lifecycle,
  loader initialization, and session-state handling follow the public
  `hello_xr` sample from OpenXR-SDK-Source release 1.1.60.
- libdatachannel 0.24.3, MPL-2.0, built without media support. CMake pins the
  immutable annotated-tag object `c47f5d77c124c35c31ac8378ad613295a124d354`,
  which resolves to commit `c6696d157b5612df2a741d9a03b192b47ab6cefb`.
- Mbed TLS 3.6.7 (commit `068ff080b369adfac81509f9b57b2afabaf82dc5`),
  Apache-2.0 or GPL-2.0-or-later; this build
  consumes it under Apache-2.0.
- nlohmann/json is consumed at libdatachannel's pinned submodule revision and
  is MIT licensed.

All dependencies are pinned by immutable source revision or Maven version in
the build files. Their complete license texts remain available from their
respective distributions.

Source integrity: the two Git objects above were checked against the upstream
GitHub Git API on 2026-09-22. Git verifies their content-addressed object graphs;
recursive submodules are fixed by those trees' gitlinks, not by moving branches.
These are Git object IDs, not SHA256 archive checksums. The build uses Git rather
than unverified source archives; Gradle and the OpenXR AAR separately verify
their downloaded archives with SHA256. No native dependency versions or source
fetch mechanism were changed during this review.

The APK now includes complete dependency license texts and immutable source links
in `assets/licenses/`, including libjuice, usrsctp, plog and Android NDK notices.
These files are copied from the pinned build inputs without altering their text.
