# PhanthyMotus Android OpenXR Capture

This ActuCore-owned module provides the native Android OpenXR pose
endpoint for the generic teleoperation runtime. One shared codebase produces
separate Meta Quest and PICO APKs. Both send the public
`motus.teleop.rtc-frame.v1` contract directly to ActuCore over WebRTC. The
Capture app does not own the Driver's teleoperation session, lease, fence,
operator authorization, start action, or robot-output configuration.

## 本次迁移

首版验收设备为 PICO 4 Ultra；原生端归属 ActuCore，天轶 Driver 不接收 OpenXR 协议。
新增 reconstruction 透视层（`XR_FB_passthrough`），双握把统一使能，扳机连续控制手开合。
透视不可用、失焦或空间重置会撤销输入；重新使能前必须松握把并把扳机回中立。
Meta flavor 保留上游共享构建能力，不在本次验收范围。来源与修改见 `../plugins/teleop/NOTICE.md`。

## Supported build targets

| Build target | Target device family | Application ID | Debug APK |
| --- | --- | --- | --- |
| `meta` | Meta Quest 3 | `com.phanthymotus.questcapture` | `app/build/outputs/apk/meta/debug/app-meta-debug.apk` |
| `pico` | PICO 4 / 4 Enterprise / 4 Ultra / Ultra Enterprise | `com.phanthymotus.picocapture` | `app/build/outputs/apk/pico/debug/app-pico-debug.apk` |

The application IDs intentionally remain compatible with the existing Meta and
PICO lab packages. An in-place upgrade also requires the same signing key; a
debug APK signed on another workstation may require uninstalling the old build,
which clears the stored Capture credential.

The PICO target uses the standard Khronos Android OpenXR loader, which PICO
supports starting with PICO OS 5.9.0. Update the headset to at least that version
before testing. See PICO's [standard-loader announcement][pico-loader] and
[Native SDK release notes][pico-native].

Use PICO OS 5.13.0 or newer as the shared acceptance baseline. At startup the
same PICO APK enables only the controller extensions actually exposed by the
runtime: `XR_BD_controller_interaction` for PICO 4-class controllers and
`XR_BD_ultra_controller_interaction` for PICO 4 Ultra-class controllers. If
neither current Khronos profile is available, the app stops instead of emitting
apparently valid input from an unknown controller.

2026-09-09 已在 PICO 4 Ultra、PICO OS 5.15.7 上完成首轮本机硬件验证：
APK 安装并进入 `FOCUSED` OpenXR 会话，reconstruction 透视层创建成功，头显和
双控制器跟踪有效，左右输入源有效；双握把与双扳机连续读到 `1.00/1.00`，松开后
回到 `0.00/0.00`。随后真实 WSS 配对和两条 RTC DataChannel 建连成功，
ActuCore 零输出适配器收到超过 1,300 帧真实输入。本次 WSS 使用 ADB/SSH 转发，
不能代替现场局域网、失焦/跟踪丢失 HOLD、完整恢复或 Driver 端到端验收；
具体证据与限制见 [交付记录](../../docs/plans/tianyi-teleop-pr.md)。

[pico-loader]: https://developer.picoxr.com/blog/muz6s63x/
[pico-native]: https://developer.picoxr.com/document/native/

## Operator flow

1. Open the Android connection page, discover/select the robot and request pairing.
2. Approve the fingerprint using a compatible management interface. The credential remains in private Android storage.
3. Reconnect from the headset and enter passthrough capture. The legacy ADB bootstrap below remains available for laboratory use.
4. Use the paired in-headset start, finish-and-return, and immediate-stop buttons. Both grips enable following; release either grip to hold. Finish-and-return is distinct from stop.

The dedicated Canvas pairing panel requires separate Agent Core integration and is not included in this PR.

An Android or OpenXR system permission prompt may still appear on first install.
本次零售版 PICO 4 Ultra 在未登录 PICO 账号时被系统 entitlement 流程中止；登录后
首次启动还需在头显内完成 MR 安全区提示，再重新启动 Capture。
The application cannot collect controller poses while Android backgrounds it or
OpenXR removes `FOCUSED` input ownership; that transition closes RTC locally and
ActuCore and Driver independently force the teleoperation session into HOLD.

The pairing binds the Capture credential to the exact ActuCore
`wss://.../ws/teleop-capture` origin and deployment CA. Resume cannot override
either value. WebRTC uses an ordered `teleop-control` DataChannel for liveness
and an unordered, zero-retransmit `teleop-pose` DataChannel for current poses.
Focus loss, tracking/reference-space recenter, RTC failure, or pose
backpressure closes the local streaming path instead of continuing with stale
authority or coordinates.

The supplied RTC configuration is a direct-LAN path and does not configure
STUN/TURN. Cross-NAT or SSH-tunnel operation is not an accepted deployment
shape for this version.

## Build

Required versions are fixed in the project:

- JDK 17
- Android platform 35 and build-tools 35.0.1
- NDK 27.0.12077973
- CMake 3.22.1
- Gradle 8.9 (downloaded and SHA-256 verified by the build script)
- OpenXR Android loader 1.1.60 (AAR SHA-256 verified by Gradle)
- libdatachannel 0.24.3 and Mbed TLS 3.6.7 (immutable Git revisions)

构建脚本不会执行可预测路径里预先展开的 Gradle 程序，也不会在校验后重新打开共享
缓存文件；每次都会先把发行 ZIP 复制或下载到本次构建的私有临时目录，再校验并从
同一个文件解压执行。

After accepting the Android SDK/NDK licenses in your own SDK installation, build
both APKs (the default):

```sh
export JAVA_HOME=/path/to/jdk-17
export ANDROID_SDK_ROOT=/path/to/android-sdk
./scripts/build_android.sh
```

For a shorter incremental build, select one target explicitly:

```sh
./scripts/build_android.sh --platform meta
./scripts/build_android.sh --platform pico
```

The flavors pass `MOTUS_CAPTURE_HEADSET=meta` or
`MOTUS_CAPTURE_HEADSET=pico` to the native build. The Meta APK accepts only the
Oculus Touch profile; the PICO APK dynamically accepts the current PICO 4 and
PICO 4 Ultra profiles described above.

## 日常连接（无需 ADB）

打开应用进入机器人连接页，自动发现局域网内的 teleop 卡片，也可输入地址作为备用。先在 Canvas 卡片打开 120 秒配对窗口；选择设备后两端逐组核对指纹，卡片批准、头显确认后进入原生透视采集。后续启动保留凭据，但仍停留在连接页，点击“连接已配对机器人”才进入采集；失败返回连接页，不能自动运动。忘记设备后还需在卡片撤销旧配对。

连接页是普通 Android Activity；NativeActivity 保留 DUMP 限制，只有同应用或 ADB/system 可启动并注入配置。发现需要 Wi-Fi multicast 权限，无相机扫描、云账户或额外服务。Meta 共用代码但本轮只构建验证 PICO。

## Install and legacy laboratory pairing

Install the APK that matches the headset:

```sh
adb install -r app/build/outputs/apk/meta/debug/app-meta-debug.apk
# Or:
adb install -r app/build/outputs/apk/pico/debug/app-pico-debug.apk
```

Then launch and pair it from the PC:

```sh
export DRIVER_CAPTURE_WSS_URL='wss://actucore-host:15731/ws/teleop-capture'
export PAIRING_ID='the-id-shown-on-the-PC'
export CA_CERT_BASE64='the-public-value-shown-on-the-PC'
./scripts/launch_capture.sh --platform meta
# Or: ./scripts/launch_capture.sh --platform pico
```

The script prompts without terminal echo for the one-time pairing code. Resume a
previously paired app with:

```sh
./scripts/launch_capture.sh --platform meta --resume
./scripts/launch_capture.sh --platform pico --resume
```

When multiple or wireless-ADB devices are visible, target one explicitly without
changing the command:

```sh
ADB_SERIAL='<headset-adb-ip>:5555' \
  ./scripts/launch_capture.sh --platform meta --resume
```

`CA_CERT_FILE=/path/to/public-chain.pem` may be used instead of the Base64 field.
The public PEM chain is limited to 32 KiB after decoding; its canonical Base64
form is therefore at most 43,692 characters. The launcher rejects both encoded
text and decoded PEM that exceed those bounds before invoking ADB.
If the Driver was reinstalled or the enrollment was revoked, generate a new
pairing from the card and run the first-pairing command again. An explicit new
bootstrap replaces the stored credential; clearing app data or using a headset
menu is not required. The exported NativeActivity requires the platform-only
`android.permission.DUMP` caller permission, so ordinary headset apps cannot
inject a pairing origin or replace an enrollment; the ADB shell is the intended
laboratory bootstrap caller. Meta and PICO use different application IDs, so
their credentials cannot be mixed accidentally. Confirm the ADB launch
permission on each target headset OS before treating that SKU as accepted.
The one-time code is intentionally passed as an `adb shell am start` argument;
it is absent from script stdout but can be observed by trusted ADB-host/device
process inspection during its 60-second, single-use lifetime.

No robot connection or robot output is involved in build, pairing, or Shadow
validation.

## Host protocol and launcher tests

```sh
./tests/launch_capture_test.sh

NLOHMANN_JSON_INCLUDE=/path/to/nlohmann/include \
./tests/run_host_tests.sh
```

After building both APKs, verify package identity, permissions, PICO-only
metadata, ABI, ELF dependencies, alignment and signature:

```sh
ANDROID_SDK_ROOT=/path/to/android-sdk \
JAVA_HOME=/path/to/jdk-17 \
python3 ./tests/verify_android_apks.py
```

The launcher test uses a fake ADB executable and covers exact Meta/PICO package
selection, `ADB_SERIAL`, resume, first pairing, secret-free output, CA PEM/Base64
size boundaries, and malformed arguments. The protocol checks cover
release-neutral-regrip, reconnect watermarks, terminal authentication/protocol
error classification, retryable operational errors, exact WSS envelopes,
duplicate-key rejection, one-shot SDP,
exported-Activity origin/CA pinning, and the ActuCore strict RTC frame
parser without starting a publisher.


## 双臂模型与当前验收范围

绿色显示新鲜实测 FK，橙色显示 IK 或带 HELD/STALE 标识的历史模型，粉色显示映射目标。G1 每臂六点、天轶每臂八点；躯干辅助辨认平视背后视角。显示与实际机器人没有外参配准，橙色不代表目标已执行。

PICO 0.3.16 提供透视内中文开始、结束并收臂、立即停止按钮。停止优先，停止/收臂后抑制输入直到显式再次开始。运行时须有 OpenXR 焦点；系统定位窗口遮挡时不能认为握把输入有效。

历史现场记录验证了真实开始请求和服务端收臂，但实体 finish/stop 按钮、持续 15 分钟及 10 轮恢复尚未完成。此 PR 不重新安装 APK 或操作机器人。详见交付记录。
