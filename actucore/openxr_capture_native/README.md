# PICO / Meta 原生遥操客户端

此模块随 ActuCore 维护，使用原生 Android OpenXR 采集头显、双手柄位姿与按键，通过 WSS 配对和 WebRTC 将 `motus.teleop.rtc-frame.v1` 输入发送给 ActuCore。PICO 提供透视画面、模型显示与就地操作入口；映射、IK、会话及执行权由 ActuCore 和 Driver 管理，客户端不直接下发机器人关节命令。

完整操作流程见 [遥操卡片手册](../plugins/teleop/README.md)，服务边界见 [架构说明](../../docs/design/teleop-architecture.md)。日常配对和遥操不需要 ADB、USB 或后端脚本。

## 当前版本与设备范围

当前源码的 Gradle 包版本与 Native 握手版本均为 `0.3.17-operator1-ikview2`，`versionCode = 21`。本手册描述源码行为，不将版本号、宿主测试或 APK 构建视为本轮设备安装和真机验收的证据。

| 构建目标 | 设备系列 | Application ID | Debug APK |
|---|---|---|---|
| `pico` | PICO 4 / 4 Enterprise / 4 Ultra / Ultra Enterprise | `com.phanthymotus.picocapture` | `app/build/outputs/apk/pico/debug/app-pico-debug.apk` |
| `meta` | Meta Quest 3 | `com.phanthymotus.questcapture` | `app/build/outputs/apk/meta/debug/app-meta-debug.apk` |

主要验收设备为 PICO 4 Ultra。Meta flavor 保留共享构建能力，不继承 PICO 的设备验收结果。两个包使用不同应用 ID，配对凭据各自保存。覆盖安装还要求签名一致；卸载重装会清除该应用的本地配对信息。

项目使用标准 Khronos Android OpenXR loader，以 PICO OS 5.13.0 或更新版本作为兼容性检查基线。标准 loader 支持和系统版本要求参考 PICO [loader 说明](https://developer.picoxr.com/blog/muz6s63x/)与 [Native SDK 文档](https://developer.picoxr.com/document/native/)。设备列表和系统版本基线不是所有机型都已实测的声明。

PICO APK 按运行时实际提供的扩展选择 `XR_BD_controller_interaction` 或 `XR_BD_ultra_controller_interaction`；不认识的控制器配置不会被当作有效输入。透视使用 `XR_FB_passthrough` reconstruction layer。Meta APK 使用 Oculus Touch profile。

## 连接与配对

1. 在机器人 Canvas 中添加或打开 **执行（ActuCore）→ teleop** 卡片，展开 **连接与配对**，点击 **允许新设备配对**，打开 120 秒配对窗口。
2. 在 PICO 打开“连接机器人”。应用可发现局域网中的卡片，也可手动输入地址。
3. 选择目标机器人，申请配对；核对 Canvas 和头显显示的指纹，一致后在卡片批准并在头显确认。
4. 进入透视采集页。确认头显和左右手柄跟踪有效；配对成功不表示遥操已经开始。

后续启动仍先显示连接页，点击已配对机器人后进入采集。连接失败会返回连接页，不自动开始 Live。忘记设备只删除头显本地信息；需要同时取消信任时，在 Canvas 撤销配对。ActuCore 配对状态或部署 CA 更换后，需要重新配对；普通 Driver 重建不应单独决定头显凭据是否失效。

连接页是普通 Android Activity；采集页才创建 OpenXR 会话。首次启动时，系统权限、账号或 MR 安全边界提示需要在头显内完成。透视画面可见不代表应用已经取得 `FOCUSED` 输入状态；失焦、跟踪丢失、空间重置或透视不可用时，客户端停止发送可使能的输入，服务端另行处理保持和恢复。

## 开始、保持与结束

先按 [卡片手册](../plugins/teleop/README.md#从-canvas-开始一次遥操)配置模式并标定，再开始会话。Shadow 只计算和显示，Live 才可能取得 Driver 执行权。双手柄侧面的握把共同使能；松开任意握把即保持，再次握住时由服务端按当前手柄和实测机器人姿态建立相对基准。扳机是连续输入，是否输出手部动作取决于机型和标定配置。

天轶服务声明操作能力时，透视页显示三个中文按钮。用任一手柄射线指向按钮并扣一下扳机：

| 按钮 | 客户端行为与服务端职责 |
|---|---|
| 开始遥操 | 要求双握把松开；发送开始请求，由 ActuCore 执行标定和会话准备 |
| 结束并收臂 | 0.3.17 允许按住握把时点击；先抑制后续握把使能，再请求 ActuCore 收臂并释放控制权 |
| 立即停止 | 不要求松握把；同一帧与其他按钮同时触发时优先，抑制输入并请求停止，不主动收臂 |

请求已接收不等于机器人已完成动作。点击结束或停止后，客户端保持输入抑制，直到再次明确点击开始。重连不重发旧操作。按钮状态与失败原因来自服务器；服务端未声明操作能力时隐藏按钮。**G1 当前未接入这套头显操作按钮，也没有天轶的结束收臂能力**；使用 Canvas 中该机型实际声明的操作，不能将按钮缺失当作配对失败。

## 如何看模型

模型采用从机器人背后平视的方向：机器人左侧在画面左侧，前向远离观察者，高度向上。它是随头部显示的模型面板，没有机器人与头显的空间外参，不能当作与透视中的真实机器人或手柄重合的 AR 测量。

| 显示 | 含义 |
|---|---|
| 绿色实测线 | Driver 新鲜关节反馈经 FK 计算的位置，不是视觉测量 |
| 黄／橙色期望线 | IK 结果，不等于已接受或执行到位的目标 |
| 粉色十字 | 手柄映射后的末端目标 |
| `LAST VALID - HELD` | 天轶保留的最后有效 IK 历史快照；与当前解使用相同橙色亮度，依靠文字区分 |
| 白灰躯干与外框 | 方向参考与标定活动边界，不是实测躯干外壳；框内并非所有姿态都可达 |

当前不绘制蓝色命令线。天轶模型每臂八点，G1 每臂六点，解析和历史保留行为按机型区分。绿色反馈过期时隐藏；历史橙色不应解读为仍有有效运动指令。`NO FRESH DATA`、`STALE` 与错误码需要结合卡片输入和执行反馈排查。

当前 `-ikview2` 客户端接收独立的 WSS `visualization` 推送，服务端上限约 30 Hz；实际更新率还取决于输入、IK 和设备。旧 `-ikview1` 通过 `presence_ack` 携带显示，约 4 Hz。只更新 APK 不会让不支持新协议的服务端提供显示。可视化复用配对 WSS，不额外申请执行权或启动另一套服务。

无效 IK 不下发，天轶保持靠近最后有效执行位置，条件恢复后继续；当前没有额外的越界目标搜索。具体保持和故障恢复由服务端负责，不能以头显模型仍显示判断机器人正在执行。

## 通信与信任边界

配对凭据绑定准确的 ActuCore `wss://.../ws/teleop-capture` 地址与部署 CA；恢复连接不能覆盖这些值。默认 WSS 端口为 **15741**，由普通 ActuCore 内的 teleop 插件提供。连续输入使用有序 `teleop-control` DataChannel 和无序、零重传的 `teleop-pose` DataChannel。传输失效、跟踪重置或积压会关闭本地采集传输，避免把旧姿态继续作为当前输入。

当前默认使用局域网直连，没有配置 STUN/TURN；跨 NAT 或 SSH 隧道不是标准部署方式。连接页发现使用 Wi-Fi multicast，不需要二维码相机或新增外部配对服务。NativeActivity 使用 `android.permission.DUMP` 限制外部配置注入，普通第三方应用不能替换配对来源；同应用入口和受信任 ADB 调试仍可启动它。

## 构建与安装

以下命令均从本目录执行。先在自己的 SDK 安装中接受许可证并准备固定依赖：

- JDK 17。
- Android platform 35、build-tools 35.0.1、NDK 27.0.12077973、CMake 3.22.1。
- Gradle 8.9，由构建脚本下载或读取缓存 ZIP，并校验 SHA-256 后在私有临时目录解压执行。
- OpenXR Android loader 1.1.60，AAR 由 Gradle 校验 SHA-256。
- libdatachannel 0.24.3 和 Mbed TLS 3.6.7，CMake 固定完整 Git revision；nlohmann/json 随已固定的子模块构建。

```sh
export JAVA_HOME=/path/to/jdk-17
export ANDROID_SDK_ROOT=/path/to/android-sdk
./scripts/build_android.sh --platform pico
```

不传 `--platform` 默认构建两个 flavor；可用 `--platform meta` 单独构建 Meta。构建入口为 [build_android.sh](scripts/build_android.sh)，依赖定义位于 [Gradle 配置](app/build.gradle.kts)和 [CMake 配置](app/src/main/cpp/CMakeLists.txt)。构建不连接机器人。

```sh
adb install -r app/build/outputs/apk/pico/debug/app-pico-debug.apk
```

Meta 使用表中对应 APK。安装会中断正在运行的应用；成功后应独立核对包版本、连接页、配对、透视、控制器输入和模型显示，不能仅以安装命令成功宣布设备验收完成。本轮文档整理只核对源码版本与入口，不声明重新构建 APK 或安装到设备。

## 开发用 ADB 配对入口

日常使用连接页。自动化实验仍可通过 [launch_capture.sh](scripts/launch_capture.sh)注入一次性配对，保留旧环境变量名 `DRIVER_CAPTURE_WSS_URL`，其地址实际指向 ActuCore：

```sh
export DRIVER_CAPTURE_WSS_URL='wss://actucore-host:15741/ws/teleop-capture'
export PAIRING_ID='one-time-pairing-id'
export CA_CERT_FILE=/path/to/public-chain.pem
./scripts/launch_capture.sh --platform pico
```

也可提供 `CA_CERT_BASE64`。脚本无回显读取一次性配对码，检查 PEM 不超过 32 KiB、规范 Base64 不超过 43,692 字符。配对码不会写到脚本输出，但短时存在于 ADB 命令参数中，因此调试主机和设备必须可信。

恢复现有凭据或选择多设备中的一个：

```sh
./scripts/launch_capture.sh --platform pico --resume
ADB_SERIAL='<headset-serial>' ./scripts/launch_capture.sh --platform pico --resume
```

Meta 改用 `--platform meta`。新的显式配对参数可替换旧凭据；`--resume` 沿用已保存的地址、CA 和凭据。不同设备 OS 对 ADB 启动权限的支持仍需分别验证。

## 本地输入录制与宿主测试

开发用 NativeActivity 字符串 extra `record_input_only=10` 只启动 OpenXR 透视录制，不加载连接配置、不创建 WSS/RTC，也不访问机器人。最多等待 120 秒，先松握把再有效双握把触发 10 秒采样；失焦、退出或达到 2,000 帧容量标记为不完整。原始输入与头显单调时间保存在应用私有 `files/input-<monotonic>.jsonl` 及对应 `.result.json`，可用 ADB `run-as com.phanthymotus.picocapture` 导出。

该文件没有机器人反馈、标定或执行凭据，离线 IK 必须另行绑定机器人姿态基准，不能直接称为真机验收轨迹。需要同时记录服务端信息时，使用卡片手册中的天轶 Shadow 录制接口。

```sh
./tests/launch_capture_test.sh
python3 tests/launcher_manifest_test.py
NLOHMANN_JSON_INCLUDE=/path/to/nlohmann/include ./tests/run_host_tests.sh
```

[启动脚本测试](tests/launch_capture_test.sh)使用假 ADB；[宿主测试](tests/run_host_tests.sh)覆盖帧契约、配对、恢复、消息解析、按钮和可视化。完整协议与模型测试需要 C++20 编译器和 nlohmann/json 头文件；不设置 `NLOHMANN_JSON_INCLUDE` 会跳过相应检查。可另外设置 `MOTUS_IK_REPLAY=/path/to/visualizations.jsonl`，执行跨语言可视化回放。这些检查不需要头显和机器人，不代替 OpenXR 运行时或硬件执行验证。

构建后检查 APK 的应用 ID、权限、元数据、ABI、ELF 依赖、对齐与签名：

```sh
ANDROID_SDK_ROOT=/path/to/android-sdk \
JAVA_HOME=/path/to/jdk-17 \
python3 tests/verify_android_apks.py --platform pico
```

[APK 检查器](tests/verify_android_apks.py)默认检查两个 flavor；`--platform meta` 可只检查 Meta。检查器读取构建产物，不安装设备。

## 来源与许可证

客户端选择性迁移自 Driver PR #151，并在 ActuCore 维护连接、透视、可视化和操作入口。来源提交与采用范围见 [NOTICE.md](../plugins/teleop/NOTICE.md)；OpenXR、libdatachannel、Mbed TLS 与 nlohmann/json 的版本、许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。保留这些文件及第三方分发中的完整许可证。
