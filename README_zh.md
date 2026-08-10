# Phanthy Motus

[English](README.md) | [官网](https://motus.phanthy.com)

**赋予具身智能真正的灵魂。** PhanthyMotus 是新一代开源具身智能 Agent 框架与平台。基于稳健的 ROS2 内核，无缝连接多模态传感器与机器人执行层，灵活集成 World Model、LLM 和 VLM，将传统硬件转化为能够自主感知、思考并行动的智能助手。

## 快速开始

一行命令安装并运行：

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash
```

或指定版本：

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash -s <tag>
```

安装脚本会自动安装 Docker（如未安装）、拉取最新 Agent Core 镜像并启动服务。

打开 `https://<设备主机名>:15678` 进入 Web Dashboard。Agent Core 的兼容证书为自签名证书，在按下文配置部署证书前，浏览器可能显示证书警告。

在 [Resource Center](https://motus.phanthy.com) 浏览可用版本和镜像。

### 连接硬件

从 **[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)** 部署硬件驱动。驱动启动后会自动注册到 Agent Core，无需手动配置。

### 从源码构建

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何从源码构建和运行。

## 特性

- **可视化编排** — 拖拽式 Web Dashboard，在画布上连接设备、传感器和 AI 模型
- **MCP 数据总线** — 统一的 [Model Context Protocol](https://modelcontextprotocol.io) 硬件接口
- **事件驱动 Agent Loop** — LLM 驱动的推理引擎，支持多轮工具调用，由实时传感器事件触发
- **ROS2 集成** — 原生 DDS Bridge，无缝中继和监控 ROS2 Topic
- **可插拔感知栈** — 模块化 ASR/TTS，支持本地推理（Jetson）
- **Web Dashboard** — 浏览器内实时监控设备、查看 Agent 活动流、管理配置

## 架构

![架构](docs/images/architecture.jpg)

硬件驱动在独立仓库维护：**[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**。

## Web Dashboard

Dashboard（`https://<设备主机名>:15678`）提供：

### Canvas — 可视化编排

将所需的传感器与执行器放入画布，连接到核心 Agent Loop，框架自动完成数据流转与动作执行。像搭积木一样搭建你的具身智能体。

![Canvas](docs/images/home.png)

### 实时监控

传感器数据实时可视化 — 音频波形、电池状态、3D 骨骼/点云等。

![监控面板](docs/images/dashboard.png)

### 智能体定义

在 UI 中直接定义 Agent 的身份、系统提示词和长期记忆。

![智能体定义](docs/images/agent-definition.png)

### 历史日志

浏览历史 Agent 会话，查看完整事件轨迹和工具调用结果。

![历史日志](docs/images/history.png)

### 技能管理

社区驱动的技能广场，汇聚用户提交的技能。浏览并一键安装他人分享的技能，也可以用自然语言教会机器人新的特殊技能，无需编程。

![技能](docs/images/skills.png)

### 服务部署

从 Dashboard 部署和管理 Agent Core 及硬件驱动容器。

![部署](docs/images/deploy.png)

## 端口

| 服务 | 端口 |
|------|------|
| Agent Core | 15678 |
| Perception MCP | 15720 |
| Perception WebSocket | 15721 |

硬件驱动端口请参见 [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)。

## Resource Center（可选）

平台可选连接 [Resource Center](https://motus.phanthy.com) 获取：
- 预构建的驱动/感知镜像浏览和部署
- 技能和扩展管理
- OTA 更新

通过 `RESOURCE_CENTER_URL` 环境变量配置。

## 贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、架构细节和贡献指南。

### Shadow 遥操会话控制台与身份配置

Agent Core 提供 `https://<robot-host>:15678/teleop.html` Shadow 遥操控制台。viewer 只能查看去身份化的可信设备与忙闲状态；operator/owner 可以获取单机器人独占会话、查看 15 秒倒计时与 Driver 诊断、进入 Pause/HOLD、建立真实的双 DataChannel WebRTC peer、查看审计事件并释放会话。在浏览器处于安全上下文且报告支持 `immersive-vr` 时，只有精确 Core 会话为 Active、本标签页持有且 RTC 双通道均已打开，操作者才能通过明确点击进入 Quest WebXR。页面随后从 `local-floor` 采样 viewer 与唯一左右 `tracked-pointer` grip 姿态，并通过 `teleop-pose` 发送 `motus.teleop.rtc-frame.v1`。普通采样限频为 60 Hz；tracking/deadman 由安全变为不安全时可以抢占下一个限频槽，避免延迟 fail-safe frame。

为了可重复地在 Quest 上部署，请固定使用一个精确主机名访问控制台，并使证书的 Subject Alternative Name（SAN）覆盖该主机名，完整证书链也必须被 Quest Browser 接受。将证书放到宿主机已挂载的部署目录，并在 `/opt/phanthy-motus/.env` 中成对配置：

```dotenv
MOTUS_TLS_CERT_FILE=/opt/phanthy-motus/tls/fullchain.pem
MOTUS_TLS_KEY_FILE=/opt/phanthy-motus/tls/privkey.pem
```

两项必须同时配置，且都是 Core 容器内可见路径；私钥必须为 owner-only（`0600` 或 `0400`）。如果只配一项、任一路径不是普通文件、私钥权限过宽，或证书和私钥不是可解析且相互匹配的服务端 pair，Core 会直接停止启动，不会悄悄替换部署材料。证书轮换后需要重启 Core。两项都未配置时，Core 继续兼容 `/work/resource/certs/cert.pem` 与 `key.pem`，且只在两个文件都不存在时生成自签名 pair；Core 自有且没有链接的旧 fallback 私钥会自动收紧为 `0600`，但不会修改软链接或多重硬链接所指向的托管私钥权限。该 fallback 只提供 TLS 传输，**不保证**浏览器信任、WebXR 安全上下文或 Quest 真机可用。

真机会话前，用 Quest Browser 打开上述精确主机名，并逐项验证：没有证书拦截页或警告；控制台报告 `window.isSecureContext=true`；`immersive-vr` 受支持；操作者明确点击后成功获取必需的 `local-floor` XR session。仅看到 `https://` URL 或 Core 上存在证书文件，不能作为验收证据。

WebXR deadman 必须显式触发：两个有效 `xr-standard` squeeze 都按下才可能为 true；首次进入、重连或 tracking 丢失后，必须先观察到一次松开，再主动重握才能重新武装。`emulatedPosition` 一律按未追踪处理。单帧协议上限为 64 KiB，DataChannel `bufferedAmount` 高水位为 16 KiB；消息过大、背压、send 异常、RTC 断开、XR 或 document 不可见、`local-floor` reset、离页、退出登录以及 Pause/HOLD/Release 都会先释放 deadman 并关闭 XR 与 RTC，不会自动恢复、重连、重进 VR 或 Acquire。为防头显节流普通 timer，XR animation frame 会在既有认证 Core heartbeat 的 single-flight 空闲时约每 5 秒触发一次续租；RTC 姿态包仍不能续任何租约。

当前发送的是 WebXR 右手系原始 `local-floor` 数据，单位为米：+X 向右、+Y 向上、-Z 向前，四元数顺序为 `[x,y,z,w]`；**尚未映射到机器人坐标系**。Driver 仍只做 `recording` / `would_apply` Shadow 记录，`hardware output false`；页面展示 pose freshness/latest sequence 与 dispatch 证据。该链路尚未在真实 Quest 和真实机器人上完成验证，也没有机器人执行输出。当前 immersive 画面有意保持为诊断黑场：既没有机器人视频，也没有头显内 HUD；tracking、deadman、sequence 与 dispatch 证据仍只在 2D 镜像页显示，需要通过 Quest 投屏或桌面浏览器观察。

Core 默认的自签 TLS fallback **不能证明** Quest Browser 会信任站点或在真机暴露 WebXR。必须使用 Quest 实际访问 Core 的精确 hostname，确保证书 hostname/SAN 与信任链均被该浏览器接受，并以控制台显示的 origin、实时 `isSecureContext` 与 `immersive-vr` 检测结果作为验收依据。显式证书/密钥部署不属于当前切片。

在部署目录的 `.env` 中配置：

```dotenv
ACCESS_TOKEN=owner-long-random-token
MOTUS_OPERATOR_TOKENS={"lab-operator":"operator-long-random-token"}
MOTUS_VIEWER_TOKENS={"auditor":"viewer-long-random-token"}
MOTUS_DRIVER_TOKENS={"teleop-shadow-lab-a":"unique-driver-token-a-000001","teleop-shadow-lab-b":"unique-driver-token-b-000002"}
MOTUS_TELEOP_TICKET_SECRETS={"teleop-shadow-lab-a":"at-least-32-unique-random-bytes-a","teleop-shadow-lab-b":"at-least-32-unique-random-bytes-b"}
MOTUS_ENFORCE_DRIVER_AUTH=true
```

- `ACCESS_TOKEN` 保持向后兼容并映射为 owner；只有 owner 能访问已有 API、Canvas 和 WebSocket。
- operator 只能使用专用 `/api/teleop/*` 会话接口，viewer 无法创建或修改会话；两者仍不能访问已有 API、Canvas 和 WebSocket。
- `MOTUS_DRIVER_TOKENS` 是严格 JSON 对象，键必须是 Driver 注册和 `x-teleop.driver_id` 使用的精确稳定 MCP id；每个 Bearer 必须是 24–4096 个受限 ASCII 字符（`A-Z`、`a-z`、数字以及 `._~+/=-`）。每个值必须唯一且不能复用人类或旧凭证；注册凭证只接受 `X-Motus-Driver-Token` 或 `Authorization: Bearer`，URL 查询参数永远不承载 Driver 密钥。A 的凭证声明 B 会直接返回 403、零配置写入、零 ping。Core 对 capability、普通 MCP、SSE、遥操控制和 `/offer` 都按精确、非空 Driver id 选择 Bearer，缺失或非法 id 会在网络请求前失败。专属凭证注册只持久化单向、绑定 id 的 credential binding，不存 token；新增或轮换该 id 的映射后，旧 trusted 记录会在启动联网前隔离，必须由新 token 主动重新注册，旧专属 binding 也不能静默降级到 legacy fallback。
- `MOTUS_TELEOP_TICKET_SECRETS` 同样按精确 Driver id 配置唯一 HMAC secret，每个至少 32 个 UTF-8 字节，只用于该 Driver 的 20 秒、一次性 WebRTC offer ticket；Core 也会拒绝 ticket secret 与任一专属 Bearer、人类或旧凭证交叉复用。精确 id 与 legacy fallback 都无法选出 ticket secret 时，目录显示 `teleop_signaling_unavailable`，Acquire 会在 Driver 调用前拒绝。Core 可直接从 `/opt/phanthy-motus/.env` 读取；映射、密钥、ticket 和 fence 都不写入 Driver 目录、runtime registry、审计或 Quest 响应。
- `MOTUS_DRIVER_TOKEN` 与 `MOTUS_TELEOP_TICKET_SECRET` 仅保留为迁移 fallback：只对各自映射中不存在的 id 生效，已映射 id 永远不会回退。所有 Driver 迁移后应删除这两个旧变量。配置任意 Driver 或 ticket 凭据都会自动启用 API 认证边界；如果没有配置 `ACCESS_TOKEN` owner，受保护的管理 API 会以 `401` fail closed。
- `MOTUS_ENFORCE_DRIVER_AUTH=false` 仍是默认兼容模式：旧 MCP 继续工作，但永远没有遥操信任。多 Driver 隔离部署应在所有服务升级后切换为 `true`；未认证服务届时只保留 discovery 记录，不进入运行时 registry 或 LLM schema。

带认证凭据的部署默认不会向 LLM 暴露同进程代码、本地修改或任意 URL 抓取工具（`Bash`、`PythonExec`、`Write`、`Edit`、`WebFetch`），配置了人类、Driver 或遥操密钥时也禁止开启。只读文件工具会拒绝部署 `.env`、私钥、证书容器和运行时配置数据库，包括指向这些文件的符号链接。`MOTUS_ENABLE_UNSAFE_DESKTOP_CODE_TOOLS=true` 只允许用于完全无凭据的显式开发逃生口；这些同进程工具不是安全沙箱，认证部署绝不能开启。

Bearer 与 ticket 只有在 Core 启动时加载，每台 Driver 也只在自身启动时加载对应值；已经 Acquire 的会话中禁止轮换。维护时应先 Pause/Release，确认 lifecycle `stop` 或 watchdog-safe 证据，再同步更新两端并协调重启 Driver/Core；随后必须重新注册、ping、跑 Shadow 会话、确认 `health.rtc_enabled=true`，并完成一次真实 `/offer` smoke 后才能恢复使用。所有实例通过后才删除 legacy fallback。专属 credential binding 会阻止新专属 Bearer 在重新注册前被发出；如果某个 id 仍使用未绑定的 legacy fallback 且怀疑旧 endpoint 已失陷，则应先保持机器人执行停机，并在处理完 authority guard 后删除或由 owner 隔离旧 trusted target，否则启动仍可能把新 legacy Bearer 发给旧的 owner-pinned URL。回退到共享密钥不能作为普通恢复方案。

当前支持的 `x-teleop` 描述要求 `protocol="motus.teleop.shadow.v1"`、`dispatch_contract="motus.teleop.dispatch.recording.v1"`、`mode="shadow"`、`actuation_enabled=false`、64 位小写十六进制 `capability_digest`，action enum 至少包含 `prepare_shadow/heartbeat/pause/release/soft_stop/status/stop`，且 signaling 必须声明 `motus.teleop.webrtc-offer-answer.v1` + `/offer` + `authenticated-core-proxy-only`。其中 `stop` 是不携带旧 fence 的 lifecycle 安全停机边界。所有带 `x-teleop` 的工具都会从普通 Canvas、LLM schema 和通用 MCP call 中排除。

浏览器只向同源、已认证的 Core 接口发送 `{type: "offer", sdp}`。Core 将一次性 ticket 绑定到精确会话、authority fence、capability digest 和 SDP 哈希，再携带 Driver bearer 代理到已固定的 Driver `/offer`；浏览器仅收到 `{type: "answer", sdp}`。之后 WebRTC DataChannel 由浏览器/Quest 与 Driver 直连，因此即使 Driver HTTP 只绑定 loopback，Driver 生成的 ICE candidate 仍必须能被头显访问。进入 immersive VR 永远由用户点击直接触发，并强制请求 `local-floor`；能力探测不会打开 session。

控制台会显示固定且不含密钥的 final-dispatch 证据，包括状态、generation、已接收/已记录 sequence、停机回执和计数器。在第一次调用 `prepare_shadow` 之前，Core 现在会先把一条不含密钥的 authority guard 写入 SQLite。Core 重启后只把它恢复成拒绝普通写入的机器人安全锁，不会恢复旧 session、fence、浏览器身份或任何控制权；恢复完成前，普通执行器命令和新的 Acquire 都会被拒绝，受保护的 Driver/机器人根目标也不能被删除、改址或改绑。

只有 owner 可以调用 `POST /api/teleop/authority-guards/{robot_id}/reconcile`。Core 会先核对崩溃前固定的可信 Driver 身份、capability digest 和目标指纹；随后只有同一 Driver boot 证明了更高 generation 的 `safe_revoked`、新 boot 以启动停机回执证明 `safe_unarmed`，或一次不携带旧 fence 的 lifecycle `stop` 返回严格校验通过的安全结果时，才会清锁。持久记录必须先删除，进程内普通命令门才会开放；数据库不可读、目标变化、证明畸形、删除结果不确定或停机仍 pending 时都会继续隔离，并允许 owner 稍后重试。

这套恢复机制只支持单个 Core 进程使用其配置的 SQLite 数据库；active/active Core 副本或各自独立的数据库不会共享 authority gate，因此不受支持。从没有持久 guard 的旧版本升级时，必须先安全停止并 Release 所有现有遥操/机器人会话，再替换并重启 Core，因为新表无法追溯旧进程曾持有的控制权。存在任何 guard 或未完成恢复时，禁止回滚到不认识该表的旧版 Core：必须先由 owner 完成恢复核验并确认 Driver 安全，或者让 Core 与机器人执行保持停机；旧版 Core 会静默忽略这张表。当前路径仍是 `hardware output false` 的 Shadow 模式；持久恢复本身不会启用真实机器人执行。

权限由两层独立租约约束：浏览器每 5 秒续一次 Core 的 15 秒操作者租约；只有 Core 会以不超过 250 ms 的间隔续 Driver 默认 1 秒 watchdog。姿态包和 RTC ping 都不能续租。页面关闭或被节流、Core 关闭、Driver 重启、身份不匹配或心跳故障都会终止 authority，网络恢复后不会自动重新获取。Core 内部只用单调时钟判定 deadline，per-Driver epoch 持久化到 SQLite；fence 凭据不会返回浏览器，也不会进入审计库、Activity 或通用 MCP 路径。

## 许可证

[Apache License 2.0](LICENSE)
