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

打开 `http://<设备IP>:15678` 进入 Web Dashboard。

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

![架构](docs/images/architecture.png)

> 可编辑源文件：[`docs/architecture.svg`](docs/architecture.svg) —— 改完记得重新导出 PNG。

整个平台就是一个 **感知 → 决策 → 执行** 的闭环：

`Hardware → Driver·Sensor → Perception → Agent Loop → ActuCore → Driver·Actuator → Hardware`

- **驱动层（L1）** —— 每个设备一个 MCP Server。每个工具都要声明 `type`，Agent Core 按类型区别对待：`sensor`（数据流）、`actuator`（可执行动作）、`processor`（数据处理）、`resource`（静态资源，如 URDF）。sensor 和 actuator 工具通常在**同一个**驱动进程里 —— 图上把它们分列两侧是按数据流方向画的，不代表要部署两份。
- **感知层（L2，端口 15720 / 15721）** —— 把原始流转成语义：ASR、TTS、VLM 描述、视觉理解、人脸识别。
- **ActuCore（L2，端口 15730）** —— 同一层的执行模型侧，随本仓库的 [`actucore/`](actucore/) 一起发布：VLA 策略、导航、抓取、运动控制、全身控制。它是一个卡片宿主，结构与 Perception 完全一致 —— 每个执行模型以 `processor` 卡片接入，所以任何「输入目标、输出运动指令」的模型都能用同一套方式挂上来。**当前不带任何卡片**，具体用哪些模型按机器人选型决定。卡片契约见 [`actucore/README.md`](actucore/README.md)。
- **Agent Loop（L3，端口 15678）** —— FastAPI + `ros2_bridge.py`：事件采集器、L1–L4 分层 Prompt、工具分发、ACP barrier、历史压缩、Steering / 打断、任务存储、子 Agent 管理、Skills、记忆。
- **两条旁路** —— Loop 可以直接调 `sensor` 工具，绕过感知层；也可以直接用 MCP JSON-RPC 驱动 `actuator` 工具，绕过 ActuCore。简单查询和单次指令走的就是这两条路。
- **Web Dashboard** —— 通过 `/ws/bus/{topic}` 订阅总线上的全部 DDS Topic，通过 `/ws/motus` 订阅 Agent 的决策流。

硬件驱动在独立仓库维护：**[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**。

### 多机协同（Multi-Agent Peers）

> **状态：部分实现。** 在两台 Orin 测试机上逐项实测的结果：
>
> | 部分 | 状态 |
> |---|---|
> | mDNS 发现、SAS 配对 | 可用 —— 两台互相配对，角色 `operator` |
> | 状态共享（签名 HTTPS） | 可用 —— 双向都拿到对端话题清单，5s 刷新 |
> | 工具代理**入向**（服务对端） | 可用 —— 签名 `tools/list` 只返回接收方画布上绑定的工具 |
> | 工具代理**出向**（调用对端工具） | **未实现** —— 没有任何地方把 peer 注册成合成 MCP 条目，本机 LLM 看不到也调不到 |
> | 消息级（`lan` ChannelAdapter） | 代码在，**未验证** —— 两台都没有配置 `lan` 渠道 |
> | 任务委派（`peer_delegate`） | 代码与 hop 限制都在，**未做跨机验证** |
> | 云端名册发现 | 桩 |
> | 超过两台 peer | 从未尝试 |
>
> 飞书 bot 互通（`bot_to_bot_enabled` + `trusted_bots`，见[飞书渠道配置](docs/feishu-channel-setup.md)）
> 仍然作为依赖公网的那条路径存在。

![多机协同与安全](docs/images/peer-mesh.png)

> 可编辑源文件：[`docs/peer-mesh.svg`](docs/peer-mesh.svg) —— 改完记得重新导出 PNG。

机器人之间以**对等（peer）**方式协作：每一侧都跑自己的 Agent Core，各自保有自主权。发现、传输、信任是三个
互相独立、可插拔的层，所以一个 peer 可以「mDNS 发现 + mTLS 通信」，也可以「云端名册发现 + 飞书通信」。

**发现层** —— 所有 provider 产出同一种 `PeerAdvert`，主键是 `peer_id`（Ed25519 公钥指纹，**不是** IP，也不是
平台账号）。因此同一个 peer 从多条路径被发现时，仍然是一条记录、多条链路 —— 这正是降级能成立的前提。

| Provider | 依赖 | 用途 |
|---|---|---|
| mDNS / DNS-SD（`_motus._tcp.local`） | 同局域网 | 同场地，主力路径 |
| ~~DDS presence（`/motus/presence`）~~ | —— | **不可用。** DDS 现已锁在本机（见下），任何基于 DDS 的东西都不跨机 |
| 云端名册 | 公网 | 跨地点、跨网段 |
| BLE 广播 | 无 | 仅用于完全离网时的**配对引导**，不承载数据面 |
| 静态清单 | 无 | 兜底，永远保留 |

**传输层 —— 四种协作粒度：**

1. **消息级** —— 新增一个 `lan` `ChannelAdapter`，于是 peer 对话原样复用现有渠道栈：`InboundMessage`/
   `OutboundMessage`、ACL 角色、限频、`expect_reply` 防环、collector 按信任级别分批。飞书和局域网因此成为
   语义完全相同的两条链路，天然得到「有网走公网、断网走局域网」。
2. **能力级** —— 接收侧已建成：`/api/peer/tools/list` 与 `/api/peer/tools/call` 先验签，再依次套用对端的
   角色、`tool_filter`、**以及接收方自己的画布闸门**，所以 peer 只能碰到本机由人连过线的东西。发送侧
   —— 把 peer 注册成合成 MCP 条目（`transport: 'peer'`），让它的工具以 `mcp__peer:<id>__<tool>` 出现在
   本机 LLM 面前 —— **尚未实现**：它需要先定一件事，peer 工具是走画布暴露（目前没有 peer 卡片的 UI），
   还是对画布闸门开一个例外。
3. **状态级** —— 话题清单（以及后续的位姿、电量、任务状态）通过同一条签名 HTTPS 链路推送
   （`POST /api/peer/inbox/state`）。这里原本走 DDS topic；DDS 现已限制在本机，而 FastDDS 的传输隔离是
   一份**默认** profile 会套住进程内所有参与者，因此没法只靠配置为 peer 流量单独放开。（按参与者
   分别指定 profile 是可行的 —— 在每个参与者创建前设好 `FASTRTPS_DEFAULT_PROFILES_FILE` 即可，
   天轶驱动的 bridge 正是这样给它的两个 domain 分别选 profile 的 —— 但那要求掌握所有创建点，而
   agent-core、perception、actucore 加十几个驱动做不到这一点。而且签名 HTTPS 本身就是更好的答案：
   它有身份校验。）改动顺带补上了一个真实缺陷：原来的 peer DDS 总线
   **没有任何鉴权**，同一个 `ROS_DOMAIN_ID` 上任何进程都能伪造另一台机器人的状态。现在仍然只承载状态，
   绝不承载指令。
4. **任务级** —— `peer_delegate` 把 `SubagentSpec` 发给 peer，由对方在本地 spawn 一个 subagent 并回传
   `SubagentResult`。接收方会按 peer 自身角色对 `tool_filter` 二次裁剪 —— 发起方给的是请求上限，不是授权 ——
   且 `hop_count > 2` 直接拒绝，避免委派链形成风暴。

**信任层** —— 每个 Agent Core 首次启动生成 Ed25519 身份密钥；`ACCESS_TOKEN` 的职责收窄为「人类访问本机
dashboard」，不再是跨机凭据。配对沿用蓝牙的做法：双方 dashboard 显示同一个 6 位短码（由双方公钥与 nonce
派生），两边都由人确认。这样既能抗中间人、又不需要 CA，也是唯一在纯 BLE、完全无网时仍然成立的方案。链路
随后跑在 pinned mTLS 上。peer 复用 `channel/acl.py` 的角色分级，默认 `viewer`（只读传感器）。

**内部总线只留在本机。** 所有机器人一律 `ROS_DOMAIN_ID=42`，并加载同一份 loopback-only 的 FastDDS
profile（`agent-core/deploy/dds-local.xml`，挂到 `/opt/phanthy-motus/dds-local.xml`），白名单只有
`127.0.0.1`。在 `network_mode: host` 下同一台机器上所有容器共享同一个 loopback，因此本机总线照常工作、
数据出不了这台机器。各机器人配置**完全相同** —— 没有需要人去分配的 domain 编号，这正是重点：
`ROS_DOMAIN_ID` 可用范围很窄，而克隆出来的镜像之间根本无法协调。

为什么这不是可选项：`/remote_control/message` —— 一条**指令** —— 当时正在到达办公网上的每一台机器人。
在 Orin5 上敲的一条指令被 Orin6 一起执行了，两边日志里的时间戳完全一致。DDS 没有寻址、没有鉴权，同一个
domain 上每个订阅者都会收到一切。

两条运维后果：

- **所有 DDS 容器都必须加载这份 profile。** 漏掉的那个容器会把**自己**和本机其余部分隔开，症状是
  「机器人突然什么都听不见」。Agent Core 启动时会自检，并暴露 `GET /api/peer/dds_isolation`；判据是进程的
  UDP socket 是否绑在 `127.0.0.1`，而不是文件在不在。
- **文件缺失是静默失败。** 宿主机上没有 `/opt/phanthy-motus/dds-local.xml` 时，Docker 的 bind mount 会
  自动建一个同名**目录**，FastDDS 读不到有效 profile 就回退到所有网卡 —— 隔离失效，日志里什么都没有。
  Agent Core 会在文件缺失时从镜像里补齐；但已经把那个幽灵目录挂进去的容器必须**重建**而不是重启，
  因为挂载类型在创建时就固化了。

**执行器双闸门 —— 不可协商。** 即使 peer 是 `operator`，它也只能「请求」。跨 agent 的执行器调用必须**同时**满足：

1. 该工具在**本机**画布上连线到 `decision_core`（`_get_bound_tool_schemas`）；
2. 本机 LLM 自主决定要调用它 —— peer 的消息进入 collector 时是「输入」，不是「命令」；
3. 若配置了 `require_actuator_confirm`，仍需人工确认。

**peer 永远无法直接驱动电机。**

## Web Dashboard

Dashboard（`http://<设备IP>:15678`）提供：

### Canvas — 可视化编排

将所需的传感器与执行器放入画布，连接到核心 Agent Loop，框架自动完成数据流转与动作执行。像搭积木一样搭建你的具身智能体。

![Canvas](docs/images/home.png)

### 实时监控

传感器数据实时可视化 — 音频波形、电池状态、3D 骨骼/点云等。

![监控面板](docs/images/dashboard.png)

### 智能体定义

在 UI 中直接定义 Agent 的身份、系统提示词和长期记忆。

![智能体定义](docs/images/agent-definition.png)

### 飞书消息渠道

通过飞书自建应用与 Agent 双向收发文本和附件。完整步骤见[飞书 Channel 配置与收发验收](docs/feishu-channel-setup.md)。

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
| ActuCore MCP | 15730 |
| PR Review Agent（可选） | 25000 |

硬件驱动端口请参见 [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)。

多机协同**不新增端口** —— peer 之间走 Agent Core 已有的 15678，路径为 `/api/peer/*`。防火墙上无需额外放行。

## Resource Center（可选）

平台可选连接 [Resource Center](https://motus.phanthy.com) 获取：
- 预构建的驱动/感知镜像浏览和部署
- 技能和扩展管理
- OTA 更新

通过 `RESOURCE_CENTER_URL` 环境变量配置。

## 贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、架构细节和贡献指南。

## 许可证

[Apache License 2.0](LICENSE)
