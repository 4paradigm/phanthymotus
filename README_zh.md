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
