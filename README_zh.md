# Embodied AI 具身智能平台

[English](README.md)

事件驱动的具身智能平台，通过 MCP 数据总线将 LLM 与机器人硬件连接，实现感知-思考-行动闭环。

## 架构概览

```
硬件设备 (MCP Server)           Agent Core (15678)          前端 Dashboard
┌─────────────────┐            ┌──────────────────────┐       ┌─────────────┐
│  camera (MJPEG) │──MCP/HTTP─▶│                      │       │  canvas     │
│  mic (PCM-16k)  │──MCP/HTTP─▶│  Collector (throttle)│       │  sidebar    │
│  arm (joint)    │──MCP/HTTP─▶│        │             │──WS──▶│  monitor    │
│  lidar (2D)     │──MCP/HTTP─▶│        ▼             │       │  activity   │
└─────────────────┘            │  LLM Agent Loop      │       └─────────────┘
                               │  (event/llm.py)      │
ROS2 DDS                       │        │             │
┌─────────────────┐            │  mcp_client.py       │
│ sensor topics   │──DDS──────▶│  ros2_bridge.py      │
│ state topics    │            │        │             │
└─────────────────┘            │  /ws/bus/{topic}     │
                               │  /ws/motus           │
                               └──────────────────────┘

Perception Stack (15720/15721)
┌─────────────────┐
│  ASR plugin     │──MCP/HTTP + WS
│  TTS plugin     │
└─────────────────┘
```

### 三层架构

| 层级 | 组件 | 目录 |
|------|------|------|
| Layer 1 — 硬件驱动 | MCP HTTP Server（Unitree G1、Mac Audio 等）| `drivers/` |
| Layer 2 — 感知栈 | ASR/TTS 插件 + LLM 本地推理（Jetson）| `perception/` |
| Layer 3 — Agent Core | FastAPI + LLM Loop + DDS Bridge + Web UI | `agent-core/` |

### 通信机制

- **Data Plane**: ROS2 DDS → `ros2_bridge.py`（daemon thread）→ `inspection.py` fan-out → WebSocket `/ws/bus/{topic}`
- **Control Plane**: MCP HTTP JSON-RPC 2.0（Agent Core → 硬件/感知）
- **Activity Stream**: WebSocket `/ws/motus`（Agent 决策实时广播）

### 端口规范

| 服务 | 端口 |
|------|------|
| Agent Core | 15678 |
| Perception MCP | 15720 |
| Perception WebSocket | 15721 |

## 快速开始

### 环境要求

- Docker（ARM64 或 x86_64）
- ROS2 Humble（本地开发）
- Python 3.12+ 及 [uv](https://docs.astral.sh/uv/)

### Docker 部署

```bash
cp .env.example .env  # 填写 LLM API Key 等配置

# 先构建 ROS2 基础镜像
cd deploy && ./build_ros_base.sh

# 构建服务
./build_core.sh
./build_perception.sh

# 部署
cd core && cp .env.example .env && ./deploy.sh
```

部署后访问 `http://<设备IP>:15678` 进入 Web Dashboard。

### 本地开发

```bash
# 安装依赖
uv sync

# 启动 Agent Core（需 ROS2 Humble）
source /opt/ros/humble/setup.bash
cd agent-core && ./run.zsh
# 访问 http://localhost:15678
```

## MCP 协议

所有设备实现 [MCP（Model Context Protocol）](https://modelcontextprotocol.io) JSON-RPC 2.0 over HTTP：

| 方法 | 说明 |
|------|------|
| `initialize` | 握手，返回 `serverInfo.name` |
| `tools/list` | 列出工具（含 `inputSchema` + `configSchema`）|
| `tools/call` | 调用工具 `{name, arguments}` |

### 数据总线类型

格式：`category/format`

| 类别 | 示例 |
|------|------|
| `audio/` | `pcm-16k`, `pcm-48k`, `opus` |
| `video/` | `mjpeg`, `h264`, `depth` |
| `sensor/` | `imu`, `lidar-2d`, `gps`, `force-torque` |
| `control/` | `velocity`, `joint`, `gripper` |
| `state/` | `joint`, `pose`, `power` |
| `text/` | `asr`, `plain` |

## 核心流程

1. **事件收集**：Collector 从多源（MCP 设备、DDS topic、定时任务、API 推送）收集事件，按源限流
2. **Event Bus**：事件入队，trigger interval 到达后出队
3. **Prompt 构建**：4 层 Prompt（L1 系统规则 + L2 环境快照 + L3 对话历史 + L4 触发事件）
4. **LLM 推理**：多轮工具调用循环（`mcp__<device_id>__<tool>` 命名）
5. **广播**：每步通过 `/ws/motus` 推送到前端

## 配置

所有配置通过 Web UI 完成，持久化到 SQLite（`resource/data.db`），通过 `ConfigDB` 类访问。

## Resource Center（可选）

平台可选连接 [Resource Center](https://motus.phanthy.com) 获取：
- 预构建的驱动/感知镜像浏览和部署
- 技能和扩展管理
- OTA 更新

通过 `RESOURCE_CENTER_URL` 环境变量配置。

## 贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[Apache License 2.0](LICENSE)
