# Embodied AI Platform

[中文文档](README_zh.md)

An event-driven embodied AI platform that connects LLMs to robot hardware via the MCP data bus, enabling a perception-reasoning-action loop.

## Architecture

```
Hardware Devices (MCP Server)      Agent Core (15678)          Web Dashboard
┌─────────────────┐               ┌──────────────────────┐    ┌─────────────┐
│  camera (MJPEG) │──MCP/HTTP──▶  │                      │    │  canvas     │
│  mic (PCM-16k)  │──MCP/HTTP──▶  │  Collector (throttle)│    │  sidebar    │
│  arm (joint)    │──MCP/HTTP──▶  │        │             │─WS▶│  monitor    │
│  lidar (2D)     │──MCP/HTTP──▶  │        ▼             │    │  activity   │
└─────────────────┘               │  LLM Agent Loop      │    └─────────────┘
                                  │  (event/llm.py)      │
ROS2 DDS                          │        │             │
┌─────────────────┐               │  mcp_client.py       │
│ sensor topics   │──DDS─────────▶│  ros2_bridge.py      │
│ state topics    │               │        │             │
└─────────────────┘               │  /ws/bus/{topic}     │
                                  │  /ws/motus           │
                                  └──────────────────────┘

Perception Stack (15720/15721)
┌─────────────────┐
│  ASR plugin     │──MCP/HTTP + WS
│  TTS plugin     │
└─────────────────┘
```

### Three-Layer Design

| Layer | Component | Directory |
|-------|-----------|-----------|
| Layer 1 — Hardware Drivers | MCP HTTP Servers (Unitree G1, Mac Audio, etc.) | `drivers/` |
| Layer 2 — Perception Stack | ASR/TTS plugins + local LLM inference (Jetson) | `perception/` |
| Layer 3 — Agent Core | FastAPI + LLM Loop + DDS Bridge + Web UI | `agent-core/` |

### Communication

- **Data Plane**: ROS2 DDS → `ros2_bridge.py` (daemon thread) → `inspection.py` fan-out → WebSocket `/ws/bus/{topic}`
- **Control Plane**: MCP HTTP JSON-RPC 2.0 (Agent Core → hardware/perception)
- **Activity Stream**: WebSocket `/ws/motus` (real-time agent decision broadcast)

### Ports

| Service | Port |
|---------|------|
| Agent Core | 15678 |
| Perception MCP | 15720 |
| Perception WebSocket | 15721 |

## Quick Start

### Prerequisites

- Docker (ARM64 or x86_64)
- ROS2 Humble (for local development)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)

### Docker Deployment

```bash
cp .env.example .env  # Fill in your LLM API key and config

# Build the ROS2 base image first
cd deploy && ./build_ros_base.sh

# Build services
./build_core.sh
./build_perception.sh

# Deploy
cd core && cp .env.example .env && ./deploy.sh
```

Access the Web Dashboard at `http://<device-ip>:15678`.

### Local Development

```bash
# Install dependencies
uv sync

# Start Agent Core (requires ROS2 Humble)
source /opt/ros/humble/setup.bash
cd agent-core && ./run.zsh
# Visit http://localhost:15678
```

## MCP Protocol

All devices implement [MCP (Model Context Protocol)](https://modelcontextprotocol.io) JSON-RPC 2.0 over HTTP:

| Method | Description |
|--------|-------------|
| `initialize` | Handshake, returns `serverInfo.name` |
| `tools/list` | List tools (with `inputSchema` + `configSchema`) |
| `tools/call` | Invoke tool `{name, arguments}` |

### Data Bus Types

Format: `category/format`

| Category | Examples |
|----------|----------|
| `audio/` | `pcm-16k`, `pcm-48k`, `opus` |
| `video/` | `mjpeg`, `h264`, `depth` |
| `sensor/` | `imu`, `lidar-2d`, `gps`, `force-torque` |
| `control/` | `velocity`, `joint`, `gripper` |
| `state/` | `joint`, `pose`, `power` |
| `text/` | `asr`, `plain` |

## Core Flow

1. **Event Collection**: Collector gathers events from MCP devices, DDS topics, schedulers, and API pushes (with per-source throttling)
2. **Event Bus**: Events queue up; trigger interval fires processing
3. **Prompt Construction**: 4-layer prompt (L1 system rules + L2 env snapshot + L3 conversation history + L4 trigger events)
4. **LLM Reasoning**: Multi-turn tool-calling loop (`mcp__<device_id>__<tool>` naming)
5. **Broadcast**: Each step streamed via `/ws/motus` to the dashboard

## Configuration

All configuration is done through the Web UI, persisted to SQLite (`resource/data.db`), accessed via the `ConfigDB` class.

## Resource Center (Optional)

The platform can optionally connect to a [Resource Center](https://motus.phanthy.com) for:
- Browsing and deploying pre-built driver/perception images
- Managing skills and extensions
- OTA updates

Configure via `RESOURCE_CENTER_URL` environment variable.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[Apache License 2.0](LICENSE)
