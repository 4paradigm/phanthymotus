# Contributing

We welcome contributions! Here's how to get started.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ROS2 Humble (for Agent Core and DDS features)
- Docker (for building images)

### Local Development

```bash
# Clone the repo
git clone https://github.com/4paradigm/phanthymotus.git
cd phanthymotus

# Install Agent Core dependencies
cd agent-core
uv sync

# Run locally (requires ROS2)
source /opt/ros/humble/setup.bash
./run.zsh
# Visit http://localhost:15678
```

### Building Docker Images

```bash
cd deploy
cp .env.example .env  # Configure registry settings

# Build ROS2 base image (first time only)
./build_ros_base.sh

# Build Agent Core
./build_core.sh

# Build Perception Stack
./build_perception.sh
```

## Project Structure

```
phanthymotus/
├── agent-core/        — Layer 3: Agent Core (FastAPI + LLM Loop + Web UI)
│   ├── src/           — Python source
│   │   ├── api/       — REST & WebSocket endpoints
│   │   ├── event/     — Event-driven agent loop
│   │   ├── client/    — MCP client implementations
│   │   ├── start.py   — FastAPI lifespan & router registration
│   │   ├── config.py  — SQLite ConfigDB
│   │   ├── prompt.py  — 4-layer prompt construction
│   │   └── ros2_bridge.py — ROS2 DDS daemon thread
│   ├── web/           — Vanilla JavaScript UI (no build step)
│   │   ├── js/        — canvas.js, sidebar.js, dashboard.js
│   │   └── css/       — style.css (CSS custom properties)
│   └── resource/      — Memory & config files
├── perception/        — Layer 2: Perception Stack (ASR/TTS MCP Server)
│   ├── main.py        — MCP server entry point
│   └── plugins/       — ASR/TTS plugin implementations
├── deploy/            — Build & deployment scripts
└── docker-compose.yml — Full stack orchestration
```

Hardware drivers are in a separate repository: [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver).

## Architecture Details

### Three-Layer Design

| Layer | Component | Description |
|-------|-----------|-------------|
| Layer 1 — Hardware Drivers | MCP HTTP Servers | Physical device interfaces ([phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)) |
| Layer 2 — Perception Stack | ASR/TTS plugins | Speech processing with local inference support (Jetson) |
| Layer 3 — Agent Core | FastAPI + LLM Loop | Event-driven agent with DDS bridge and web dashboard |

### Communication

- **Data Plane**: ROS2 DDS → `ros2_bridge.py` (daemon thread) → `inspection.py` fan-out → WebSocket `/ws/bus/{topic}`
- **Control Plane**: MCP HTTP JSON-RPC 2.0 (Agent Core → hardware/perception)
- **Activity Stream**: WebSocket `/ws/motus` (real-time agent decision broadcast)

### Core Flow

1. **Event Collection**: Collector gathers events from MCP devices, DDS topics, schedulers, and API pushes (with per-source throttling)
2. **Event Bus**: Events queue up; trigger interval fires processing
3. **Prompt Construction**: 4-layer prompt (L1 system rules + L2 env snapshot + L3 conversation history + L4 trigger events)
4. **LLM Reasoning**: Multi-turn tool-calling loop (`mcp__<device_id>__<tool>` naming)
5. **Broadcast**: Each step streamed via `/ws/motus` to the dashboard

### Key Files

All paths relative to `agent-core/`:

| File | Purpose |
|------|---------|
| `src/start.py` | FastAPI lifespan: starts ros2_bridge, registers routers |
| `src/event/llm.py` | Event-driven agent loop (LLM + tool calling) |
| `src/ros2_bridge.py` | ROS2 DDS daemon thread |
| `src/api/inspection.py` | DDS topic monitoring, WS `/ws/bus/{topic}` |
| `src/api/mcp_manage.py` | MCP device registration + tool discovery |
| `src/api/canvas.py` | Visual canvas state persistence |
| `src/config.py` | SQLite ConfigDB, auto-migrates old ports on startup |
| `src/prompt.py` | Layered prompt construction (L1 system → L4 trigger) |

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

## Configuration

All runtime configuration is managed through the Web UI and persisted to SQLite (`resource/data.db`) via the `ConfigDB` class.

### Prompt / Memory System

- **L1**: `resource/memory/prompt_system.md` (system rules, read-only)
- **L1 Memory**: `resource/memory/prompt_memory.md` (LLM-editable long-term memory)
- **L2**: Environment snapshot (devices, status, recent events) — built dynamically
- **L3**: Conversation history (configurable limit)
- **L4**: Trigger event

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes
3. Ensure code runs locally
4. Submit a PR with a clear description

## Code Style

- Python: Follow PEP 8, use type hints where practical
- JavaScript: No build step required, vanilla JS
- Keep dependencies minimal

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
