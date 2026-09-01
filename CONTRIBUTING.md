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

# Build Perception Stack (Jetson only — see the script's --jp-version flag)
./build_perception.sh

# Build ActuCore (Jetson only)
./build_actucore.sh
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
├── actucore/          — Layer 2: ActuCore (execution models, MCP Server)
│   ├── main.py        — MCP server entry point
│   └── plugins/       — Execution model cards, including ControlledSemanticSpatial navigation
├── deploy/            — Build & deployment scripts
└── docker-compose.yml — Full stack orchestration
```

Hardware drivers are in a separate repository: [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver).

## Architecture Details

### Three-Layer Design

See the [architecture diagram](README.md#architecture) for how these layers connect.

| Layer | Component | Description |
|-------|-----------|-------------|
| Layer 1 — Hardware Drivers | MCP HTTP Servers | Physical device interfaces ([phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)). A single driver exposes both the sensor side (video, audio, lidar, joints, battery, status) and the actuator side (motion, hand, head, waist, speaker, LED) |
| Layer 2 — Perception Stack | ASR/TTS/VLM plugins | Raw streams → semantics, with local inference support (Jetson) |
| Layer 2 — ActuCore | Execution models | The mirror of perception on the action side: VLA, navigation, grasp policies, locomotion, whole-body control. Lives in `actucore/`, structurally identical to the perception stack — each model attaches as a `processor` card. Ships the robot-independent `ControlledSemanticSpatial` navigation contract; its bundled FAST-LIVO2/Nav2 runtime adapter and current hardware acceptance cover G1, while other bodies require compatible sensor/actuator adapters and their own acceptance |
| Layer 3 — Agent Core | FastAPI + LLM Loop | Event-driven agent with DDS bridge and web dashboard |

### Communication

- **Data Plane**: ROS2 DDS → `ros2_bridge.py` (daemon thread) → `inspection.py` fan-out → WebSocket `/ws/bus/{topic}`
- **Control Plane**: MCP HTTP JSON-RPC 2.0 (Agent Core → hardware/perception)
- **Activity Stream**: WebSocket `/ws/motus` (real-time agent decision broadcast)
- **Tool Types**: every MCP tool declares a `type` — `sensor`, `actuator`, `processor`, or `resource`. The type drives dispatch behaviour: consecutive `sensor` calls are batched in parallel, while `actuator` and `processor` calls pass through the ACP barrier and wait for pending actions to complete first (`_needs_barrier()` in `agent-core/src/event/llm.py`). Tools with no declared type default to barrier-guarded

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
| `src/config.py` | SQLite ConfigDB, seeds defaults and de-dupes the MCP list on startup |
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

### Review Checklist

- [ ] **Sensitive config fields are declared.** Any perception/actucore plugin
      `configSchema` property holding a credential, token or private endpoint
      declares `"format": "password"` or `"x-sensitive": true`. Canvas config gets
      packaged into shareable Solutions and uploaded to the Resource Center, and
      packaging only blanks declared fields — an unmarked secret is published in
      clear text. Spec: `phanthymotus-driver/README_dev.md` § "Marking sensitive fields".
- [ ] Plugins holding per-instance state follow the concurrency rules in
      [perception/README.md](perception/README.md) (`tools/call` runs per-thread;
      `stop` must be able to cancel a concurrent `start`; `destroy_node()` on stop)
- [ ] Actuator paths keep the confirmation/safety behaviour intact
- [ ] New API routes are registered in `agent-core/src/start.py` and, if they must
      skip auth, listed in `agent-core/src/auth.py`

## Code Style

- Python: Follow PEP 8, use type hints where practical
- JavaScript: No build step required, vanilla JS
- Keep dependencies minimal

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
