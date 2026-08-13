# Phanthy Motus

[中文文档](README_zh.md) | [Official Website](https://motus.phanthy.com)

**Give Embodied AI a Real Soul.** PhanthyMotus is a next-generation, open-source framework and platform for Embodied AI Agents. Built upon a robust ROS2 foundation, it seamlessly bridges diverse sensor inputs with advanced robot execution. By enabling flexible integration of World Models, LLMs, and VLMs, PhanthyMotus transforms traditional hardware into soulful, intelligent assistants capable of perceiving, thinking, and acting independently in the real world.

## Quick Start

Install and run with a single command:

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash
```

Or specify a version:

```bash
curl -fsSL https://motus.phanthy.com/install.sh | sudo bash -s <tag>
```

The install script will automatically install Docker (if needed), pull the latest Agent Core image, and start the service.

Open `http://<device-ip>:15678` to access the Web Dashboard.

Browse available versions and images at the [Resource Center](https://motus.phanthy.com).

### Connect Hardware

Deploy hardware drivers from **[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**. Drivers automatically register with Agent Core on startup — no manual configuration needed.

### Build from Source

See [CONTRIBUTING.md](CONTRIBUTING.md) for building and running from source code.

## Features

- **Visual Orchestration** — Drag-and-drop web dashboard for connecting devices, sensors, and AI models on a canvas
- **MCP Data Bus** — Unified [Model Context Protocol](https://modelcontextprotocol.io) interface for all hardware devices
- **Driver-Inferred Topics** — Output ROS2 topics are declared by drivers, not computed by the core. The canvas calls each driver's `info` action (passing `instance_id` for sensors or `input_topic` for processors) to get the exact topic path before the device starts, keeping all topic naming logic inside the driver
- **Event-Driven Agent Loop** — LLM-powered reasoning with multi-turn tool calling, driven by real-time sensor events
- **ROS2 Integration** — Native DDS bridge for seamless ROS2 topic relay and monitoring
- **Pluggable Perception** — Modular ASR/TTS stack with multi-instance support and local inference (Jetson)
- **Web Dashboard** — Real-time device monitoring, agent activity stream, and configuration — all from the browser

## Architecture

![Architecture](docs/images/architecture.jpg)

Hardware drivers are maintained in a separate repository: **[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**.

### Memory & Long-Running Agent Architecture

The Agent Core is designed for **continuous operation over days or months**. The architecture separates real-time interaction from background intelligence:

```
┌─────────────────────────────────────────────────────┐
│                   Main Agent Loop                     │
│  • Only processes user interactions (ASR/message)    │
│  • Lean history → stable prefix caching (~90% hit)   │
│  • Uses memory_recall for on-demand context retrieval│
└──────────────┬──────────────────────┬───────────────┘
               │ spawn                │ memory_recall
               ▼                      ▼
┌──────────────────────┐   ┌──────────────────────────┐
│   User Task Subagent │   │    Memory Store (SQLite)  │
│  • Isolated context  │   │  • subagent_conclusions   │
│  • Full tool access  │   │  • chat_history (FTS5)    │
│  • Returns summary   │   │  • daily_summary          │
└──────────────────────┘   └──────────────────────────┘
               ▲
┌──────────────────────┐
│    BG Monitor Agent   │
│  • Sensor analysis   │
│  • Results → DB only │
│  • urgent=true → push│
└──────────────────────┘
```

**Key design principles:**

- **Main agent stays lean** — only user interactions enter the conversation history. Background monitoring conclusions are stored in the memory database, not pushed to the main thread.
- **Memory recall on demand** — `memory_recall` tool provides FTS-based retrieval from past conversations, subagent conclusions, and daily summaries. Both main agent and subagents can use it.
- **Urgent interrupts only** — background subagents only interrupt the main agent for safety-critical alerts (battery critical, hardware faults). Routine reports go to the database silently.
- **Daily auto-summary** — a scheduled subagent generates daily reports covering user interactions, task completion, anomalies, performance review, and skill discovery opportunities.
- **Prefix caching optimized** — stable system prompt (L1 + L2-static) is frozen per turn; dynamic status is minimal and placed in user messages to maximize LLM prefix cache hits.

## Web Dashboard

The dashboard at `http://<device-ip>:15678` provides:

### Canvas — Visual Orchestration

Add sensors and actuators you need onto the canvas, connect them to the core Agent Loop, and the framework handles data flow and execution automatically. Build your embodied AI agent like stacking building blocks.

![Canvas](docs/images/home.png)

### Real-Time Monitoring

Live sensor data visualization — audio waveforms, battery status, 3D skeleton/point cloud, and more.

![Monitoring Dashboard](docs/images/dashboard.png)

### Agent Definition

Define the agent's identity, system prompt, and long-term memory directly from the UI.

![Agent Definition](docs/images/agent-definition.png)

### History Logs

Browse past agent sessions with full event traces and tool call results.

![History Logs](docs/images/history.png)

### Skill Management

A community-driven Skill Marketplace where users share and discover skills. Browse and install skills contributed by others, or teach your robot new capabilities using natural language — no coding required.

![Skills](docs/images/skills.png)

### Service Deployment

Deploy and manage Agent Core and hardware driver containers from the dashboard.

For a local Agent Core build, `./deploy/build_core.sh --mirror tuna` passes the
selected Ubuntu, ROS 2, and PyPI mirrors into the Docker build. The Dockerfile
rewrites inherited base-image APT sources before installing additional ROS 2
packages, so a stale Tencent VPC source cannot block a TUNA build. TUNA ROS 2
uses binary package indexes only; inherited `deb-src` entries are removed before
`apt-get update` to avoid requesting an unavailable `source/Sources` index.

![Deploy](docs/images/deploy.png)

## Deployment Architecture

All services run as Docker containers managed by a single `docker-compose.yml` at `/opt/phanthy-motus/` on the target device.

### How it works

1. **Install**: The `install.sh` script pulls the Agent Core image, extracts the initial `docker-compose.yml` from the image, and starts the service
2. **Add drivers**: When you deploy a driver via the Web Dashboard, Agent Core pulls the driver image, extracts its `deploy/service.yml` fragment, and merges it into the compose file
3. **Unified orchestration**: All containers (core, drivers, perception) are managed by the same compose file with `docker compose up -d`

### Container privileges

All driver and perception containers run with `privileged: true` and `/dev:/dev` mounted to access hardware devices (cameras, USB, GPIO). Network is set to `host` mode for ROS2 DDS communication.

```yaml
# Example: how a deployed service looks in /opt/phanthy-motus/docker-compose.yml
services:
  agent-core:
    image: registry/core:tag
    network_mode: host
    ipc: host
    pid: host
    privileged: true
    volumes:
      - /dev:/dev
      - /opt/phanthy-motus/data:/work/resource
    ...
  unitree-g1:
    image: registry/drivers/unitree/g1:tag
    network_mode: host
    ipc: host
    pid: host
    privileged: true
    volumes:
      - /dev:/dev
    ...
```

## Ports

| Service | Port |
|---------|------|
| Agent Core | 15678 |
| Perception MCP | 15720 |
| Perception WebSocket | 15721 |

Hardware driver ports are documented in [phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver).

## Resource Center (Optional)

The platform can optionally connect to a [Resource Center](https://motus.phanthy.com) for:
- Browsing and deploying pre-built driver/perception images
- Managing skills and extensions
- OTA updates

Configure via the `RESOURCE_CENTER_URL` environment variable.

## System Hooks

System hooks provide **instant, bypass-LLM actions** for time-critical responses. Drivers declare hook bindings via `x-hooks` in their MCP tool schema; Agent Core fires them directly on system events without waiting for LLM or ACP barrier.

### Architecture

```
System Event (ASR arrives / LLM starts / error)
  → Agent Core hooks.fire("on_thinking")
  → call_tool_direct() to driver (bypasses barrier + ACP)
  → Driver executes immediately (LED effect, interrupt, etc.)
```

### Available Hooks

| Hook | Trigger | Example |
|------|---------|---------|
| `on_hearing` | Voice activity detected | LED blink blue |
| `on_kws_wakeup` | Wake word detected | LED solid blue 2s |
| `on_thinking` | LLM inference starts | LED rainbow breathe |
| `on_error` | LLM failure | LED red flash 5s |
| `on_interrupt_all` | User barge-in | Stop TTS + motion |

### API

```bash
POST /api/hooks/fire  {"hook": "on_interrupt_all"}
GET  /api/hooks       # list all registered hooks
```

See [phanthymotus-driver/README_dev.md](../phanthymotus-driver/README_dev.md) for driver implementation guide.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture details, and guidelines.

## License

[Apache License 2.0](LICENSE)
