# Phanthy Motus

[中文文档](README_zh.md) | [Official Website](https://motus.phanthy.com)

An event-driven embodied AI platform that connects LLMs to robot hardware via the [MCP](https://modelcontextprotocol.io) data bus, enabling autonomous perception-reasoning-action loops.

## Features

- **Visual Orchestration** — Drag-and-drop web dashboard for connecting devices, sensors, and AI models on a canvas
- **MCP Data Bus** — Unified [Model Context Protocol](https://modelcontextprotocol.io) interface for all hardware devices
- **Event-Driven Agent Loop** — LLM-powered reasoning with multi-turn tool calling, driven by real-time sensor events
- **ROS2 Integration** — Native DDS bridge for seamless ROS2 topic relay and monitoring
- **Pluggable Perception** — Modular ASR/TTS stack with local inference support (Jetson)
- **Web Dashboard** — Real-time device monitoring, agent activity stream, and configuration — all from the browser

## Architecture

```
Hardware Drivers (MCP)         Agent Core (15678)          Web Dashboard
┌─────────────────┐           ┌──────────────────────┐    ┌─────────────┐
│  camera          │──MCP/HTTP─▶│  Event Collector     │    │  Canvas     │
│  microphone      │──MCP/HTTP─▶│        │             │    │  Sidebar    │
│  locomotion      │──MCP/HTTP─▶│        ▼             │─WS─▶│  Monitor    │
│  arm             │──MCP/HTTP─▶│  LLM Agent Loop      │    │  Activity   │
└─────────────────┘           │        │             │    └─────────────┘
                               │  Tool Execution      │
ROS2 DDS                      │        │             │
┌─────────────────┐           │  DDS Bridge          │
│ sensor topics   │──DDS─────▶│        │             │
│ state topics    │           │  WebSocket Relay     │
└─────────────────┘           └──────────────────────┘

Perception Stack (15720)
┌─────────────────┐
│  ASR / TTS      │──MCP/HTTP + WS
└─────────────────┘
```

Hardware drivers are maintained in a separate repository: **[phanthymotus-driver](https://github.com/4paradigm/phanthymotus-driver)**.

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

## Web Dashboard

The dashboard at `http://<device-ip>:15678` provides:

### Canvas — Visual Orchestration

Drag devices and models onto a canvas, draw connections to define data flows, and configure tool parameters visually.

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

Install, activate, and manage skills that extend the agent's capabilities.

![Skills](docs/images/skills.png)

### Service Deployment

Deploy and manage Agent Core and hardware driver containers from the dashboard.

![Deploy](docs/images/deploy.png)

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture details, and guidelines.

## License

[Apache License 2.0](LICENSE)
