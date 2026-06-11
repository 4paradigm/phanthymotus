# Contributing

We welcome contributions! Here's how to get started.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ROS2 Humble (for Agent Core and drivers)
- Docker (for building images)

### Local Development

```bash
# Clone the repo
git clone https://github.com/<org>/embodied.git
cd embodied

# Install Agent Core dependencies
cd agent-core
uv sync

# Run locally (requires ROS2)
source /opt/ros/humble/setup.bash
./run.zsh
```

### Building Docker Images

```bash
cd deploy
cp .env.example .env  # Configure registry settings

# Build ROS2 base image
./build_ros_base.sh

# Build Agent Core
./build_core.sh

# Build Perception Stack
./build_perception.sh
```

## Project Structure

```
agent-core/     — Layer 3: Agent Core (FastAPI + LLM Loop + Web UI)
perception/     — Layer 2: Perception Stack (ASR/TTS MCP Server)
drivers/        — Layer 1: Hardware MCP Drivers
deploy/         — Build & deployment scripts
```

## Writing a New Driver

Drivers are MCP HTTP servers. Implement these JSON-RPC 2.0 methods:

1. `initialize` — Return server info
2. `tools/list` — Declare available tools with `inputSchema`
3. `tools/call` — Handle tool invocations

See `drivers/` for examples. Tool naming convention: `{device}_{action}` (e.g., `mic_start`, `speaker_play`).

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
