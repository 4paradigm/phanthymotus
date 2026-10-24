"""Apply container port overrides without changing the perception application.

Darvin starts several containers with host networking and supplies MCP_PORT and
WS_PORT for each one. main.py reads ports from YAML, so pass it a private runtime
copy through its existing CONFIG_PATH interface. With no overrides, launch the
application with its original configuration.
"""

import os
from pathlib import Path
import sys
import tempfile

import yaml


def prepare_environment(main_path, environ, runtime_dir):
    """Return the child environment; never modify the source configuration."""
    child_env = dict(environ)
    overrides = {}
    for name, key in (("MCP_PORT", "mcp_port"), ("WS_PORT", "ws_port")):
        value = environ.get(name)
        if value:
            try:
                port = int(value)
            except (TypeError, ValueError):
                raise ValueError("{} must be an integer port".format(name)) from None
            if not 1 <= port <= 65535:
                raise ValueError("{} must be between 1 and 65535".format(name))
            overrides[key] = port
    if not overrides:
        return child_env

    source = Path(environ.get("CONFIG_PATH", str(Path(main_path).parent / "config.yaml")))
    with source.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("perception configuration must be a YAML mapping")
    config.update(overrides)
    if int(config.get("mcp_port", 15720)) == int(config.get("ws_port", 15721)):
        raise ValueError("MCP and WebSocket ports must be different")

    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, filename = tempfile.mkstemp(prefix="config-", suffix=".yaml", dir=str(runtime_dir))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    except BaseException:
        Path(filename).unlink()
        raise
    child_env["CONFIG_PATH"] = str(Path(filename).resolve())
    return child_env


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_perception.py /work/main.py [arguments ...]")
    main_path = Path(sys.argv[1]).resolve()
    child_env = prepare_environment(main_path, os.environ, Path("/run/perception"))
    if child_env.get("CONFIG_PATH") != os.environ.get("CONFIG_PATH"):
        # Do not log the configuration: it may contain credentials for plugins.
        print("[startup] applied container port overrides via CONFIG_PATH", flush=True)
    os.execvpe(sys.executable, [sys.executable, str(main_path)] + sys.argv[2:], child_env)


if __name__ == "__main__":
    main()
