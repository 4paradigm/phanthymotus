# Review rules — hardware drivers (`phanthymotus-driver`)

The authoritative specification is **`README_dev.md`** (856 lines) at the repo
root. Read the sections relevant to what the PR changes before judging it.
`README.md` has the driver catalogue and the rendering tables;
`unitree/go1/CONTRIBUTING.md` is the step-by-step card-authoring tutorial and the
declared blueprint.

## The check that matters most: `dispatch()` returns a plain dict

`dispatch(action, args)` must return a **plain dict** (or `None`). The MCP
handler wraps it:

```python
ok({"content": [{"type": "text", "text": json.dumps(result)}]})
```

Returning an already-wrapped `[{"type": "text", ...}]` double-encodes the JSON.
The frontend then fails to parse it and silently falls back to defaults — so it
looks like a rendering bug, not a driver bug.

**Check this on every new or modified plugin.** The reason it is worth checking
every time: `README_dev.md` contradicts itself. Line 246 bans the pattern; the
skeleton-renderer example around line 453 does exactly it. Anyone who copies
that example ships the bug, and it is easy to miss in review because the code
looks like the documentation.

## Plugin contract

- `PREFIX` class attribute; `__init__(plugin_config, namespace, executor, ...)`;
  `get_tool()` returning one dict *or* `get_tools()` returning a list; `start()`;
  `stop()`; `dispatch(action, args)`.
- **`start` and `stop` must both be handled inside `dispatch()`** — the framework
  provides no default. Expected shapes: sensor → `{"state": "running"}` /
  `{"state": "idle"}`; actuator → `{"state": "ready"}` / `{"state": "idle"}`.
  An always-on sensor still returns the state dict as a no-op.
- A `multiInstance` sensor must genuinely create/activate its node on `start` and
  destroy/release it on `stop`, not just flip a flag.
- `x-action-params` is required when one tool exposes several actions taking
  **different** parameters — Agent Core splits these into per-action LLM
  functions and drives conditional field display in the dashboard. Not needed
  when all actions share parameters.
- `x-completion` for actions that take more than ~3 s: return a unique
  `action_id`, then POST completion to `${AGENT_CORE_URL}/api/acp/complete` from
  the worker thread. Without it the agent cannot tell when the action finished.

## Logging, for a driver specifically

Drivers log almost entirely through bare `print()`, run a `ThreadingHTTPServer`
plus a ROS executor plus spawned subprocesses, and expose their port on the host
network. That combination makes them the component most likely to corrupt its own
log. Check, in a new or modified driver:

- **No `os.dup(1)` / `dup2(..., 1)` anywhere.** This pattern existed in the
  unitree drivers with the comment "Suppress C++ layer stdout" — the noise was
  actually Python `print()` in the vendored SDK, and the redirect only appeared to
  work because the spawned `RpcProxy` child inherited fd 1 = `/dev/null`. It cost
  every other subprocess its stdout and made concurrent writes non-atomic.
- **`logsafe` installed** in `main.py` *and* re-installed at the top of every
  `multiprocessing` child entry point, with `common` in the build context:
  `build_context_extras: [../../common]` in `driver.yaml` plus
  `COPY common/ /work/common/` in the Dockerfile.
- **`log_message` escapes the request line.** The stock override prints
  `fmt % args` verbatim, which embeds `self.requestline` — remote input on a
  host-network port. Require
  `msg.encode("unicode_escape").decode("ascii")[:200]`.
- **Vendored SDK prints on hot paths are gated.** Per-RPC and per-PCM-chunk
  prints belong behind an env flag (`UNITREE_RPC_DEBUG`), not in the default
  path. Never print a DDS `response.data` — it is a string field that can carry
  non-UTF-8.
- **Dockerfile has** `PYTHONUNBUFFERED=1`, `RCUTILS_COLORIZED_OUTPUT=0`, and the
  `CYCLONEDDS_URI` tracing muzzle if the driver uses CycloneDDS.

## `driver.yaml`

Required: `id`, `name`, `category: driver`, `hardware_provider`,
`hardware_model`, `image_name`, `port`, `mcp_url`, `description`.

Note `hardware_model` is what the image is named after — it does not have to
match the directory (`robotera/q5_bundle` builds `robotera/q5`).

**Port must be in 15700–15799 and actually free.** The `1572x` and `1573x`
decades are carved out for perception (15720/15721) and ActuCore (15730, with
15731 reserved), so a driver must not land there. Verify by reading the other
`driver.yaml` files, **not** the table in `README.md` — that table is already
wrong: it lists four drivers on 15702 and two on 15703. The WebSocket port is
conventionally the MCP port + 1.

## Topic formats select the dashboard renderer

`topic_out[].format` decides which renderer runs, so a wrong value produces a
wrong visualisation rather than an error. Valid values include `audio/*`,
`video/*`, `image/jpeg`, `image/depth-z16`, `image/depth-zlib`, `data/json`,
`text/*`, `sensor/skeleton`, `sensor/lidar*`, `sensor/pointcloud`,
`sensor/mapping`.

Prefer `image/depth-zlib` over `image/depth-z16`: the raw form is ~614 KB/frame
and saturates ARM64 CPU, the compressed form ~10–15 KB.

For `sensor/skeleton`, the driver needs a `model` tool (`type: resource`)
returning the URDF, and the joint `name` values published by the `joints` sensor
must match the URDF `<joint name="...">` **exactly**. A mismatch does not error —
the dashboard silently draws a humanoid stick figure whatever the real morphology
is.

## Comparing against an existing driver

Pick the closest existing driver and check the new one against it. Good models:

- `unitree/go1/` — the declared blueprint; cleanest module split, and the only
  driver with an authoring tutorial
- `robotera/q5_bundle/` — best decomposition, one module per capability
- `x-humanoid/tianyi2.0/` — largest complete bundle (30 plugins), has tests
- `unitree/r1/`, `noetix/bumi/` — compact, conventional; good for a small driver
- `dji/mavic3e/` — native-SDK C bridge; `pnpbotics/adam/` — gRPC vendor SDK;
  `unitree/go2/` — SLAM and spatial work

Not models to copy: `deep_robotics/lynx_m20/` (empty `plugins:`, no
`deploy/service.yml`, ships a committed `.zip`), and `unitree/g1/device.py`,
which is spec-canonical for IDs and ports but is 3,400 lines in one file.

## Do not flag these — the docs and the code disagree, and the code wins

- `README_dev.md` says `deploy/service.yml` must not set `network_mode`, `ipc` or
  `pid` because Agent Core injects them. **Every existing driver sets them.**
  Do not raise this against a new driver that follows the existing files.
- The doc writes paths as `drivers/<provider>/<model>/`. There is no `drivers/`
  prefix in the real layout.
- The doc's directory diagram omits `deploy/` and `resource/`, which real drivers
  do have.
- It references a `phanthy/remote_control` driver that does not exist.
