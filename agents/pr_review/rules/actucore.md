# Review rules — ActuCore (`phanthymotus/actucore`)

Authoritative reference: **`actucore/README.md`** — it holds the card contract.

ActuCore is Perception's mirror on the execution side: Perception turns raw
streams into semantics, ActuCore turns intent into motion commands. Execution
models (VLA policies, navigation, grasp policies, locomotion, whole-body
control) attach as cards.

It currently ships the `ControlledSemanticSpatial` navigation card as a
`processor`. Judge host and card changes differently: a host change affects
every card, while a navigation-card change should remain self-contained.

## The card contract

Cards are duck-typed — no base class, no ABC, no registry decorator. Required
members: `PREFIX`, `__init__(cfg, executor)`, `get_tools()`, `dispatch(name, args)`.

Four failure modes worth checking on any card PR:

- **`PREFIX` containing an underscore.** `ActuCoreBundle.dispatch()` routes with
  `full_name.partition("_")`, so `PREFIX = "grasp_policy"` can never be reached.
  The symptom is a tool that lists fine and always answers "Unknown tool".
- **`inputSchema.properties.action.enum` missing `"info"`.** Agent Core probes
  liveness by calling the tool with `{"action": "info"}`; without it the card
  registers and stays permanently offline.
- **`dispatch()` returning a pre-wrapped `[{"type": "text", ...}]`.** It must
  return a plain dict — the MCP HTTP handler does the JSON-RPC wrapping, so
  pre-wrapping double-encodes and breaks the dashboard's parsing.
- **`x-completion` / `x-hooks` placed at the tool's top level.** They belong
  inside `inputSchema`.

Also check that a new card is actually registered: writing `plugins/<name>.py`
does nothing until an `if` block is added to `ActuCoreBundle.__init__` and the
switch is added to `config.yaml`. Card discovery is explicit, not directory
scanning.

## Tool `type` drives scheduling

`type` is one of `sensor` / `actuator` / `processor` / `resource`, and it changes
how Agent Core dispatches: consecutive `sensor` calls batch in parallel, while
`actuator` and `processor` pass the ACP barrier and wait for pending actions
first (`agent-core/src/event/llm.py::_needs_barrier`). An undeclared `type`
defaults to barrier-guarded, which is the safe side.

Execution models are normally `processor`. A card declaring `sensor` for
something that moves the robot is a real bug — it would skip the barrier.

Long-running actions (navigate to a point, execute a grasp) should declare
`x-completion` so the barrier knows when they finish. A blocking `dispatch()`
that holds the HTTP request for the duration of a motion is the wrong shape.

## Jetson only

There is exactly one Dockerfile — `Dockerfile.jetson`. Execution models need the
GPU, so `deploy/build_actucore.sh` takes no `--variant`, only `--jp-version`
(5.11 / 6.1). Build context is the **repo root**, not `actucore/`, because the
image also needs `deploy/ros-base/audio_msgs/` — so `COPY` paths inside it are
`actucore/…`. A PR adding a `COPY` with a path relative to `actucore/` will fail
the build.

The image is deliberately thin: base CUDA torch + ROS2 + `pyyaml requests`,
nothing else. A card's dependencies belong in their own `RUN` layer. Watch for
PRs that pile a card's heavyweight deps into the shared base layers — that is
how perception's image reached several GB.

`COPY actucore/deploy/ /deploy/` must stay: Agent Core extracts
`/deploy/service.yml` from the image to merge the compose fragment, and dropping
it silently degrades to the legacy `docker run` path. Perception's Jetson
Dockerfile has exactly this bug — don't copy it.

## Ports

MCP HTTP on **15730**; 15731 is reserved but unused (unlike perception, ActuCore
has no WebSocket server — there is no audio-stream equivalent). A PR that adds a
second listener should say why.

Changing the port means changing all of: `actucore/config.yaml`, the `EXPOSE` in
`Dockerfile.jetson`, `_SERVICE_ENDPOINTS['actucore']` in
`agent-core/src/api/drivers.py`, the register payload in
`deploy/build_actucore.sh`, and the README port tables. Flag any partial move.

## Identity strings that must stay unique

`serverInfo.name` is `actucore-bundle`, and the registration name is `ActuCore`
with `category: "actucore"`. Agent Core dedupes MCP entries by url / name /
server_name, so a name collision with perception makes the two evict each other.
The `registryImage` (`actucore`) must also keep matching the `_SERVICE_ENDPOINTS`
key, or the deploy manifest loses its port and mcp_url.
