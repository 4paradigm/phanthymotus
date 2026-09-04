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

Long-running actions must return promptly with a task ID and expose terminal
state asynchronously. Use `x-completion` only when Agent Core can scope that
completion barrier to the owned resource; otherwise use the card's status topic
and an explicit wait action. A blocking `dispatch()` that holds the HTTP request
for the duration of a motion is the wrong shape.

## Jetson only

The daily image uses `Dockerfile.jetson`; locked navigation dependencies are
built separately by `Dockerfile.navigation-base` and consumed through an exact
`@sha256` base reference. `deploy/build_actucore.sh` takes no `--variant` and
currently supports the published JP 5.11 navigation base; `--base` rebuilds the
heavy base only on native ARM64. Build context is the **repo root**, not
`actucore/`, because the image also needs `deploy/ros-base/audio_msgs/` — so
`COPY` paths inside it are `actucore/…`. A PR adding a `COPY` with a path relative
to `actucore/` will fail the build.

Keep the daily image thin: FAST-LIVO2/Nav2 and their system dependencies belong
in the digest-pinned navigation base, while repository-owned ROS packages are
rebuilt in `Dockerfile.jetson`. Flag changes that silently rebuild the full
third-party stack in every PR or add heavyweight dependencies to unrelated
images.

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
