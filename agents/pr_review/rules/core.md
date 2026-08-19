# Review rules — Agent Core (`phanthymotus/agent-core`)

There is **no `agent-core/README.md`**. Do not go looking for one. The
authoritative references are:

- `CONTRIBUTING.md` — §Architecture Details, §Key Files, §MCP Protocol,
  §Data Bus Types, §Prompt-Memory System
- `README.md` — §Memory & Long-Running Agent Architecture, §Ports, §System Hooks

Agent Core is the largest component (~14.7k lines under `src/`) and the least
documented, so reading the neighbouring code is usually more informative than
reading prose. Before judging a change to a subsystem, read its siblings in the
same directory to learn the local convention.

## Where things live

| Area | File |
|------|------|
| FastAPI lifespan, router registration, static mount | `src/start.py` |
| The agent loop (LLM + tool calling) | `src/event/llm.py` |
| MCP client, tool schema conversion, ACP barrier | `src/mcp_client.py` |
| Device registration and tool discovery | `src/api/mcp_manage.py` |
| Layered prompt construction (L1–L4) | `src/prompt.py` |
| SQLite config | `src/config.py` |
| ROS2 DDS bridge | `src/ros2_bridge.py` |
| Driver deploy (extract `service.yml`, merge compose) | `src/api/drivers.py` |
| Self-update via restart helper | `src/api/system.py` |

## Things that break in this component specifically

- **The prompt is built for prefix caching.** Only stable content belongs in the
  system message; anything volatile (clock, task list) goes in a trailing user
  message. A change that puts dynamic content into the system message silently
  destroys cache hit rate — flag it.
- **Transcript integrity.** A trailing `assistant` message carrying `tool_calls`
  whose ids are never answered makes the API reject the whole request. Any new
  path that can truncate a turn (cancellation, an exception mid-dispatch, restore
  from persistence) must sanitise before sending.
- **Sync SQLite from async handlers.** `sqlite3` is synchronous. A new
  `async def` endpoint that calls it inline blocks the event loop, which stalls
  the ROS2 bridge and every in-flight tool call. Prefer a plain `def` handler
  (FastAPI runs those in a threadpool) or `asyncio.to_thread`.
- **The DB has no WAL and a 5 s busy timeout.** Concurrent writers surface as
  `database is locked`, and several existing call sites swallow it in a bare
  `except`. A new writer on a hot path deserves a comment on this.
- **`dispatch()` results from drivers are plain dicts.** Code that assumes the
  MCP-wrapped shape will double-decode.
- **ACP barrier scope.** The barrier blocks actuator and processor tools, not
  sensor or resource ones. A change to `_needs_barrier` that widens this adds
  latency to every turn.

## Ports

15678 for Agent Core. Perception is 15720/15721, ActuCore 15730 (15731 reserved),
drivers 15700–15799 minus those two decades. A change that hardcodes a port
somewhere new should use the existing config instead.
