"""
peer/mcp_bridge.py — a paired peer's tools, as if they were a local MCP device.

The receiving half of tool sharing was built first: `/api/peer/tools/{list,call}`
serve a peer. This is the sending half. Without it the local LLM had no way to
reach a peer's tools at all — the tool list existed but nothing was ever offered
to the model, so "call the other robot's camera" was not expressible.

Shape: each paired peer becomes a synthetic entry in `mcp_client.registry` under
`peer:<12 hex>`, so its tools appear as `mcp__peer:<12 hex>__<tool>` and travel the
same path as any other tool — same schema plumbing, same dispatch, same history.
The id deliberately has no `__` in it: `call_tool` splits full names on `__` with
maxsplit=2, and an id containing one would be torn in half.

Two decisions worth stating, because both could reasonably have gone the other way:

* **The remote name is kept.** A peer's `tools/list` already answers with OpenAI
  schemas named `mcp__<their mcp id>__<tool>`, and those names are what its
  `/tools/call` expects back. Rather than translate in both directions, the remote
  name is carried in `remote_names` and the local alias is what the model sees.

  The alias uses the bare tool name, which **collides**: a real peer advertises
  `tts` from perception *and* `tts` from its device driver, and the second silently
  overwrote the first — one of the two became unreachable with nothing to show why.
  A colliding alias therefore carries a tail of the remote MCP id (`tts_70461`),
  paid only by the ones that clash. Single underscores are safe; only `__` is the
  separator `call_tool` splits on.

* **Peer tools are exempt from the local canvas gate** (`canvas_binding.is_bound`).
  The canvas is this operator's authority over what *this* machine exposes; a peer's
  tools are the other operator's to gate, and they do gate them — by role, by
  `tool_filter`, and by their own canvas. Requiring a local card for a remote tool
  would mean inventing a peer card with nothing behind it, and would let a local
  omission look like a remote refusal.

Refreshed on a timer rather than on pairing events: a peer that reboots, gains a
card or has its role lowered changes what it advertises, and nothing tells us.
"""

import asyncio

import mcp_client
from peer import store, transport
from peer.registry import registry as peer_registry

# How much of the fingerprint goes in the tool name. Long enough that two peers
# never collide in practice, short enough that the model does not pay for 32 hex
# characters on every tool it reads.
ID_LEN = 12

REFRESH_INTERVAL_S = 60.0

_task: asyncio.Task | None = None

# peer_id → tool names currently offered to the local LLM. Exposed through
# /api/peer/paired: a bridge that quietly offers nothing looks identical to one
# that is working, and /api/mcp does not show these entries (it is built from the
# configured device list, not from the registry).
offered: dict[str, list[str]] = {}


def mcp_id_for(peer_id: str) -> str:
    return f'peer:{peer_id[:ID_LEN]}'


def peer_id_of(mcp_id: str) -> str:
    """The peer whose synthetic entry this is, or '' if it is not one.

    Resolved against the peers table rather than by string surgery: the entry key
    carries a prefix of the fingerprint, and the full id is what transport needs.
    """
    if not mcp_id.startswith('peer:'):
        return ''
    prefix = mcp_id[len('peer:'):]
    for p in store.list_peers():
        if p['peer_id'].startswith(prefix):
            return p['peer_id']
    return ''


def is_peer_mcp(mcp_id: str) -> bool:
    return mcp_id.startswith('peer:')


def _aliases(remote_names: list[str]) -> dict[str, str]:
    """remote full name → local tool alias, disambiguated only where needed.

    A peer really does advertise the same short name twice (`tts` from perception
    and `tts` from its driver), and the plain alias made the second overwrite the
    first — the first tool then simply did not exist locally, with no error anywhere.
    Clashing aliases get a tail of the remote MCP id appended.
    """
    shorts: dict[str, list[str]] = {}
    for remote in remote_names:
        shorts.setdefault(remote.split('__', 2)[-1], []).append(remote)

    out = {}
    for short, remotes in shorts.items():
        if len(remotes) == 1:
            out[remotes[0]] = short
            continue
        for remote in remotes:
            mcp_part = remote.split('__', 2)[1] if remote.count('__') >= 2 else ''
            tail = ''.join(ch for ch in mcp_part if ch.isalnum())[-5:] or 'x'
            out[remote] = f'{short}_{tail}'
    return out


async def refresh_one(peer: dict) -> int:
    """Fetch one peer's tools and (re)register them. Returns how many were offered."""
    peer_id = peer['peer_id']
    mcp_id = mcp_id_for(peer_id)
    if peer['role'] == 'blocked':
        mcp_client.registry.pop(mcp_id, None)
        offered.pop(peer_id, None)
        return 0

    endpoints = peer_registry.endpoints_for(peer_id)
    if not endpoints:
        # Keep whatever was registered: "unreachable right now" is not the same as
        # "offers nothing", and dropping the tools would make the model rebuild its
        # plan every time a link blinks.
        entry = mcp_client.registry.get(mcp_id)
        if entry:
            entry['online'] = False
        return 0

    resp, reason = await transport.get_json(endpoints, '/api/peer/tools/list', timeout=8)
    if resp is None:
        entry = mcp_client.registry.get(mcp_id)
        if entry:
            entry['online'] = False
        return 0

    label = peer.get('display_name') or peer_id[:ID_LEN]
    advertised = [s for s in (resp.get('tools') or []) if s.get('name')]
    aliases = _aliases([s['name'] for s in advertised])
    acp_meta = resp.get('acp_meta')
    acp_meta = acp_meta if isinstance(acp_meta, dict) else {}

    schemas, tools, tool_meta, remote_names = {}, [], {}, {}
    for schema in advertised:
        remote = schema['name']
        tool = aliases[remote]
        local = f'mcp__{mcp_id}__{tool}'
        aliased = {**schema, 'name': local}
        desc = aliased.get('description') or ''
        aliased['description'] = f'[{label}] {desc}'.strip()
        schemas[local] = aliased
        tools.append(tool)
        remote_names[local] = remote
        # Type is not carried by tools/list, and guessing it locally is what
        # peer/tools.py already got burned by. Left empty: the remote decides what
        # it allows, and all_schemas() only uses type to trim processor actions.
        #
        # `completion` and `resource` are different in kind: the remote is not
        # guessing them either, it is repeating what its own driver declared, so
        # taking them at face value is safe. Without them every peer tool counted as
        # undeclared, and undeclared means exclusive against everything — one peer
        # tool call blocked all local actuation. A malformed `resource` from the far
        # side falls back to None via parse_resources, i.e. to that same
        # conservative reading, never to "conflicts with nothing".
        remote_meta = acp_meta.get(remote) or {}
        tool_meta[local] = {
            'completion': remote_meta.get('completion'),
            'resource': mcp_client.parse_resources(remote_meta.get('resource')),
        }

    mcp_client.registry[mcp_id] = {
        'name': f'{label} (peer)',
        'url': '',
        'online': True,
        'tools': tools,
        'render_hint': '',
        'schemas': schemas,
        'tool_meta': tool_meta,
        'split_map': {},
        'tool_groups': {},
        'input_schemas': {},
        'transport': 'peer',
        'category': 'peer',
        'peer_id': peer_id,
        'remote_names': remote_names,
    }
    if offered.get(peer_id) != tools:
        print(f'[peer] {label} offers {len(tools)} tool(s): {", ".join(tools) or "none"}')
    offered[peer_id] = tools
    return len(tools)


async def refresh_all() -> int:
    total = 0
    for peer in store.list_peers():
        try:
            total += await refresh_one(peer)
        except Exception as e:
            print(f'[peer] tool refresh for {peer["peer_id"][:12]} failed: '
                  f'{type(e).__name__}: {e}')
    return total


async def call(mcp_id: str, tool_name: str, args: dict) -> str:
    """Invoke a tool on the peer this synthetic entry stands for."""
    peer_id = peer_id_of(mcp_id)
    if not peer_id:
        return f'Error: no paired peer behind "{mcp_id}"'
    entry = mcp_client.registry.get(mcp_id) or {}
    local = f'mcp__{mcp_id}__{tool_name}'
    remote = (entry.get('remote_names') or {}).get(local)
    if not remote:
        return (f'Error: "{tool_name}" is not among the tools {entry.get("name", mcp_id)} '
                f'advertises. Its list may have changed; peer_list shows what it offers.')

    endpoints = peer_registry.endpoints_for(peer_id)
    if not endpoints:
        return f'Error: no known address for that peer right now.'

    resp, reason = await transport.post_json(
        endpoints, '/api/peer/tools/call',
        {'tool_name': remote, 'arguments': args}, timeout=60)
    if resp is None:
        # The refusal text from the far side is the useful part — it says whether
        # the role, the filter or their canvas turned it down.
        return f'Error calling {tool_name} on peer: {reason}'
    if resp.get('error'):
        return f'Peer reported an error: {resp["error"]}'
    return str(resp.get('result', ''))


def start() -> None:
    """Begin refreshing peers' tool lists."""
    global _task
    if _task and not _task.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        print('[peer] tool bridge not started: no running event loop')
        return
    _task = loop.create_task(_loop(), name='peer_tool_bridge')
    print('[peer] peer tool bridge started')


def stop() -> None:
    global _task
    if _task:
        _task.cancel()
        _task = None
    for mcp_id in [k for k in mcp_client.registry if is_peer_mcp(k)]:
        mcp_client.registry.pop(mcp_id, None)
    offered.clear()


async def _loop() -> None:
    while True:
        try:
            await refresh_all()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f'[peer] tool bridge round failed: {type(e).__name__}: {e}')
        await asyncio.sleep(REFRESH_INTERVAL_S)
