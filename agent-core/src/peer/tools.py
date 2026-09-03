"""
peer/tools.py — tool proxy logic: filter and call MCP tools on behalf of peers.

The trust boundary between robots. Access is decided by two things:

  1. `role` — what class of tool this peer may reach at all:
       * `viewer`   — read-only: `sensor`/`resource` from any layer, plus the whole
                      **perception** layer (ASR, TTS synthesis, OCR, VOP compute on
                      data and publish to a topic; they move nothing)
       * `operator` — the above **and** tools that act: `actuator`, the whole
                      **actucore** layer (the execution layer — its `vla` and
                      navigation tools drive the robot despite declaring
                      `processor`), and `controller`
       * `blocked`  — nothing
  2. `tool_filter` — a comma-separated glob list narrowing further within the role,
     e.g. `camera_*,battery`. A bare `*` means "everything the role allows".

**What `operator` means, stated plainly:** an operator peer can drive this robot's
actuators through `/api/peer/tools/call` with no local LLM in the path and no human
confirmation. What still constrains it: the pairing had to be confirmed by a human
on both screens, the role was chosen by a human, the tool must be wired to the
decision core on *this* machine's canvas (api/peer.py, gate 2), `tool_filter` can
narrow it per tool, and every call now announces itself on the activity stream.
Granting `operator` is therefore a real decision, not a formality; `viewer` is the
default for a newly paired peer for that reason.

An undeclared `type` counts as acting — a trust boundary should fail closed, and a
driver that declares nothing is exactly where a guess would be worst.

Two earlier versions of this rule were wrong in ways worth recording, because both
looked reasonable:

* **Guessing the class from the tool name.** It tested for `move`, `grasp`,
  `speak`, `set_`, `execute`, `control` in the name — and a real robot's actuators
  are called `loco`, `led`, `speaker`, `switch_mode`, none of which match. So a
  *viewer* peer could drive locomotion. The declared type from the driver's
  metadata is used instead.
* **Trusting the declared type alone.** `type` does not separate "computes on data"
  from "moves hardware": actucore documents `vla` as a `processor`, and a VLA policy
  drives the robot; navigation's `goto` has the same shape. The layer has to be part
  of the judgement, which is why `category` (sent at registration: `perception`,
  `actucore`, `driver`) is read alongside it.

Hence the rule above: a bare `*` can never grant motion, and granting more is an
explicit act by the local operator, per peer, per tool. Whatever is granted here is
still subject to the receiver's canvas gate — the canvas says what may be reached
at all, this file says what a given peer may ask for.
"""

import fnmatch

import mcp_client
from peer import store


# Types that read and cannot act, in any layer.
READ_ONLY_TYPES = frozenset({'sensor', 'resource'})

# Layers, as sent in the registration payload's `category`.
#   perception — ASR/TTS/OCR/VOP: computes on data, publishes to a topic
#   actucore   — the execution layer: navigation, VLA. Acts, whatever it declares.
READ_ONLY_CATEGORIES = frozenset({'perception'})
ACTING_CATEGORIES = frozenset({'actucore'})


def _mcp_info(full_name: str) -> dict:
    parts = full_name.split('__', 2)
    if len(parts) != 3:
        return {}
    return mcp_client.registry.get(parts[1]) or {}


def tool_type(full_name: str) -> str:
    """The driver's declared type for a tool, or '' when it declared none.

    Read from the MCP registry rather than guessed from the name — see the module
    docstring for what guessing cost.
    """
    info = _mcp_info(full_name)
    meta = (info.get('tool_meta') or {}).get(full_name) or {}
    return str(meta.get('type') or '')


def tool_category(full_name: str) -> str:
    """Which layer offers this tool: 'perception', 'actucore', 'driver', or ''."""
    return str(_mcp_info(full_name).get('category') or '')


def is_read_only(full_name: str) -> bool:
    """Whether this tool only reads — the set a `viewer` peer may reach.

    Layer first, then type: the execution layer acts even when its tools declare
    `processor`, and the perception layer does not act even when it declares one.
    An undeclared type is treated as acting, so a viewer cannot reach it.
    """
    category = tool_category(full_name)
    if category in ACTING_CATEGORIES:
        return False
    if tool_type(full_name) in READ_ONLY_TYPES:
        return True
    return category in READ_ONLY_CATEGORIES


def _role_allows(name: str, role: str) -> tuple[bool, str]:
    """Whether `role` may reach this tool at all."""
    if role == 'blocked':
        return False, 'blocked'
    if is_read_only(name):
        return True, ''
    if role == 'viewer':
        layer = tool_category(name) or 'unknown layer'
        ttype = tool_type(name) or 'undeclared type'
        return False, (f'"{name}" ({layer}/{ttype}) acts, and this peer is a viewer. '
                       f'Raise its role to operator to allow it, or ask over a message '
                       f'or peer_delegate so the local agent decides.')
    return True, ''


def filter_schemas(peer_id: str, all_schemas: list[dict]) -> list[dict]:
    """Return the subset of tools this peer may call.

    Enforces role (viewer ≠ operator) and tool_filter glob. Returns OpenAI
    function-calling schemas, so a peer can present them to its own LLM.
    """
    peer = store.get(peer_id)
    if peer is None or peer['role'] == 'blocked':
        return []

    role = peer['role']
    tool_filter = peer.get('tool_filter', '*')
    patterns = [p.strip() for p in tool_filter.split(',') if p.strip()]
    if not patterns:
        patterns = ['*']

    allowed = []
    for schema in all_schemas:
        name = schema.get('name', '')
        if not name:
            continue

        if not any(fnmatch.fnmatch(name, pat) for pat in patterns):
            continue

        # Advertise only what a call would accept. Listing a tool that always 403s
        # leaves the peer's LLM planning around a capability it does not have —
        # the same reason api/peer.py filters unbound tools out of the list.
        ok, _why = _role_allows(name, role)
        if ok:
            allowed.append(schema)

    return allowed


def check_tool_permission(peer_id: str, tool_name: str) -> tuple[bool, str]:
    """Pre-flight check before calling a tool on behalf of a peer.

    Returns `(allowed, reason)`. Static permission only: the caller must still
    apply the canvas gate (api/peer.py does), which is the local operator's
    separate authority over what a remote machine may reach at all.
    """
    peer = store.get(peer_id)
    if peer is None:
        return False, 'unknown_peer'
    if peer['role'] == 'blocked':
        return False, 'blocked'

    tool_filter = peer.get('tool_filter', '*')
    patterns = [p.strip() for p in tool_filter.split(',') if p.strip()]
    if not patterns:
        patterns = ['*']

    if not any(fnmatch.fnmatch(tool_name, pat) for pat in patterns):
        return False, f'tool "{tool_name}" not in filter: {tool_filter}'

    return _role_allows(tool_name, peer['role'])
