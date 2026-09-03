"""
api/peer.py — peer discovery, pairing and messaging endpoints.

## Two sides

**Dashboard → this Agent Core** — authenticated with ACCESS_TOKEN per auth.py,
exactly like every other dashboard-facing endpoint. These tell the operator what
peers are out there, kick off pairing, list paired peers, and mutate their roles.

**Peer Agent Core → this Agent Core** — authenticated with Ed25519 signatures
per peer/transport.py. The pairing handshake is the only request type that comes
from an unpaired peer, so it runs `verify_signed_request(require_paired=False)`
and relies on the human comparing a short code. Everything else requires the
peer be in the `peers` table, and the signature is checked against the key we
recorded at pairing time — that pin is what makes trust durable.

The `/api/peer/inbox/*` paths are exempt from `auth.py` middleware because they
carry their own authentication. See the rule in auth.py and the CRITICAL comment
in start.py's router ordering.
"""

import base64
import time

import fastapi
from fastapi import Request
from pydantic import BaseModel

from peer import identity, pairing, store, transport
from peer.registry import registry
from peer import tools as peer_tools
from peer import delegation
from channel.adapters import lan
import mcp_client
from subagent.protocol import SubagentSpec


# NOTE: no '/api' here — start.py mounts app_api at '/api', so this router's
# prefix is relative to that. Spelling it '/api/peer' produced '/api/api/peer'
# and every endpoint 404'd while the process looked perfectly healthy.
router = fastapi.APIRouter(prefix='/peer', tags=['peer'])


def _local_display_name() -> str:
    """What to call this agent on the other side's pairing screen.

    Falls back to the hostname: an operator confronted with two bare
    fingerprints has no way to tell which robot is which.
    """
    import socket
    import config
    return (config.main.get('peer_settings', {}).get('display_name')
            or socket.gethostname())


def _peer_label(peer_id: str) -> str:
    """Best known name for a peer — whatever discovery advertised, else the id."""
    advert = registry.get(peer_id)
    if advert is not None and advert.display_name:
        return advert.display_name
    return peer_id[:12]


# ── dashboard-facing (ACCESS_TOKEN) ──────────────────────────────────────────

@router.get('/identity')
async def get_identity():
    """This agent's durable identity. The fingerprint is its peer_id."""
    ident = identity.ensure_identity()
    return {
        'peer_id': ident['peer_id'],
        'public_key': ident['public_key'],
        'created_at': ident['created_at'],
    }


@router.get('/discovered')
async def list_discovered(include_paired: bool = True):
    """Peers seen by any discovery provider, freshest first.

    `include_paired=false` narrows to unpaired peers, which is the pairing
    screen's default filter. Already-paired peers are still *discovered* (the
    registry keeps tracking them) so the LAN link for a paired peer is always
    current even after a restart.
    """
    return {'peers': registry.discovered(include_paired=include_paired)}


@router.get('/paired')
async def list_paired():
    """Peers we have paired with, and what they may do."""
    return {'peers': store.list_peers()}


class StartPairingReq(BaseModel):
    peer_id: str


@router.post('/pair/start')
async def start_pairing(req: StartPairingReq):
    """Begin pairing: generate our nonce, request theirs, derive the short code.

    Returns the 6-digit code the operator compares on both screens. The code
    expires in 5 minutes; that timeout is what makes an unattended robot refuse
    to complete a pairing approved before a restart.
    """
    advert = registry.get(req.peer_id)
    if advert is None:
        raise fastapi.HTTPException(404, 'peer not discovered')

    endpoints = registry.endpoints_for(req.peer_id)
    if not endpoints:
        raise fastapi.HTTPException(400, 'peer has no known endpoint')

    local_nonce = pairing.new_nonce()
    local_pubkey = identity.public_key_b64()
    payload = {
        'nonce': local_nonce,
        'public_key': local_pubkey,
        'display_name': _local_display_name(),
    }
    result, err = await transport.post_json(
        endpoints, '/api/peer/inbox/pair_request', payload,
        extra_headers={'x-motus-public-key': local_pubkey},
        timeout=10.0,
    )
    if result is None:
        raise fastapi.HTTPException(503, f'peer unreachable: {err}')

    remote_nonce = result.get('nonce', '')
    remote_pubkey = result.get('public_key', '')
    if not remote_nonce or not remote_pubkey:
        raise fastapi.HTTPException(502, 'peer response missing nonce or public_key')

    try:
        remote_pubkey_raw = base64.b64decode(remote_pubkey)
    except (ValueError, TypeError):
        raise fastapi.HTTPException(502, 'peer public_key is not valid base64')
    if len(remote_pubkey_raw) != 32:
        raise fastapi.HTTPException(502, 'peer public_key is not 32 bytes')
    remote_peer_id = identity.fingerprint(remote_pubkey_raw)
    if remote_peer_id != req.peer_id:
        raise fastapi.HTTPException(
            502, f'peer public_key fingerprint mismatch: advertised {req.peer_id}, '
            f'key hashes to {remote_peer_id}'
        )

    session = pairing.PairingSession(
        peer_id=req.peer_id,
        peer_public_key=remote_pubkey_raw,
        display_name=result.get('display_name', advert.display_name),
        endpoints=endpoints,
        local_nonce=local_nonce,
        remote_nonce=remote_nonce,
        local_public_key=identity.public_key_raw(),
    )
    pairing.put(session)
    return session.to_dict()


class ConfirmPairingReq(BaseModel):
    peer_id: str
    code: str


@router.post('/pair/confirm')
async def confirm_pairing(req: ConfirmPairingReq):
    """Complete pairing after the human confirms the short code matches.

    Creates the peer record with the pinned public key and a default role.

    Idempotent for an already-paired peer, but the response says so explicitly.
    Once the session is consumed the submitted code cannot be checked against
    anything, and answering a plain 200 made a wrong code look verified — the
    caller had no way to tell "code matched" from "nothing was compared". No
    access is granted either way (this endpoint needs ACCESS_TOKEN and only
    echoes an existing record), but a confirmation step that reports success
    without confirming anything is worth being loud about.
    """
    session = pairing.get(req.peer_id)
    if session is None:
        # Maybe already confirmed, check the store.
        peer = store.get(req.peer_id)
        if peer is not None:
            return {
                'peer': peer,
                'already_paired': True,
                'code_verified': False,
                'note': ('no active pairing session — returned the existing pairing '
                         'unchanged; the submitted code was not checked. To re-verify, '
                         'unpair and pair again.'),
            }
        raise fastapi.HTTPException(404, 'no active pairing session for this peer_id')

    if session.expired:
        pairing.pop(req.peer_id)
        raise fastapi.HTTPException(410, 'pairing session expired')

    if session.code != req.code:
        raise fastapi.HTTPException(403, 'code does not match')

    import config
    default_role = config.main.get('peer_settings', {}).get('default_role', 'viewer')
    peer = store.upsert(
        peer_id=req.peer_id,
        public_key_b64=base64.b64encode(session.peer_public_key).decode(),
        display_name=session.display_name,
        role=default_role,
        endpoints=session.endpoints,
    )
    pairing.pop(req.peer_id)
    print(f'[peer] paired with {req.peer_id[:12]} ({peer["display_name"]}) as {default_role}')
    return {'peer': peer, 'already_paired': False, 'code_verified': True}


@router.get('/pair/active')
async def active_pairings():
    """In-flight pairing sessions. For the dashboard to show the operator what
    codes are live."""
    return {'sessions': pairing.active()}


class UpdatePeerReq(BaseModel):
    display_name: str | None = None
    role: str | None = None
    tool_filter: str | None = None


@router.post('/paired/{peer_id}')
async def update_peer(peer_id: str, req: UpdatePeerReq):
    """Mutate a paired peer's display name, role or tool filter."""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        raise fastapi.HTTPException(400, 'no fields to update')
    peer = store.update(peer_id, **fields)
    if peer is None:
        raise fastapi.HTTPException(404, 'peer not found')
    return {'peer': peer}


@router.delete('/paired/{peer_id}')
async def unpair(peer_id: str):
    """Remove a peer. The dashboard shows this as "unpair"."""
    if not store.delete(peer_id):
        raise fastapi.HTTPException(404, 'peer not found')
    return {'deleted': True}


@router.get('/providers')
async def provider_status():
    """Which discovery providers are running, and why any failed."""
    return {'providers': registry.provider_status()}


@router.get('/dds_topology')
async def dds_topology():
    """ROS2 topic lists from all peers (DDS state sharing).

    Returns {peer_id: {topics: [...], last_seen: float}}. Empty if DDS sharing
    is disabled (no rclpy or ROS_DOMAIN_ID).
    """
    from peer import dds_state
    if not dds_state.is_available():
        return {'available': False, 'peers': {}}
    return {'available': True, 'peers': dds_state.get_peer_topics()}


# ── peer-facing (Ed25519 signature) ──────────────────────────────────────────

@router.post('/inbox/pair_request')
async def inbox_pair_request(req: Request):
    """The other half of the pairing handshake.

    Unpaired by definition, so `require_paired=False` — the consistency check
    (key hashes to peer_id) is what transport.verify_signed_request does, and
    the human comparing a short code on both screens is what stops a MITM.

    Crucially this must *store* a PairingSession, not just answer with a nonce.
    The whole security property of SAS is that a human sees the same six digits
    on both dashboards; a receiver that discards its own nonce can neither
    derive that code nor offer the operator anything to confirm, which reduces
    pairing to "whoever asked first wins".
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=False
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    remote_nonce = payload.get('nonce', '')
    remote_pubkey = payload.get('public_key', '')
    if not remote_nonce or not remote_pubkey:
        raise fastapi.HTTPException(400, 'missing nonce or public_key')

    try:
        remote_pubkey_raw = base64.b64decode(remote_pubkey)
    except (ValueError, TypeError):
        raise fastapi.HTTPException(400, 'public_key is not valid base64')
    if len(remote_pubkey_raw) != 32 or identity.fingerprint(remote_pubkey_raw) != peer_id:
        raise fastapi.HTTPException(400, 'public_key does not match signing identity')

    local_nonce = pairing.new_nonce()
    local_pubkey = identity.public_key_b64()

    # sas_code() sorts its inputs, so this derives the identical code the
    # initiator computed — neither side needs to know who started.
    session = pairing.PairingSession(
        peer_id=peer_id,
        peer_public_key=remote_pubkey_raw,
        display_name=payload.get('display_name', '') or _peer_label(peer_id),
        endpoints=registry.endpoints_for(peer_id),
        local_nonce=local_nonce,
        remote_nonce=remote_nonce,
        local_public_key=identity.public_key_raw(),
    )
    pairing.put(session)
    print(f'[peer] pairing requested by {peer_id[:12]} — code {session.code}')

    return {
        'nonce': local_nonce,
        'public_key': local_pubkey,
        'display_name': _local_display_name(),
    }


@router.post('/inbox/ping')
async def inbox_ping(req: Request):
    """Health probe from a paired peer."""
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')
    store.touch(peer_id)
    return {'pong': True, 'timestamp': time.time()}


@router.post('/inbox/message')
async def inbox_message(req: Request):
    """Inbound peer message, routed to the lan ChannelAdapter."""
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    accepted, err = await lan.deliver(peer_id, payload)
    return {'accepted': accepted, 'reason': err if not accepted else ''}


# ── tool proxy (peer-facing, Ed25519 signature) ──────────────────────────────

@router.get('/tools/list')
async def list_tools(req: Request):
    """Return the subset of tools this peer may call.

    Filtered by role + tool_filter. Schemas are OpenAI function-calling format,
    so a peer can pass them to its own LLM.
    """
    # Signature verification with no body (GET)
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, b'', require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    all_schemas = mcp_client.all_schemas()
    allowed = peer_tools.filter_schemas(peer_id, all_schemas)
    return {'tools': allowed, 'count': len(allowed)}


class ToolCallReq(BaseModel):
    tool_name: str
    arguments: dict


@router.post('/tools/call')
async def call_tool(req: Request):
    """Execute a tool on behalf of the peer.

    The actuator double gate still applies: if the tool is bound to a canvas and
    the peer's role is operator, the canvas must have a connection from the peer
    for the call to succeed. That enforcement happens in mcp_client.call_tool or
    event/llm.py, not here — this endpoint only pre-checks the peer's static
    permission.
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    tool_name = payload.get('tool_name', '')
    arguments = payload.get('arguments', {})
    if not tool_name:
        raise fastapi.HTTPException(400, 'tool_name required')

    # Pre-flight permission check
    allowed, perm_reason = peer_tools.check_tool_permission(peer_id, tool_name)
    if not allowed:
        raise fastapi.HTTPException(403, f'tool call denied: {perm_reason}')

    # Call the tool
    try:
        result = await mcp_client.call_tool(tool_name, arguments)
        store.touch(peer_id)
        return {'result': result, 'error': None}
    except Exception as e:
        return {'result': None, 'error': str(e)}


# ── task delegation (peer-facing, Ed25519 signature) ─────────────────────────

class DelegateReq(BaseModel):
    goal: str
    priority: int = 2
    model: str | None = None
    tool_filter: list[str] | None = None
    tool_deny: list[str] | None = None
    max_rounds: int = 10
    timeout_s: float = 300.0
    # How many peers this task has already been handed through. The receiver
    # increments it and refuses past delegation.MAX_HOP_COUNT.
    hop_count: int = 0


@router.post('/delegate')
async def delegate_task(req: Request):
    """Spawn a subagent on behalf of the peer and return the result.

    The delegated spec's tool_filter is intersected with the peer's role
    permissions, and hop_count is incremented. Returns SubagentResult JSON.
    """
    body = await req.body()
    peer_id, reason = transport.verify_signed_request(
        req.method, req.url.path, req.headers, body, require_paired=True
    )
    if not peer_id:
        raise fastapi.HTTPException(403, f'signature verification failed: {reason}')

    try:
        payload = await req.json()
    except Exception:
        raise fastapi.HTTPException(400, 'invalid JSON')

    # Build SubagentSpec from request.
    #
    # hop_count MUST be read from the payload. Omitting it left the field at its
    # default of 0 on every request, so validate_delegation()'s limit could never
    # trigger and the circuit breaker against delegation storms was inert —
    # A→B→C→D… would have chained without bound.
    try:
        hop_count = int(payload.get('hop_count', 0) or 0)
    except (TypeError, ValueError):
        raise fastapi.HTTPException(400, 'hop_count must be an integer')
    if hop_count < 0:
        raise fastapi.HTTPException(400, 'hop_count must not be negative')

    spec = SubagentSpec(
        goal=payload.get('goal', ''),
        priority=payload.get('priority', 2),
        model=payload.get('model'),
        tool_filter=payload.get('tool_filter'),
        tool_deny=payload.get('tool_deny'),
        max_rounds=payload.get('max_rounds', 10),
        timeout_s=payload.get('timeout_s', 300.0),
        hop_count=hop_count,
    )
    if not spec.goal:
        raise fastapi.HTTPException(400, 'goal required')

    # Validate delegation
    allowed, val_reason = delegation.validate_delegation(peer_id, spec)
    if not allowed:
        raise fastapi.HTTPException(403, f'delegation denied: {val_reason}')

    # Prepare augmented spec (increment hop, intersect filter)
    augmented_spec = delegation.prepare_delegated_spec(peer_id, spec)

    # The manager is owned by the agent loop (event/llm.py) and only exists once
    # that has initialised, so it is resolved per-request rather than imported at
    # module load. Importing a manager singleton at import time is what broke
    # startup before: subagent.manager exposes only the class.
    import subagent
    subagent_manager = subagent._manager_instance
    if subagent_manager is None:
        raise fastapi.HTTPException(
            503,
            'agent loop is not running — delegation needs the local subagent '
            'manager, which starts with the main event loop'
        )

    # Spawn and wait
    try:
        result = await subagent_manager.spawn_and_wait(augmented_spec, timeout=spec.timeout_s)
        store.touch(peer_id)
        return result.to_dict()
    except Exception as e:
        return {
            'agent_id': '',
            'status': 'failed',
            'output': '',
            'error': str(e),
        }


