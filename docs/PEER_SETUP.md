# Multi-Agent Peer Mesh Setup Guide

## Installation

### Core Dependencies

```bash
cd agent-core
pip install -e .
```

This installs the base peer mesh with:
- Ed25519 identity & signing
- mDNS discovery (via `zeroconf`)
- BLE scanning (via `bleak`)
- HTTPS transport with signature verification
- Tool proxy & task delegation

### Optional: BLE Advertising (Linux only)

BLE **scanning** works out of the box (`bleak` is a core dependency). BLE
**advertising** (letting other robots discover *this* one) needs an extra
package that is installed by hand:

```bash
pip install bluez-peripheral
```

Requires:
- Linux with BlueZ >= 5.43
- Root or `CAP_NET_ADMIN` for BLE operations

**Why not a `[ble-advertise]` extra?** `uv lock` resolves every declared extra
when building the lockfile, so a Linux-only, pre-1.0 package would break the
container build on every platform — including the ARM64 robots that will never
advertise. `peer/ble_advertiser.py` imports it lazily and degrades to
scan-only when it is absent, so a manual install is the honest trade.

On macOS/Windows or without BlueZ, scanning still works; advertising is skipped
with a log line.

---

## Configuration

### Enable Discovery Providers

Edit `~/.config/motus/config.json` or set via API:

```json
{
  "peer_settings": {
    "enabled": true,
    "discovery": {
      "mdns": true,
      "ble": false,
      "static": [
        {
          "url": "https://192.168.1.100:15678",
          "display_name": "Orin6",
          "peer_id": ""
        }
      ]
    }
  }
}
```

**Discovery Providers:**
- `mdns` (bool): auto-discover peers on the local network. The primary path, on
  by default. Link-local multicast — **does not cross routers**.
- `static` (list): hand-entered addresses, for a peer one subnet away or a
  network that filters multicast. `peer_id` is filled in automatically once
  pairing proves the fingerprint.
- `ble` (bool): Bluetooth LE. Off by default because it needs host-side setup
  (below) that no amount of Python can do for you.

All three can be changed from the dashboard's Peers panel and take effect
without restarting the container.

### BLE Prerequisites (host-side)

BLE is the one provider whose failures are mostly not in this codebase. Before
turning the toggle on:

```bash
# 1. The radio ships soft-blocked on Jetson — unblock it (bluetooth only;
#    `rfkill unblock all` would also enable WiFi, which is not what you want on
#    a robot wired to a fixed address).
sudo rfkill unblock bluetooth

# 2. BlueZ has to be running — bleak talks to it over D-Bus, not to the kernel.
sudo systemctl enable --now bluetooth

# 3. The adapter has to be up.
sudo hciconfig hci0 up && hciconfig -a | head -3
```

In a container, `/var/run/dbus/system_bus_socket` must be mounted (agent-core's
compose template already does this) — without it every scan raises a D-Bus
error that mentions neither Bluetooth nor the missing mount.

Set `BLE_ADAPTER` if the machine has more than one radio; otherwise the first
one BlueZ reports is used.

**What BLE does and does not buy you.** It removes the need for *discovery*
infrastructure: two robots in the same room with no shared subnet still find
each other and exchange public keys. It does **not** remove the need for
*connectivity* — the SAS pairing handshake runs over HTTPS, so if the two have
no IP path between them they will appear in each other's list and pairing will
fail. Peers discovered with no reachable endpoint are shown without an address
for exactly this reason.

---

## Usage

### 1. Start the Agent

```bash
cd agent-core
python src/start.py
```

The peer system starts automatically:
```
[peer] generated identity a1b2c3d4e5f6...
[peer] discovery provider "mdns" started
[peer] discovery provider "static" started
```

With `discovery.ble` on, two more lines appear — the second only when the
peripheral half is installed:

```
[peer] ble advertising a1b2c3d4e5f6 on hci0
[peer] discovery provider "ble" started
```

### 2. Discover Peers

**Via Dashboard:**
```
http://localhost:15678/dashboard
→ Peers tab → Discovered section
```

**Via API:**
```bash
curl http://localhost:15678/api/peer/discovered
```

Returns:
```json
{
  "peers": [
    {
      "peer_id": "f7e8d9c0b1a2...",
      "display_name": "Orin6-mDNS",
      "endpoints": ["https://192.168.1.100:15678"],
      "source": "mdns",
      "last_seen": 1735000000.0
    }
  ]
}
```

### 3. Pair with a Peer

**Initiate pairing (on Orin5):**
```bash
curl -X POST http://localhost:15678/api/peer/pair/start \
  -H "Content-Type: application/json" \
  -d '{"peer_id": "f7e8d9c0b1a2..."}'
```

Returns:
```json
{
  "session_id": "abc123",
  "sas_code": "428197",
  "expires_at": 1735000300.0
}
```

**Confirm pairing (on Orin6):**
Display the same 6-digit code `428197` on Orin6's dashboard and confirm.

```bash
curl -X POST http://localhost:15678/api/peer/pair/confirm \
  -H "Content-Type: application/json" \
  -d '{
    "peer_id": "a1b2c3d4e5f6...",
    "sas_code": "428197",
    "display_name": "Orin5",
    "role": "operator"
  }'
```

Both sides are now paired. Check:
```bash
curl http://localhost:15678/api/peer/paired
```

### 4. List Peer Tools

See which tools a peer can call:

```bash
curl http://localhost:15678/api/peer/tools/list \
  -H "X-Motus-Peer-Id: f7e8d9c0b1a2..." \
  -H "X-Motus-Timestamp: $(date +%s)" \
  -H "X-Motus-Signature: <Ed25519_signature>"
```

Returns OpenAI function-calling schemas filtered by peer's `role` and `tool_filter`.

**Role-based filtering:**
- `viewer`: sensors, queries (camera, status, etc.)
- `operator`: all tools including actuators (move, grasp, etc.)
- `tool_filter`: glob pattern, e.g., `camera_*,status`

### 5. Call a Tool on Another Peer

**From Orin5, call a tool on Orin6:**

```bash
curl -X POST https://192.168.1.100:15678/api/peer/tools/call \
  -H "Content-Type: application/json" \
  -H "X-Motus-Peer-Id: a1b2c3d4e5f6..." \
  -H "X-Motus-Timestamp: $(date +%s)" \
  -H "X-Motus-Signature: <Ed25519_signature>" \
  -d '{
    "tool_name": "camera_capture",
    "arguments": {"resolution": "1080p"}
  }'
```

Returns:
```json
{
  "result": "data:image/jpeg;base64,...",
  "error": null
}
```

The actuator double gate still applies: if `camera_capture` is bound to a canvas and Orin5's role is `operator`, the canvas must have a connection from Orin5 for the call to succeed.

### 6. Delegate a Task

**From Orin5, spawn a subagent on Orin6:**

```bash
curl -X POST https://192.168.1.100:15678/api/peer/delegate \
  -H "Content-Type: application/json" \
  -H "X-Motus-Peer-Id: a1b2c3d4e5f6..." \
  -H "X-Motus-Timestamp: $(date +%s)" \
  -H "X-Motus-Signature: <Ed25519_signature>" \
  -d '{
    "goal": "Take a photo and describe what you see",
    "priority": 2,
    "max_rounds": 5,
    "timeout_s": 60.0
  }'
```

Returns:
```json
{
  "agent_id": "subagent_xyz",
  "status": "completed",
  "output": "I see a red cube on a table.",
  "rounds_used": 3,
  "duration_s": 12.4,
  "error": null
}
```

The delegated task runs on Orin6 with Orin5's `tool_filter` applied. `hop_count` is incremented to prevent infinite chains (max 2 hops).

---

## Security Model

### Identity

Each peer has a persistent Ed25519 key pair stored in `~/.config/motus/config.json`:
```json
{
  "peer_identity": {
    "private_key": "base64...",
    "public_key": "base64..."
  }
}
```

`peer_id` = first 16 bytes of SHA256(public_key), hex-encoded (32 chars).

### Pairing

1. Discovery (mDNS/static/BLE) is **unauthenticated** — anyone can advertise
2. Pairing uses **SAS (Short Authentication String)** — both sides compute the same 6-digit code from `HKDF-SHA256(ecdh_shared_secret, "motus-sas")`
3. User compares codes on both dashboards and confirms
4. After confirmation, peer's public key is pinned in the `peers` table

### Transport

Every HTTP request carries an Ed25519 signature:
```
X-Motus-Peer-Id: <sender_peer_id>
X-Motus-Timestamp: <unix_seconds>
X-Motus-Nonce: <random_16_bytes_base64>
X-Motus-Signature: <Ed25519(timestamp||nonce||method||path||body)_base64>
```

Server verifies:
1. Signature matches sender's pinned public key
2. Timestamp within ±30s
3. Nonce not seen before (10-minute cache)

### Access Control

**Roles:**
- `owner`: full control (not used for peers; reserved for human channels)
- `operator`: all tools including actuators
- `viewer`: sensors and queries only (no `move`, `grasp`, `speak`, `write`, `set_`, `execute`, `control`)
- `blocked`: all requests denied

**Tool Filter:**
Glob pattern narrowing the role's permissions, e.g., `camera_*,status` lets an operator call camera tools and status, but not `move_forward`.

**Actuator Double Gate:**
Even if a peer has `operator` role, actuator tools bound to a canvas require a connection from that peer on the canvas. This is the "trust + authorization" model:
- Trust: peer is in the `peers` table (pairing established identity)
- Authorization: canvas has an edge from the peer (user drew the connection)

---

## API Reference

### Discovery & Pairing

- `GET /api/peer/identity` — Our peer_id and public key
- `GET /api/peer/discovered` — Peers discovered via mDNS/static/BLE
- `GET /api/peer/paired` — Peers we've paired with
- `POST /api/peer/pair/start` — Initiate pairing, returns SAS code
- `POST /api/peer/pair/confirm` — Confirm pairing with SAS code
- `POST /api/peer/{peer_id}` — Update peer (display_name, role, tool_filter)
- `DELETE /api/peer/{peer_id}` — Unpair
- `GET /api/peer/providers` — Discovery provider status

### Tool Proxy (peer-facing, Ed25519 signature required)

- `GET /api/peer/tools/list` — Tools this peer may call
- `POST /api/peer/tools/call` — Execute a tool

### Task Delegation (peer-facing, Ed25519 signature required)

- `POST /api/peer/delegate` — Spawn a subagent, returns SubagentResult

### Messaging (peer-facing, Ed25519 signature required)

- `POST /api/peer/inbox/pair_request` — Inbound pairing request
- `POST /api/peer/inbox/ping` — Health check
- `POST /api/peer/inbox/message` — Inbound peer message (routed to lan ChannelAdapter)

### Optional Features

- `GET /api/peer/dds_topology` — ROS2 topic lists from all peers (requires `rclpy`)

---

## Troubleshooting

### mDNS Discovery Not Working

**Symptoms:** `GET /api/peer/discovered` returns empty, even though peers are on the same network.

**Causes:**
1. Multicast blocked by firewall/router
2. `zeroconf` library not installed
3. Peer not advertising (check `/api/peer/providers` on the other peer)

**Solutions:**
```bash
# Check firewall allows mDNS (224.0.0.251:5353)
sudo ufw allow from 224.0.0.0/4
# or
sudo firewall-cmd --add-service=mdns --permanent

# Check zeroconf is installed
pip list | grep zeroconf

# Restart peer discovery
curl -X POST http://localhost:15678/api/peer/providers/restart
```

### Pairing Fails with "SAS code mismatch"

**Causes:**
1. Network latency caused ECDH race (both sides initiated pairing simultaneously)
2. One side's clock is wrong (timestamp drift > 30s)
3. Man-in-the-middle (MITM) attack (rare, but the whole point of SAS)

**Solutions:**
1. Cancel both sessions and start fresh (only one side calls `/pair/start`)
2. Sync clocks: `sudo ntpdate pool.ntp.org`
3. If codes genuinely differ, **do not confirm** — someone is intercepting

### Tool Call Returns 403 "tool call denied"

**Causes:**
1. Peer's `tool_filter` doesn't match the tool name
2. Peer is `viewer` but tool is an actuator
3. Peer is `blocked`

**Check peer role and filter:**
```bash
curl http://localhost:15678/api/peer/paired | jq '.peers[] | select(.peer_id=="f7e8d9c0...")'
```

**Update filter:**
```bash
curl -X POST http://localhost:15678/api/peer/f7e8d9c0... \
  -H "Content-Type: application/json" \
  -d '{"tool_filter": "camera_*,move_*"}'
```

### BLE Not Working

The provider badge in the Peers panel carries the reason. The common ones:

| Badge text | Cause | Fix |
|---|---|---|
| `BLE discovery unavailable (missing bleak)` | Image predates the dependency | Rebuild the image — `bleak` is declared in `pyproject.toml` |
| `scan only — bluez_peripheral not installed` | Not an error. This robot finds peers but cannot be found | `pip install bluez-peripheral` inside the container, on both machines |
| `scan failed: BleakDBusError: ...` | Radio blocked, `bluetoothd` down, or no D-Bus socket in the container | See "BLE Prerequisites" above |
| `advertising failed: RuntimeError: no BLE adapter present` | BlueZ sees no adapter | `hciconfig -a` on the host; check the USB dongle |

A scan failure clears itself on the next successful round, so fixing the host
side is enough — no settings save or restart needed. Rounds are 45 s apart.

**Range matters more than you would expect.** Advertising packets are one-way
broadcasts and survive a weak link; a GATT connection does not. Measured between
the two Orin test rigs: the advert arrived reliably at −93…−99 dBm while every
connection attempt timed out. If peers appear in the list and pairing never
gets a key, check RSSI (shown on the row) before suspecting the code — under
roughly −90 dBm, move the robots closer.

```bash
# Is the radio actually usable, independent of Agent Core?
rfkill list bluetooth
hciconfig -a | head -3
pgrep -a bluetoothd     # more reliable than systemctl on Jetson: some images
                        # ship a Python systemctl shim that reports 'inactive'
                        # for a bluetoothd that is demonstrably running

# Does the container see the bus?
docker exec phanthy-motus-agent-core-1 ls -l /var/run/dbus/system_bus_socket
```

---

## Development

### Run Tests

```bash
cd agent-core
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is not optional: a globally-installed
`fugue_test` plugin fails to import and aborts collection before any test runs,
which makes the suite look broken when it is not.

The BLE tests need no Bluetooth hardware — GATT reads are stubbed, so what runs
is the provider's own logic.

### Enable Debug Logging

```python
import logging
logging.getLogger('peer').setLevel(logging.DEBUG)
```

### Add a New Discovery Provider

1. Subclass `peer.discovery.base.DiscoveryProvider`
2. Implement `start()`, `stop()`, `discovered()`
3. Register in `peer.registry.PROVIDER_REGISTRY`

Example: see `peer/discovery/mdns.py` or `peer/discovery/static.py`.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         Peer Mesh Layer                         │
├─────────────┬───────────────┬──────────────┬───────────────────┤
│  Identity   │  Discovery    │  Transport   │  Application      │
│  (Ed25519)  │  (mDNS/BLE)   │  (HTTPS+sig) │  (Tools/Delegate) │
├─────────────┼───────────────┼──────────────┼───────────────────┤
│ • key gen   │ • mDNS        │ • sign req   │ • tool proxy      │
│ • peer_id   │ • static list │ • verify sig │ • delegation      │
│ • SAS pair  │ • BLE scan    │ • replay     │ • lan messages    │
│             │ • registry    │   guard      │ • DDS topology    │
└─────────────┴───────────────┴──────────────┴───────────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │ Existing Systems  │
                    ├───────────────────┤
                    │ • mcp_client      │
                    │ • subagent        │
                    │ • channel/acl     │
                    │ • config          │
                    └───────────────────┘
```

**Design Principles:**
1. **Zero-trust by default**: pairing required, every request signed
2. **Graceful degradation**: optional features (BLE, DDS) disable cleanly
3. **Reuse existing abstractions**: lan is a ChannelAdapter, peers use ROLES
4. **Testable offline**: all tests run without network/hardware

---

## Next Steps

1. Deploy to Orin hardware and verify mDNS discovery
2. Test pairing flow between two Orins
3. Verify tool proxy with real MCP tools
4. Test task delegation with perception → manipulation pipeline
5. (Optional) Enable DDS topology if ROS2 is in use
6. (Optional) Install BLE advertise deps for offline pairing

**Questions? Check:**
- Code: `agent-core/src/peer/`
- Tests: `agent-core/tests/test_peer_*.py`
- Architecture: `docs/peer-mesh.svg`
