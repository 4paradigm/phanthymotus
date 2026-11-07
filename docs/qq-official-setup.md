# Tencent QQ Official private-text Channel

This is an **Agent Core Channel adapter** using the Tencent QQ official Bot
Gateway/SDK, not a standalone Driver, Card, NapCat gateway, or OneBot service.
It is designed to work on any supported Agent Core host, including Bumi, subject
to the host's Core image and normal deployment approvals.

## Architecture

```text
Mobile QQ private chat
  -> Tencent official Gateway -> QQOfficialAdapter
  -> ChannelManager (identity, ACL, deduplication, per-chat context)
  -> channel_request (Canvas sensor Card) -> decision_core (LLM)
  -> channel_reply (Canvas executable tool) -> Tencent API -> QQ
```

The application runs inside the **existing Agent Core container**. Do not
start a second production Core or a separate NapCat container for this adapter.

## Supported behavior and security boundary (V1)

- **Supported:** QQ C2C private-text inbound messages and passive text replies
  associated with the exact triggering message ID.
- Gateway health/reconnect, inbound duplicate suppression, and per-private-chat
  conversation isolation are implemented.
- New QQ identities default to the `viewer` role, regardless of the global
  default-role setting.
- The server additionally limits **all** QQ roles, including `operator` and
  `owner`, to `mcp__channel__channel_reply` and `finish`. Elevating the QQ
  user's role **does not** grant robot motion, TTS, music or shell access.
- **Not supported:** QQ group messages, proactive sending, images/files/audio,
  robot actuation, delegated tools, or a separate QQ hardware Driver.
- The LLM's natural-language response is not proof of its actual tool
  permissions. Verify exposed tools using the Core MCP registry/Canvas.

## Requirements

1. A Tencent **official QQ Bot** with its AppID and AppSecret, and an eligible
   QQ user who can send the bot a private message.
2. An Agent Core release that includes `qqbot-agent-sdk==1.2.2`. The normal
   company Core build resolves dependencies from `agent-core/pyproject.toml`.
3. An available LLM service configured in the running Agent Core.
4. A Canvas project containing `channel_request`, `decision_core`, and
   `channel_reply`, with the project started.

Do not commit, screenshot, or print AppSecrets, API keys, QQ user identifiers,
chat logs, session files, or any unredacted channel configuration.

## Configure the QQ Channel

### Option A: Dashboard (environment variable reference)

1. In an approved Core deployment, make the environment variable `QQ_OFFICIAL_APP_SECRET`
   available **inside the running Agent Core container**, not only on Windows.
   Do not put its value in the Dashboard, PR, source code or logs.
2. In Dashboard -> Channels, choose QQ Official and enter a unique Channel ID
   and Tencent Bot AppID.
3. Select Existing Core environment variable; enter its **name**, not its
   value. Alternatively, select Mounted credential file only if a managed
   secret is already mounted at `/run/qq-official-app-secret`.
4. Enable the Channel and confirm connected status with no health error.
5. In Canvas, set the same Channel ID on the `channel_request` sensor Card and
   the `channel_reply` executable Card. Wire `channel_request` to `decision_core`, and
   add the executor connection from `decision_core` to `channel_reply`.
6. Start Canvas and verify a real QQ private-text round trip.

The QQ Dashboard sends **only credential references**, never the credential
value. If the Core service cannot access the configured environment variable
or file, correct the deployment configuration before enabling the Channel.

### Option B: approved configuration API

The authorized Channel API accepts `protocol` set to official,
`private_only` set to true, the Bot AppID and **exactly one** credential
reference:

| Configuration field | Accepted value |
| --- | --- |
| `app_secret_env` | Environment variable starting with `QQ_OFFICIAL_`, available in the Core container |
| `app_secret_file` | Exactly `/run/qq-official-app-secret`, readable by the Core process |

New QQ Channel POST/PUT requests with inline `app_secret` are rejected.
Previously saved inline credentials remain *readable* for existing runtimes,
so upgrades do not immediately disconnect a working Bot. Existing legacy
Channels can be enabled, disabled or restarted without rewriting config;
to edit them, first choose an environment or file reference in Configure.

When a legacy Channel is migrated, the new config no longer includes the
old inline field. Old data backups still require secure handling, and the
Bot credential should be rotated following a successful migration if the
previous value was stored in plaintext. Do not delete runtime data volumes.

Official QQ uses an outbound Gateway connection and does not require
OneBot, NapCat, public webhook or NAT tunnel.

## Review and installation

The pull-request review bot builds an **Agent Core image**. Unlike a Driver
image, Core **does not** ship a `deploy/service.yml` fragment and should not
be started with `run-pr-image.sh`. Follow the build-result comment and use
the existing **Dashboard deployment/upgrade flow**, which replaces the
existing Core through its restart helper. Obtain normal review and deployment
approvals before updating a shared robot.

**Preserve** runtime data volumes, channel settings, LLM configuration and
Canvas layout across the upgrade. The published image does **not** contain
the developer's local QQ credentials, account login, LLM key or saved Canvas.

Before activating a new Core against the same official Bot AppID, stop the
QQ Channel on the old test Core to avoid competing Gateway consumers. Keep the
previous image tag available for rollback.

A merge, a successful CI image build, and a successful runtime deployment are
three distinct milestones. A real Bumi deployment additionally requires
Machine Owner authorization and acceptance evidence.

## Verification and acceptance

Local unit/integration checks from the repository root:

```bash
python -m pytest agent-core/tests/test_qq_official.py agent-core/tests/test_qq_official_canvas.py agent-core/tests/test_qq_official_sdk_contract.py agent-core/tests/test_qq_api_security.py agent-core/tests/test_qq_channel_integration.py agent-core/tests/test_qq_security_isolation.py agent-core/tests/test_qq_policy_async.py agent-core/tests/test_qq_credential_migration.py agent-core/tests/test_qq_real_sdk_delivery.py -q
python -m compileall -q agent-core/src
node --check agent-core/web/js/channels.js
node --check agent-core/web/js/sidebar.js
git diff --check
```

Tests that require the real official SDK may skip in a Python environment
without that package. Verify their contract in the final Linux Core image;
a passing unit suite is not proof of Gateway connectivity.

Minimal real-message acceptance after the approved image is deployed:

1. **Connected:** Channel and project are active, LLM readiness is confirmed.
2. **Round trip:** Mobile QQ private text reaches Agent Core and the response
   returns to the same sender without accidental duplicates.
3. **Authorization:** New QQ users remain viewer; no QQ user can invoke
   unsupported robot/privileged tools through this V1 Channel.
4. **Isolation:** Independent user conversations do not leak into one another.
5. **Recovery:** Confirm Gateway reconnect after a permitted service restart.
6. **Independence (for Bumi):** With the development computer offline, the
   authorized robot Core continues to reply over QQ.

Record sanitized evidence for reviewers and machine owners. Do not perform
physical robot movement as part of a private-text Channel acceptance test.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Channel status disconnected | Verify official AppID, selected credential source and outbound network; check sanitized Core health logs |
| Connected, but no LLM reply | Confirm Canvas project running, matching input/output channel IDs, decision_core readiness and executor connection to channel_reply |
| Reply fails or duplicates | Check original message ID correlation, SDK errors and whether a second Core instance uses the same Bot AppID |
| New QQ user is viewer | Expected security default; operator/owner still cannot bypass V1 QQ tool restrictions |
| Works in developer image but not official image | Check official image tag/architecture, `qqbot-agent-sdk` installation and fresh runtime configuration |
| Behaves differently after restart | Check persisted configuration volumes, enabled Channel and startup order |

## Rollback

1. Disable the QQ Channel or stop the approved Core project before switching
   images, so there is only one active Gateway consumer for that AppID.
2. Use the supported Dashboard Core upgrade/rollback mechanism and the
   recorded **previous Core image tag**. Do not start two production Core
   containers concurrently.
3. Restore previous runtime configuration from the authorized backup if
   necessary, then verify baseline Agent Core health and existing Channels.
4. Re-enable only the intended QQ instance and repeat a private-text roundtrip.

Rollback details depend on the host's approved deployment mechanism; never
delete persistent data volumes to recover from an adapter issue.
