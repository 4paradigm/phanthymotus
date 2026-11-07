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

### Option A: Dashboard (quick setup)

1. Open the running Agent Core Dashboard and navigate to **Channels**.
2. Add **QQ Official**, choose a unique channel ID (for example
   `qq_official_main`), and enter the Tencent Bot **AppID** and **AppSecret**
   in the password field. Set the channel to enabled.
3. Verify the Channel status reports connected and no health error.
4. Open the Canvas. Configure the `channel_request` sensor Card and
   `channel_reply` executable Card to use **the same exact channel ID**.
5. Wire `channel_request -> decision_core`, and give `decision_core` an
   **executor connection** to `channel_reply`. Start the Canvas project.
6. Send one private QQ message. Confirm that the reply reaches the same
   private chat.

The Dashboard **quick-add form uses the raw AppSecret field**. It does not
expose an environment-variable/file selector, even though the backend
supports those sources. Do not assume Dashboard entry automatically creates
a persistent environment-variable secret.

### Option B: managed secret source (production)

An operator may configure the Channel through the authorized Channel API or
approved configuration mechanism instead of entering the AppSecret in the UI.
Use `protocol=official`, `private_only=true`, the Bot AppID, and **exactly
one** credential source:

| Credential field | Value / condition |
| --- | --- |
| `app_secret_env` | Environment variable name matching `QQ_OFFICIAL_...`; must be present in the **Core container** |
| `app_secret_file` | Exactly `/run/qq-official-app-secret`; file must be mounted and readable by the Core process |
| `app_secret` | Inline AppSecret (Dashboard quick setup); avoid when a managed secret source is available |

Use only one of these sources. Do not mix old OneBot `access_token` or
`ws_url` settings into this official protocol configuration. The official
Gateway establishes its own outbound connection; this Channel does not
require a public inbound webhook, a NAT tunnel, or NapCat.

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
python -m pytest agent-core/tests/test_qq_official.py agent-core/tests/test_qq_official_canvas.py agent-core/tests/test_qq_official_sdk_contract.py agent-core/tests/test_qq_api_security.py agent-core/tests/test_qq_channel_integration.py agent-core/tests/test_qq_security_isolation.py agent-core/tests/test_qq_policy_async.py -q
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
