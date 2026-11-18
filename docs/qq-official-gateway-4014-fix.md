# QQ Official Gateway 4014 — local bug fix and acceptance record

Date: 2026-11-12. Scope: local Core only; no PR push or Bumi deployment.

## Root cause

The official QQ SDK 1.2.2 supplies a broad DEFAULT_INTENTS mask.
The previous adapter used it even for a C2C-only channel. When the
"plain Guild" option was enabled, it added GUILD_MESSAGES (1 << 9),
which may not be authorized for this AppID. QQ closed the Gateway
connection with code 4014, "disallowed intents", before READY. The
ChannelRequest/ChannelReply Canvas cards consequently failed self-check.

## Fix

- `qq_official_gateway.py`: QQScopedWebSocket overrides the SDK
  Identify payload per adapter, with a mask computed exclusively from
  `enabled_scopes`, `enable_guild_plain_messages` and
  `enable_interactions`.
- Private C2C and Group @ share `GROUP_MESSAGES = 33554432`; C2C-only
  does not request Guild, Direct Messages, Interactions or Guild messages.
- The adapter uses the scoped Gateway implementation regardless of
  whether Guild mode is selected. No process-global SDK patch is used.
- Error reporting preserves a useful 4014 authorization diagnosis
  rather than overwriting it with a generic READY failure.

## Tests actually run

- 63 QQ tests passed in the local Linux Core container, including the
  official installed SDK Identify payload contract and 4014 error test.
- 10 Channel authorization/role tests passed in the same container.
- Python compile and Dashboard JavaScript syntax checks passed.
- Channel configuration now uses the real existing QQ AppID, valid
  credential file and C2C-only scope. It is deliberately disabled
  pending exclusive Gateway handover.
- No test has confirmed a real Tencent READY or mobile QQ reply from
  the V2 Core after this change. A passed mock/contract test is not
  a production connection test.

## External blocker / handover safety

The Bumi QQ Channel `qq_bumi_test` was last confirmed online using
the same AppID. Attempts to reach Bumi over the configured SSH endpoint
([redacted private SSH endpoint]) were closed before the SSH banner, and the HTTPS
endpoint could not complete TLS. We have not stopped Bumi's Channel.
The original local QQ test Channel is stopped. We deliberately did
not enable the V2 QQ session concurrently with the existing Bumi bot.

Once Bumi connection is restored, temporarily disable only
`qq_bumi_test` through its normal authenticated Channel API, enable
`qq_official_v2_acceptance` in the local Core and verify Gateway READY.
Then validate a user-originated phone QQ message passing through
`channel_request -> decision_core -> channel_reply`; finally restore
the original Bumi Channel unless a permanent transfer is approved.

Current local Dashboard: [local Dashboard address omitted]
## Group full-message compatibility follow-up (2026-11-12)

- Reproduced a second SDK 1.2.2 limitation: its message dispatch
  allow-list omits `GROUP_MESSAGE_CREATE`. When QQ is configured to
  deliver all group messages, @bot messages can arrive as that event
  instead of `GROUP_AT_MESSAGE_CREATE`.
- `QQScopedWebSocket` now forwards the full-group event without modifying
  SDK process globals; existing duplicate-message checks remain active.
- The adapter accepts it only when mention metadata unambiguously
  identifies this bot (`is_you=true`, or explicit `bot_group_openid`),
  then normalizes the event through the existing group parser and ACL.
  Regular group chatter cannot trigger an LLM response.
- In-container QQ suite: **66 tests passed, 0 failed**.
- V2 Core restarted: Gateway `connected`, C2C/group scopes enabled;
  Canvas start returned HTTP 200, project `running=true`, both Channel
  input/output active.
- **Real GROUP_MESSAGE_CREATE/GROUP_AT_MESSAGE_CREATE arrival and QQ group
  reply remain unverified until a message is sent after this restart.**
- SDK defaults to production `https://api.sgroup.qq.com`. QQ sandbox
  groups require verifying the configured sandbox group and environment;
  never silently replace the production endpoint without confirmation.


## Updated live acceptance — Bumi powered off

The user confirmed Bumi was powered off. The local QQ Channel was
enabled with AppID [runtime AppID redacted] and minimal C2C intents. Tencent Gateway
READY was observed through the adapter status: `connected`, enabled,
with an empty health error. The local Canvas start API returned HTTP 200;
`channel_request`, `channel_reply`, `remote_message`, and
`decision_core` all started. The project-running API returned `true`.
The old local QQ Channel remains disabled. The prior "handover blocker"
section above is superseded by this result.

**Pending:** genuine user-originated phone QQ private message, LLM
processing, and a corresponding QQ reply. These are not yet verified.
