# QQ Official SDK 1.2.2 — Agent Core integration

Status: LOCAL DEVELOPMENT. Not deployed to the Bumi production Core.
Existing production Core version 80600c3 remains text-only C2C.

## Channel capability matrix

| SDK capability | Local integration | Controlled by |
| --- | --- | --- |
| C2C text receipt and passive reply | Implemented, default | c2c scope |
| Group @bot and passive reply | Implemented, mocked | group scope |
| Guild @bot and passive reply | Implemented, mocked | guild scope |
| Plain Guild messages | Implemented with isolated Identify Intent override | guild scope + guild plain switch |
| Guild direct-message reply | Implemented via authenticated SDK REST API | guild_dm scope |
| Input media (image/video/voice/files) | QQ CDN HTTPS restricted stream download, max 10 MiB each | inbound media switch |
| Output media (image/video/voice/files) | SDK upload + passive rich-media reply, max 3 files per trigger | outbound media switch |
| Text and media in a single turn | Media first then text; partial delivery recorded | outbound media switch |
| Typing indicator | SDK C2C typing call | typing switch |
| Markdown text | SDK Markdown output | markdown switch; platform authorization |
| Gateway heartbeat and reconnection | Official SDK | Existing transport |
| Gateway Resume storage | Persistent WSSessionStore with owner-only files | session resume switch |
| Inline keyboard callback | Exact-chat and actor-bound random token registry | interactions switch |
| SDK ApprovalSender | One-time approval with explicit server-side caller | interactions + approvals switches |
| Voice transcription | SDK STT audio conversion/engine, QQ built-in ASR fallback | inbound media + voice transcription |
| QR onboarding | Dedicated administrator-only CLI | Local administrator |
| Reply deduplication and DNS recovery | Existing guarded retry / passive message binding | Core built-in |

## Platform / product boundaries

- SDK support does not grant QQ platform entitlements. Enterprise and personal
  developer registrations have different Group/Guild and C2C permissions.
- QQ is passive-message first. We do not expose unbounded proactive push.
- QQ users remain in the server-side reply-only tool policy; group members
  cannot command Bumi's motors, shell, TTS or private data.
- Interaction approval records **one-time intent only**, with exact actor and
  group validation. It does not automatically elevate an ACL role or run a
  physical action. Even the SDK's "always" button is narrowed to one-time.
- Group/Guild and rich media are opt-in. Existing C2C configurations keep
  working without any migration.
- Media URLs must be HTTPS and inside a QQ CDN allowlist; redirects are denied.
  Files are streamed with a hard size cap. Outbound files must be copied into
  the dedicated QQ outbox path by trusted server-side operations.
- Each reply must cite the original QQ message ID and exact chat ID.
  C2C passive replies expire after 60 minutes; groups after five minutes.
- The SDK's default Gateway Intent set excludes plain Guild messages.
  The optional subclass enables that Intent *only for that QQ adapter*, avoiding
  a global module patch. Account permission must be verified in production.
- SDK administration methods (token renewal, onboard, upload part completion,
  low-level DTOs) are used indirectly or available through the admin CLI.
  They are not independently exposed as LLM tools.

## Explicit gaps before claiming all functions are production-ready

- Real account authorization for Group, Guild, Markdown, keyboards and media.
- End-to-end media format and size acceptance on the QQ platform.
- Approval buttons need a privileged internal workflow to consume decisions;
  the adapter never exposes this directly to a low-privilege QQ user.
- Audio conversion and STT require a compatible runtime, backend API and
  outbound connectivity. Built-in QQ ASR fallback needs no STT API.
- QR onboarding is operator-initiated and intentionally not remotely callable.
- Complete Linux image test, PR review, machine owner release and field evidence.

## Validation

From repository root:

```sh
python -m pytest agent-core/tests/test_qq_official.py agent-core/tests/test_qq_official_media.py agent-core/tests/test_qq_official_extra.py agent-core/tests/test_qq_official_interactions.py agent-core/tests/test_qq_official_sessions.py agent-core/tests/test_qq_official_canvas.py agent-core/tests/test_qq_official_sdk_contract.py agent-core/tests/test_qq_api_security.py agent-core/tests/test_qq_channel_integration.py agent-core/tests/test_qq_security_isolation.py agent-core/tests/test_qq_policy_async.py agent-core/tests/test_qq_credential_migration.py agent-core/tests/test_qq_real_sdk_delivery.py agent-core/tests/test_channel_authorization.py agent-core/tests/test_channel_role_authorization.py -q
node --check agent-core/web/js/channels.js
python -m compileall -q agent-core/src/channel/adapters agent-core/scripts/qq_official_onboard.py
```

Local result (2026-11-12): 68 passed, 3 skipped; JavaScript syntax and
Python compile checks passed. Linux SDK integration and physical device
acceptance are not yet complete.

## Release rules

Build from the reviewable fork branch, retain the original Core image and
Compose/database backups. Upgrade only one Core instance on Bumi and keep the
QQ bot AppID bound to a single running Gateway consumer. Never start a second
production Core or silently replace the shared robot's core service.
