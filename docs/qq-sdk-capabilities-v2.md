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
- QQ users default to server-side reply-only tool policy. An **explicitly provisioned
  owner role on a verified C2C principal** (three-part `official:<appid>:<openid>`)
  can access the full set of tools bound on the Canvas. QQ group/Guild principals
  and ordinary users remain reply-only, even if misconfigured with owner role.
  Runtime owner grants are stored in the ACL database, not checked into Git.
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

## Explicit owner access (opt-in; exact C2C identity only)

- The **user's existing private QQ C2C identity** may be promoted to `owner`
  by a trusted operator **after identity verification**, using the local-only
  `agent-core/scripts/qq_promote_c2c_owner.py --promote` script.
- The script refuses to select an account unless exactly one registered C2C
  principal exists for the active official QQ AppID. No real OpenIDs, account
  secrets, or personal QQ numbers are committed.
- An `owner` C2C principal with `tool_filter="*"` may call **all tools actually
  bound on the Agent Core Canvas**, including Bash, file writes and robot tools.
  This is equivalent to privileged remote access and must be deliberately
  enabled and audited by a trusted system administrator.
- Group, Guild and regular QQ users remain reply-only, including users with
  incorrectly configured elevated roles. Privilege is re-checked at dispatch.
- Runtime ACL roles are persisted in the Core database; a Git commit does not
  transfer the owner's runtime user record between separate Core installations.
  To revoke this privilege, use `--revoke` in the same local admin environment.
- QQ AppID, Gateway entitlement, credentials and robot deployment approvals
  remain separate from the owner role.

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

Local result (2026-11-13): QQ tests 103 passed and Channel tests 10 passed
against the installed SDK in the Linux local test container. Core+Flash image
bridge also passed synthetic image verification; live full media/interaction
acceptance and company CI for the new commit are still pending.

## Release rules

Build from the reviewable fork branch, retain the original Core image and
Compose/database backups. Upgrade only one Core instance on Bumi and keep the
QQ bot AppID bound to a single running Gateway consumer. Never start a second
production Core or silently replace the shared robot's core service.
