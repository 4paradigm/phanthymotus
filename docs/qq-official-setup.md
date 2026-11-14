# Tencent QQ Official Channel — setup, security and Bumi acceptance

This Channel runs inside the **existing PhanthyMotus Agent Core** using Tencent's
official QQ Bot Gateway. It needs **no NapCat, OneBot, Windows relay, new Driver,
or separate Core container**. The official Bot's actual events and media rights
are controlled by Tencent; toggling a scope locally is not evidence of platform
authorization.

## System architecture

```text
Mobile QQ (private / group / guild, where Tencent permits)
  -> Tencent QQ Gateway and official SDK
  -> QQOfficialAdapter (scope parsing, dedup, bounded media download)
  -> ChannelManager (identity, permission isolation, per-chat state)
  -> Canvas channel_request -> decision_core -> channel_reply
  -> Tencent reply API -> originating QQ conversation
```

The same Core also hosts other Channels; QQ setup changes must not alter Feishu
or device Drivers. The QQ AppID is an application identifier, not a login session.

## Runtime image dependencies

- `qqbot-agent-sdk==1.2.2` in `agent-core/pyproject.toml`: official Gateway
  client and C2C/group media APIs.
- `pypdf>=5.0`: bounded, text-only previews of PDF attachments; scanned and
  encrypted PDFs are **not** OCRed. DOCX/XLSX/text preview uses existing parsers.
- `ffmpeg`: converts GIF uploads into at most two JPEG frames for the configured
  multimodal LLM. The existing Core Dockerfile installs `ffmpeg`.
- A model that **actually accepts image input**, and `event.llm.vision_input`
  enabled in the deployed Core. Merely enabling inbound QQ media is insufficient.
- A usable speech-to-text backend is required if QQ voice transcription is
  expected; enabling that event flag alone is insufficient.

**Dependency-lockfile policy:** the company `agent-core/Dockerfile` copies
`pyproject.toml` and executes `uv lock && uv sync --no-dev --no-install-project`
inside the Core image before copying source code. The tracked `agent-core/uv.lock`
is already larger than the repository's changed-file review limit. This PR
deliberately does **not** include the locally regenerated ~856 KB lockfile.
The Docker build regenerates it. This means a separate `uv sync --frozen`
workflow with the unchanged tracked lockfile is **not** guaranteed to be in
sync; treat this as a known repo-wide dependency-policy limitation rather than
proof of a frozen or reproducible offline build. Verify installed packages and
image behavior in the new company **ARM64 Core image**.

## Configure and onboard

1. Build/review the new Core image; an authorized Machine Owner later upgrades
   Bumi through its existing Dashboard Core upgrade procedure. Keep previous
   image tags and persistent resource/config volumes for rollback.
2. Open **Dashboard -> Channels -> Add QQ Official** and follow the onboarding
   page. Enter the Tencent official Bot AppID and select a managed, persistent
   credential source. The guided flow can persist the AppSecret into a private
   credential file with `0600` permissions outside source code; it never
   belongs in Git, Canvas, a screenshot, logs or PR comments. An ephemeral
   credential source must not be treated as reboot-persistent.
3. Bind the Owner from the exact Tencent QQ private-chat identity by completing
   the local verification workflow. Owner replacement/revocation is available
   under advanced administration.
4. Ensure a Canvas project has `channel_request`, `decision_core` and
   `channel_reply` appropriately linked and started. Verify connected status,
   readiness of the chosen LLM, and replies in the originating QQ chat.
5. For testing accounts, configure the bot's allowed users through the
   Tencent QQ developer console. Group users become chat-enabled when Tencent
   emits authorized group events; **no separate membership approval** is needed.
6. To transfer the Bot from a local test Core to Bumi, **stop the local QQ
   Channel first**, then connect it in Bumi. Do not run competing Tencent
   Gateway sessions against the same Bot AppID indefinitely.

The managed credential directory is derived from the Core database location.
Preserve the installation's database/config parent directory across upgrades.
The published image contains no developer AppSecret, live login, QQ conversations,
or machine-local Canvas configuration.

## Authorization model — security-critical

- An ordinary QQ private user has **chat-only** access. QQ group participants,
  including the Owner speaking in a group, also have **chat-only** access.
- A QQ C2C Owner receives full `Agent Core` tool permission *only* when all
  server-side requirements hold: the exact verified private principal, enabled
  owner grant, ACL `role=owner` and `tool_filter=*`. Group principal formats
  cannot inherit this grant. Permission is rechecked at tool dispatch.
- This Owner may be allowed to invoke **Bash, file operations and motion tools**
  if those tools are already registered and available in the running Core.
  It is a deliberately privileged remote-access surface and requires explicit
  maintainer security review. It does **not** install missing robot tools, grant
  missing ROS2 connectivity, or create new Canvas cards.
- Separate QQ users and chats must not share one another's private turn
  context or memory. A short-lived same-sender/same-chat attachment reference
  can support immediate "can you see the picture?" follow-ups.
- User-controlled QQ text is never printed verbatim into the diagnostic LLM
  request summary. Signed QQ CDN URLs and credentials must not enter logs.
- The deployment's existing actuator confirmation and owner authorization
  processes still apply; do **not** use motion to test permissions.

## Feature coverage and explicit limits

| Capability | Status / verification required |
| --- | --- |
| C2C text, passive reply, Gateway reconnect | Local real-message path demonstrated; retest with deployed image |
| QQ group @ replies and auto-join | Local path demonstrated; retest on Bumi and with ordinary participants |
| Images, attachment upload/download | Code + focused tests; revalidate Tencent real CDN payloads |
| GIF | Frame sampling only, not full temporal comprehension |
| PDF, DOCX, XLSX/text | Bounded preview; PDF is text-only and reads at most 10 pages |
| Owner tools | ACL unit tests pass; actual tools depend on Bumi Core registry/Canvas |
| Conversation/memory isolation | Automated regressions; multi-user concurrency remains an acceptance case |
| Guild events, guild DMs, ordinary guild messages, interaction buttons | Gateway subscription alone is not end-to-end proof; verify Tencent entitlement |
| Voice transcription | Requires actual usable STT backend or Tencent ASR text |
| Markdown | Requires Tencent capability authorization; not enabled by default |

## Checks and acceptance

Before submitting this PR:

```bash
python -m pytest -q agent-core/tests/test_qq_log_safety.py agent-core/tests/test_qq_owner_acl.py agent-core/tests/test_qq_group_auto_join.py agent-core/tests/test_qq_memory_isolation.py agent-core/tests/test_qq_vision.py agent-core/tests/test_qq_attachment_preview.py
python -m compileall -q agent-core/src
node --check agent-core/web/js/qq-settings-ui.js
git diff --check
```

After the **company-built image** is available, verify in the shipped image
`import qqbot_agent_sdk, pypdf`, `ffmpeg -version`, the web/JS assets,
and Gateway readiness on the platform it ships for (typically linux/arm64).
Running the image test suite and obtaining a successful build are prerequisites
for Bumi deployment, **not a replacement for physical-device acceptance**.

**Bumi deployment acceptance** (authorized deployment only):

1. Upgrade via the existing Core Dashboard; preserve saved QQ credentials,
   memory DB, other Channel settings and Canvas. Check Core and Dashboard health.
2. Stop the developer PC QQ Channel. On Bumi check Gateway connected and
   reconnect after restart or network outage, without re-entering a secret.
3. With mobile QQ, validate ordinary private text and group @ replies.
4. From the verified Owner's QQ **private** chat, invoke a harmless read-only
   available Core tool; from ordinary users/groups, verify privileged tools are
   rejected. Do not use robot motion as a permission smoke test.
5. Validate real static image, GIF and selectable-text PDF; ask a follow-up in
   a separate message. Do not claim scanned-PDF OCR or full GIF motion analysis.
6. Verify two concurrent users and separate private/group histories; run reboot
   and disconnect/recovery tests. Turn off the Windows PC and confirm QQ replies
   still originate from Bumi independently.
7. Record image tag/commit SHA, sanitized test outcomes and rollback version.
   A failing item remains unaccepted. Advanced guild/STT/Markdown features need
   their own Tencent platform authorization and end-to-end evidence.

## Review, upgrade and rollback

This PR builds an **Agent Core image**, not a Driver. The bot review procedure
on the PR is `/request_bot_review core`. The company build result supplies an
image tag; the authorized Dashboard upgrade/restart helper installs it onto the
approved Core machine. Do **not** use the Driver deployment path or run a second
Core next to the production one. The PR must not merge or trigger Bumi deployment
without appropriate approval.

A successful PR build/test, a successful Dashboard upgrade, and real Bumi QQ
acceptance are three separate milestones. If upgrade causes regressions, use
the official Core rollback procedure and known previous image tag, preserve
persistent data, then recheck the original Channels and QQ Gateway.
