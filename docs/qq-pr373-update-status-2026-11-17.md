# PR #373 2026-11-17 update and acceptance status

> **Review note:** This document describes the latest local update pushed to the **existing** [PR #373](https://github.com/4paradigm/phanthymotus/pull/373). Older PR descriptions referring to “private text only” and an intentionally unsynchronized lockfile are out of date; review the current code and this document instead.

## Implemented

- QQ official SDK QR-code onboarding and connection recovery, expiring/cancellable QR sessions, private credential persistence and redaction.
- Existing QQ installations and legacy mounts: preserve configured channels, SDK defaults, Canvas bindings, and recovery behavior.
- QQ private/group messaging, media receipt, Gateway health/reconnect, diagnostics, improved dashboard/channel onboarding.
- Multi-channel **input** subscription reconciliation for Canvas-bound Channel Cards; active subscription readiness is distinguished from configured intent.
- Per-Card startup statuses identified by Card IDs rather than only tool names.
- Updated `pyproject.toml` and synchronized `uv.lock` (including SDK and QR dependencies).
- Preserved the remote branch's prior fixes for legacy credential recovery and default group events while integrating this local update.

## Verified before push

- Focused Python tests after remote merge: **75 passed, 1 skipped**.
- QQ onboarding front-end tests: **5 passed**.
- `git diff --check`: passed.
- Full company Linux CI and real concurrent three-Bot E2E are **not verified** for this HEAD. Earlier CI results must not be extrapolated.

## Open issues / review blockers

1. **P0 multi-Bot LLM tools:** Several `channel_reply` Card instances register duplicate LLM tool names (`Tool names must be unique`); naïve de-duplication may also select the wrong instance/QQ Bot. This bug is **intentionally left unresolved** by user request. See [multi-Bot routing issue](qq-multi-bot-tool-collision-routing-2026-11-17.md) for reproducibility, safe design and acceptance criteria. No claim of complete multi-Bot reply isolation.
2. **Visual understanding:** QQ image attachments can be downloaded and stored, but the observed local `event.llm.vision_input` was disabled and provider-specific model image requests remain unverified.
3. **Approval and deployment:** The updated PR does not authorize deployment to Bumi, change Watch7 Driver containers, or imply CI/review approval.

## Security and changed-scope notes

- Never check in AppSecret values, QQ private credential directory, local runtime databases, smoke-test temporary paths or unrelated developer utilities.
- Preserve QQ's server-enforced chat-only tool boundary; QQ must not implicitly gain robot/device/shell permissions.
- Keep Feishu compatibility and independent QQ/Feishu chat history and permissions. Do not conflate inbound attachment support with visual LLM capability.

## For reviewers

Please check the QR authorization and persistence flow, DDS subscription lifecycle, group-event filtering, Canvas multi-instance behavior and migration of old QQ settings. Before full multi-Bot support is declared, add deterministic context-bound reply routing and simultaneous QQ+Feishu regression tests.
