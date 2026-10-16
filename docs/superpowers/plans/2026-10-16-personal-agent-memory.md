# Personal Agent Memory Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User explicitly asked to develop now.

**Goal:** Automatically unlock isolated personal memories through either existing biometric modality without affecting disabled-mode behavior.
**Architecture:** Feature-gated SQLite personal store and per-turn server context. Strict perception-source identity routing and separate private history; scoped memory tools and a psychology support skill.
**Tech Stack:** Existing Python, SQLite, FastAPI and pytest; no new dependencies.
**Spec:** docs/superpowers/specs/2026-10-16-personal-agent-memory-design.md

## Global Constraints
- Default disabled; original checkout untouched; no deployment/push/merge.
- Do not use names or equal face/voice p-N values as implicit identity binding.
- No private transcript in legacy history or global request logs.
- Guest, conflict and unknown identities never inherit the previous user.
- Face/voice-only route deliberately accepts recognition/replay risk.

## Review Focus
- Concurrent/global caches cannot retain another user's personal content.
- Steering/batching cannot join two identities in a single private turn.
- Remote user text cannot manufacture biometric authentication.
- Disabled mode must retain all previous tools and startup behavior.
- Model-invented public scope cannot publish private analysis.

### Task 1: Identity, storage and scoped tools
Files: create agent-core/src/personal_memory.py, personal_memory_tools.py, tests/test_personal_memory.py.
Interfaces: enabled(); current_user(); resolve_event(event); bind_context(user_id); create_user(name, remember, onboarding); bind_identity(user_id, modality, external_id); remember/recall/history/delete functions.
- [x] Write and run failing tests for independent face/voice namespaces, OR matching, conflict/guest cases, scoped storage, opt-out and public approval.
- [x] Implement minimal SQLite store and context-bound tools, supportive instructions.
- [x] Run focused tests and commit.

### Task 2: Turn routing, prompt and persistence integration
Files: collector.py, event/llm.py, prompt.py, event/skills.py, daily_summary.py, llm_logger.py, log.py; tests/test_personal_memory_integration.py.
Interfaces: Task 1 identity resolver and context, personal history functions.
- [x] Write and run failing tests for batch separation, forged remote identity, switch/restart isolation, prompt separation, restricted dispatch and legacy paths.
- [x] Integrate gated personal-mode lifecycle, scoped persistence and memory tools; block global side channels in this mode.
- [x] Run focused and existing affected tests and commit.

### Task 3: Operator API, documentation and regression
Files: api/personal_memory.py, start.py, config.py, docs/PERSONAL_MEMORY.md, tests/test_personal_memory_api.py.
Interfaces: Task 1 user/binding/store functions; existing app auth protects operator routes.
- [x] Write and run failing API tests for setup, conflicts, memory approval and deletion.
- [x] Implement API, configuration defaults and deployment/rollback instructions.
- [x] Run full agent-core suite, inspect diff, fresh reviewer; fix material issues, commit.

## Execution ledger
- Original baseline interpreter was Python 3.9 with missing dependencies; use existing /tmp/pr306-core-testenv Python 3.12 environment.
- Native worktree tool returned Not a git repository for the parent chat directory; Git fallback created the sibling worktree from cdbbcd6.
- User authorized development; spec and plan are recorded without another approval round.
- Task 1 complete: 7 store/identity tests RED→GREEN; no new runtime dependencies.
- Task 2 complete: integration tests RED→GREEN, including the real tool dispatch loop; global HTTP request logging needed an additional guard verified RED→GREEN.
- Task 3: operator API and usage documentation written; focused feature tests 24 passed.
- Test isolation repair: an existing test leaves config.main as a dictionary. Our fixture now binds a fresh ConfigDB rather than modifying that leaked dictionary. Reproducer 24 failed → 96 passed; no unrelated production code changed.
- Baseline: 1506 passed, 1 skipped, 19 subtests passed (included first 7 feature tests; legacy baseline is 1499 passing tests).

- Final review (independent reviewer review_personal_memory): Important queued-ASR attribution bug confirmed. Fixed by freezing typed identity snapshots at ingress, verified with queue and actual collector-ingress regressions RED→GREEN.
- Final review boundary ruling: DDS subscribers are not assumed to be trusted operators. Personal-mode decision broadcasts are disabled; actual turn-loop regression RED→GREEN. Cost: consumers of /decision_core receive no private-mode decisions; disabled mode unchanged.
- Final quality fix: registered users opting out of durable memory retain only their current conversation in RAM (18 turns, clears on switch or next entry after 5 minutes). Regression RED→GREEN.
- Final quality fix: identical ASR text from different users is not deduplicated. Actual ingress regression RED→GREEN.
- Focused suite after fixes: 28 passed. No deferred minor findings. Real biometric accuracy/replay resistance and hardware/cloud integration remain unverified, as specified.
- Final verification: full agent-core suite 1527 passed, 1 skipped, 19 subtests passed in 39.46s; focused final suite 28 passed; git diff --check clean.
- Original checkout verified clean, still feat/asr-wake-direction at cdbbcd6. Feature remains disabled by default. Keep independent branch/worktree as requested; no merge, push or deployment.
