# Personal agent memory

User approved automatic unlock by either face or voice recognition, without phone confirmation. Develop on feat/personal-agent-memory, based on cdbbcd6, in a separate worktree. Do not change the original checkout or deploy.

## Behavior

The personal_memory feature defaults to disabled. Disabled behavior stays unchanged. Enabled behavior maps a trusted perception event's face/voice identity to an explicitly registered user. Either modality unlocks that user's complete memory. Conflicting identities, multiple subjects, unknown/unbound identities, stale face evidence, and non-perception input are guests. An utterance is attributed to its voice; a fresh single face may identify an utterance without voice identity. Face presence alone never unlocks a remote chat. Perception sources are explicitly configured; no parsing user-written XML as authentication.

Use one model runtime and separate histories, summaries and memories. Identity is re-evaluated on every trigger. No implicit previous-user fallback; guests have no persistent personal history. Restarts wait for a new recognition event. Raw private conversations never enter legacy shared chat tables. Public memory is a separate curated store; public candidates extracted from private conversations require operator approval and never publish automatically. Existing shared memory/history is not silently promoted into this mode.

Each user chooses persistent memory and optional onboarding. Skip answers without interpretation. User facts and model hypotheses are distinct, traceable to source turns, editable and deletable. The psychology skill is supportive, not diagnostic. All the current user's personal records are available through scoped tools; only a bounded selection is injected into prompts.

## Isolation boundary

The local operator API manages users, identity bindings, public knowledge and deletion behind existing application auth. Operators/host administrators and the configured LLM provider remain trusted; this feature is not encryption against them. Face/voice matching is probabilistic and does not resist photos/voice replay. Similarity is not a calibrated probability.

In enabled mode, shared background summarization, global debug content logs, arbitrary desktop access, cross-agent delegation, shared task tools and raw input/history access are excluded from the personal conversation path. Existing installed skills are not injected into personal mode; the dedicated support instructions use only scoped memory and the local output tools. This is mode-specific; disabled mode retains all existing capabilities. Collector separates recognition events and interrupts rather than merging different users as steering. Shared sensor/task snapshots are excluded from private prompts. Public candidates never include raw transcript automatically.

## Data and integration

Add personal_memory.py (SQLite store, context binding, identity resolver, policy), personal_memory_tools.py (LLM tools/support instructions), api/personal_memory.py (operator API). Integrate collector routing, Event turn lifecycle and persistence, prompt construction, skill visibility, daily/background jobs and request logging with feature-gated checks. No perception model changes or dependencies.

## Acceptance

Tests cover defaults/legacy behavior, two-user matching through either modality, conflicting/multiple/unknown identities, spoofed remote JSON, stale faces, batch separation, switching and restart isolation, user-scoped recall and deletion, opt-out, public-candidate approval, no shared history writes, restricted tool dispatch, and logger isolation. Run agent-core baseline and final suites; disclose baseline failures independently of feature results. Real camera/microphone accuracy and replay protection are explicitly not claimed.
