# Built-in personal memory skill and user controls

User requested a complete necessary built-in skill, active only with the personal-memory switch, and user control over personal data. Keep the psychology support instructions built-in for now; do not split/install an external plugin.

Deliverable: a bundled immutable personal-memory SKILL.md with identity, consent, extraction/classification, provenance, hypotheses, public-review boundary, recall, pause/resume and user-data controls. Register it as the sole active mandatory skill in personal mode; disabled mode unchanged.

Backend controls: owner-scoped paginated structured export/view, correction, withdrawal of public contributions, clear data and delete profile/bindings. Pause recording without deleting old data. Destructive/corrective changes invalidate active context and suppress saving the management turn, so deleted information is not written back by the same turn. Single-memory correction/deletion also clears the user's raw chat history to avoid reconstructing the old information from echoes; the skill must disclose this scope and obtain the user's agreement before executing it. It preserves other structured memories. Full deletion includes public contributions; biometric templates remain managed by perception and must be clearly distinguished from identity links.

No claim of encrypted storage, physical-disk erasure, offline-backup cleanup, medical validation, or automatic wiping of perception recordings. Classification/consent understanding remains a model behavior; owner checks and write/retrieval controls are deterministic.

Tests first: scope/guest controls, pause preservation/no writes, export pagination, correction/delete history invalidation, public withdrawal, profile deletion, skill gating. Add real-loop regressions for stale prompt/history and management-turn resaves. Full Python/JS regression and independent review before commit. Original checkout and deployment untouched.

Implemented and tested:15 rights tests green, including real-loop stale-context and post-barrier revocation, SQL conditional inserts/upserts against permission-read races, old four-column user schema migration and per-owner control. Management tools require confirmation before broad history clearing. Built-in skill registered in visible/active lists only in personal mode.

Independent review found one Important idle transient-cache issue: the next turn adopted the new DB revision before reusing old RAM history. Fixed by tagging the transient cache with its revision and checking before reuse; also reject final-turn RAM caching when an external revocation lands after model completion. Both reproductions RED→GREEN. Focused suite54 passed after fixes. No minor findings. Real recognition/replay resistance, clinical/legal compliance, operator trust, physical erasure/provider copies and perception-template deletion remain explicitly outside the claimed implementation.

Final verification after review fixes:1553 Python tests passed,1 skipped,19 subtests passed in40.69s. All213 JavaScript tests passed; git diff --check clean. Original checkout remains clean on feat/asr-wake-direction. Feature remains default-off; no deployment, merge or push.
