# decision_core personal memory switch

Goal: configure the existing personal-memory backend in the decision_core card, with automatic sources/output from canvas wiring. Do not split the psychology skill or modify the original checkout.

User correction: no duplicate ASR/face/TTS selectors. Inbound ASR/face connections are identity sources; local TTS executor links are output authorization. Initial identity enrollment/binding remains separate.

Mode changes are staged until AgentCore restarts, because initialization chooses shared tools/background workers versus private mode. The modal reports active versus saved mode, and start refuses a pending mismatch before starting hardware. Route changes are resolved on each project start using the existing live topic resolver.

Tasks:
- [x] Add failing backend tests for card saving/reading, authentication, active-vs-pending mode, unrelated configuration, route derivation and missing wires.
- [x] Add runtime settings snapshot and canvas route adapter; integrate config save synchronously so errors reach the user before persistence.
- [x] Add the schema switch and actual modal state/notice loading; execute modal preamble tests RED→GREEN.
- [x] Exercise real project start with a driver fake: live ASR topic overrides stale snapshots; pending mode starts no devices (RED→GREEN).
- [x] Update usage instructions for wiring and card switch, retaining manual configuration compatibility.
- [x] Full Python/JavaScript regression and independent review; fix material findings and commit.

Focused verification: 59 tests passed. Full pre-final run: 1534 Python tests passed, 1 skipped, 19 subtests; 213 JavaScript tests passed. Two project-start tests were added afterward and are included in final verification.

Final regression: 1536 Python passed, 1 skipped, 19 subtests; 213 JavaScript passed. Independent review found no Important/minor issues; independently ran37 feature tests. Original checkout remains clean on feat/asr-wake-direction. No deployment.
