# QQ Official Channel — cross-device deployment hardening (PR #373)

## Scope and acceptance boundary
The QQ adapter runs inside the existing Agent Core. It does not require a
second container or a workstation relay. Do not place Tencent application
credentials in source, images, logs, PR comments or Canvas records.

This change addresses migration between a working developer installation and
an existing device, including a device with robot Cards already on its Canvas.

## Changes
- Existing QQ channels referencing the historical mounted credential location
  can migrate once into the durable managed credential store, only when a
  known installation-local legacy file is readable and has strict POSIX file
  permissions. Missing/unsafe files leave the old configuration unchanged.
  The old file remains available for rollback.
- Empty Canvas behavior remains the same. An existing Canvas with exactly one
  decision_core gets the missing QQ input/output Cards and connections added,
  without replacing its robot Cards or existing edges. Repeated setup is
  idempotent. Missing or ambiguous decision_core requires operator selection.
- QQ inbound conversation state persistence runs outside the Gateway event
  loop. Group full-message events respect the saved per-group mention-only
  choice; undiscovered groups still require a bot mention.
- QQ group @ messaging is enabled by default for new channels. Existing
  pre-default QQ channel configurations receive an idempotent, QQ-only group
  scope migration on Core startup unless explicitly opted out. No per-group
  pairing, allowlist or approval is required: the first authenticated group @
  event automatically registers that group with chat-only permissions.
  Restoring AppSecret in either UI preserves group/other transport scopes.
- Advanced Tencent event scopes and media feature flags remain under
  administrator control. The code does not silently grant motion or system
  permissions to ordinary group members, nor bypass platform event permissions.

## Local checks
- Focused cross-device / QQ-related regression suites: 72 passed, 1 skipped.
- All discovered QQ tests on Windows: 197 passed, 10 skipped, 6 unable to pass.
  The six are two unsupported Windows symlink-creation cases, two cases
  requiring the official QQ SDK not installed in this Windows Python, and two
  cases where Windows holds a temporary SQLite database file open.
- Python compileall and QQ Dashboard JS syntax checks passed.
- A Linux/ARM64 company-image test run is required to close the POSIX
  credential permission and SDK test cases.

## Deployment and real-device checklist
1. Obtain a fresh company Core image from a commit containing this change,
   via the existing PR build/review flow. Do not confuse an earlier image
   with the new commit, and keep the previous image tag for rollback.
2. Upgrade the authorized Agent Core through its existing Dashboard/update
   service. Preserve its data/config volume and existing robot Drivers.
3. Check that the QQ credential is reported usable and Tencent Gateway
   reports connected. If the legacy credential is inaccessible, use the
   Dashboard recovery interface; do not add a mandatory new Docker mount.
4. Verify existing robot Cards and connections are unchanged. Ensure the
   Canvas project is running before testing reply execution; changing the
   Canvas layout does not deliberately auto-start robot motion or the
   project as a whole.
5. Run QQ private messages, group @, group mention policy, pictures/files,
   owner-vs-ordinary permission tests, independent chats, reboot persistence,
   and operation without the development PC.
6. Keep real media/scope entitlement and remote model availability separate
   from locally passing mocked transport tests. Record failures and roll back
   through the existing Core upgrade service when needed.

Do not call an image build, a container launch or a unit test a successful
end-to-end QQ acceptance until the mobile QQ exchange succeeds on Bumi.

