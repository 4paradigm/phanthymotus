# PR #373 — Review Agent 7e76144 findings and remediation

Date: 2026-11-13. Scope: existing Tencent Official QQ SDK PR #373. No new PR.

## High: trusted QQ C2C owner tool access

**Previous behavior:** A valid QQ private OpenID granted the ACL `owner` role
automatically bypassed the QQ platform tool cap. A role-only mistake could
expose Bash and robot movement tools.

**New mechanism (default-deny):**

- QQ principals are reply-only unless **all** conditions hold:
  1. Verified three-part `official:<appid>:<openid>` C2C identity.
  2. Runtime ACL entry is `owner`, with explicit tool filters.
  3. Local-only `qq_c2c_owner_grant` record has `enabled: true` **and**
     `principal` exactly equal to that identity.
- The separately signed-off grant is set by a trusted local operator using
  `agent-core/scripts/qq_promote_c2c_owner.py --promote`; `--revoke`
  disables it before demoting the ACL. The script refuses to select an
  identity unless the active official AppID has exactly one registered C2C user.
- No QQ command, group message, UI toggle, or repository default sets this grant.
  No personal OpenID, runtime ACL record, or credential is shipped in Git.
- Other C2C users, all group/Guild principals, and wrongly assigned owner roles
  remain reply-only. Dispatch-time ACL rechecking remains intact.
- The trusted owner's **full tool set remains available on their own
  locally approved Core**. This is deliberate privileged remote access and
  requires repository maintainers to accept that security boundary.

Validation: `test_qq_owner_acl.py` and `test_qq_owner_provisioning.py`
cover missing grant, mismatched identity, group, revoke, role, and tool filter.

## High: device stop lifecycle

**Previous behavior:** _stop_cards slept before the first `info` check
after a device declared a pending-release flag. Under the test's short 0.2s
per-card deadline, that unnecessary delay could turn a successful release
into `ok: false` and a false device-running warning.

**Fix:** Query `info` immediately after the initial `stop`, never reissue
`stop` while pending, and rate-limit only *subsequent* `info` polls.
Retains independent per-card deadlines, concurrent teardown and explicit
failure reporting. No hard-coded device-specific release vocabulary.

Validation: **29/29** `tests/test_stop_project_stages.py` cases passed
in the local Linux Core environment, including both boolean branches of
`test_a_declared_pending_flag_polls_info_without_reissuing_stop`.

## Medium: lockfile consistency and repository-size policy

- `agent-core/pyproject.toml` pins `qqbot-agent-sdk==1.2.2`.
- The committed `agent-core/uv.lock` is synchronized with that dependency;
  `uv lock --offline --check` passes. This preserves `uv sync --frozen`.
- Baseline company `main` already contains an approximately **800,283 B**
  lockfile (above the 500 KB review threshold). Regenerated lock: **855,093 B**
  (**+54,810 B** to synchronize the added SDK dependencies).
- **Maintainer policy exception requested:** retain the lockfile inside the
  repository despite the rule's 500 KB threshold. Moving it to COS would
  break conventional locked/frozen local builds and integrity review. The
  oversized file is not a binary/model asset. No new Dockerfile changes.
- This exception cannot be self-approved by a code change; maintainers must
  explicitly approve it or provide a repository-wide lockfile strategy.

## Medium: SDK package / Core image footprint

Measured in the local Linux test image, installed package file sizes:

| Package | Version | Installed bytes |
| --- | --- | ---: |
| `qqbot-agent-sdk` | 1.2.2 | 234,041 |
| `cryptography` | 50.0.2 | 15,057,177 |
| `cffi` | 2.1.1 | 726,781 |
| `pycparser` | 3.11 | 214,051 |
| **Sum of these packages** | | **16,232,050 B (~15.5 MiB)** |

This is installed package footprint, **not a measured incremental Docker image
size**: some transitive dependencies may already be present. The official SDK
requires its crypto dependencies; an SDK-less implementation would reimplement
QQ authentication and Gateway transport, increasing maintenance and security
risk. The Core Dockerfile remains unchanged.

## Suggestion: group log growth

`QQOfficialAdapter` logs SDK group-event diagnostics at event number 1 and
every 100th event; non-@ events use the same counter. No QQ IDs, message
contents, media URLs or credentials are logged. Regression test sends 201
messages and asserts only 3 sampled diagnostic pairs.

## Remaining release gates

- Company Core image build + full CI and review **on the new commit**.
- Explicit maintainer approval of the lockfile size exception and privileged
  C2C owner feature, plus the normal Machine Owner rollout authorization.
- Live QQ media and interaction permissions, and Bumi hardware acceptance,
  remain separate; local unit tests do not count as those platform checks.
