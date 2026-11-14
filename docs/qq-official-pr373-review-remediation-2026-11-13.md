# PR #373 — Review findings and resolution

Date: 2026-11-13. Target: the existing Tencent Official QQ SDK PR #373.

## Gateway log-volume blocker

- Removed the unconditional `print` in `QQScopedWebSocket._handle_dispatch`
  for `GROUP_MESSAGE_CREATE`. A busy non-@ group now produces **zero
  gateway-boundary per-message log lines**.
- The event still receives the SDK duplicate check and is dispatched to
  `QQOfficialAdapter`, where mention filtering and every-100-event sampled
  diagnostic logging already exist. This does not relax the @bot gate.
- Added `test_full_group_websocket_does_not_log_every_message`: 201 synthetic
  full-group events reach the adapter callback, a synthetic duplicate is
  suppressed, and the Gateway itself prints nothing. The regression runs even
  on developer machines without the official SDK installed.
- Local targeted validation: the new test and baseline stop-card suite,
  **30 passed**.

## Dependency / lockfile size blocker

- `agent-core/pyproject.toml` retains the necessary pinned dependency
  `qqbot-agent-sdk==1.2.2`.
- Restored `agent-core/uv.lock` **exactly** from company `origin/main`
  (tracked baseline: **800,283 bytes**). This PR no longer submits the
  regenerated 855,093-byte lockfile as a changed file.
- The existing Dockerfile copies `pyproject.toml` (not `uv.lock`) and runs
  `uv lock && uv sync --no-dev --no-install-project` at image-build time.
  Therefore the newly declared SDK is resolved and installed at build time.
- **Tradeoff:** the baseline committed lockfile does not encode the new
  dependency. Workflows that require `uv sync --frozen` must regenerate a
  lockfile with `uv lock` first. This is the existing Docker build strategy;
  a repository-wide synchronized-lockfile policy would need a maintainer-level
  decision because `main` itself exceeds the 500-KB check.
- No COS upload or Dockerfile change was introduced.

## Core image footprint

Installed package sizes observed in the local Linux test image (not
incremental Docker image measurements):

| Package | Version | Installed bytes |
| --- | --- | ---: |
| `qqbot-agent-sdk` | 1.2.2 | 234,041 |
| `cryptography` | 50.0.2 | 15,057,177 |
| `cffi` | 2.1.1 | 726,781 |
| `pycparser` | 3.11 | 214,051 |
| **Combined installed files** | | **16,232,050 B (~15.5 MiB)** |

Some transitive packages may already exist in the base image. The exact
incremental Core image size needs a reproducible before/after image comparison
in the company build environment. The SDK is required by the official
Gateway adapter and is not a development-only package.

## Separate unrelated stop-card change

The short-deadline stop-card polling adjustment was unrelated to QQ.
`agent-core/src/api/config.py` and
`agent-core/tests/test_stop_project_stages.py` were restored to company
`origin/main` for this PR. The behavior change should be reviewed, tested,
and merged separately, rather than coupling QQ rollback to device lifecycle.

## Privileged C2C owner security boundary

- QQ access remains reply-only unless **all** conditions hold: verified
  `official:<appid>:<openid>` C2C principal, an explicit ACL `owner` role
  with tool filters, and a local `qq_c2c_owner_grant` that matches the
  identity and has `enabled: true`.
- Grant/revoke happens through the trusted local
  `agent-core/scripts/qq_promote_c2c_owner.py` operator path.
- Nonowner users, group/Guild principals and mismatched grants do not inherit
  owner tools. Dispatch-time ACL checks remain required.
- Maintainer approval is still required for enabling privileged remote tools
  on a deployed Core. No personal OpenID or credential is committed.

## Remaining release gates

1. Push the updated PR branch and rerun company Agent Core CI/review on the
   new commit; local tests do not substitute for CI.
2. Verify that `uv.lock`, `test_stop_project_stages.py` and any unrelated
   stop behavior are absent from the PR diff.
3. Obtain deployment authorization; live QQ media/interactions and Bumi
   hardware verification are separate acceptance checks.
