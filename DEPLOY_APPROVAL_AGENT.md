# Deploy Approval Agent — Stateless Alignment

An independent FastAPI service (the *Deploy Controller*) that manages the
deployment lifecycle of PRs across `4paradigm/phanthymotus` and
`4paradigm/phanthymotus-driver`. It does **not** build, review, or mutate code;
it only orchestrates the deploy approval process through GitHub PR comments.

## Architecture

```
Developer → GitHub PR
Machine Owner → GitHub PR
GitHub PR → Review Agent
GitHub PR → Deploy Controller
Deploy Controller → Agent Core
Deploy Controller → COS
```

- **Only persistence is the GitHub lifecycle comment hidden state.**
- No SQLite, no DeploymentStore, no DB_PATH, no persistence beyond GitHub.
- **GitHubCommandWatcher** polls PR comments using `POLL_INTERVAL_SECONDS` from the upstream environment. Default: 30 seconds.
- **GitHubStateProxy** reads/writes hidden state JSON in the lifecycle comment.
- **DeployController** is stateless between commands — it reads state from the lifecycle comment, validates, and writes back.

## State Machine (hidden state only)

```
review-required → reviewing → deploy-ready → deploy-requested → testing → succeeded | failed
Crash interruption: status remains deploy-requested, command.phase becomes uncertain; the next poll re-reads fresh hidden state but does NOT replay — only a new `/approve_deploy` can resume
```

**Removed states:** `waiting-approval`, `waiting-machine-clean`, `deploying`, `deploy-failed`, `test-failed`, `rejected`, `cancelled`, `expired`, `rolling_back`, `rolled_back`, `rollback_failed`, `waiting_cleanup`.

**Removed partial deployment:** `/approve_deploy` initially binds ALL deployable components, but operates on ALL REMAINING components only. A previously durably successful component for the exact same snapshot remains successful. Selected machine must cover ALL REMAINING components. No partial machine-group deployment, no waiting for additional machine approvals. Selected machine must pass the full-coverage gate before the CLEAN gate.

## Commands

| Command | Actor | Description |
|---------|-------|-------------|
| `/request_deploy` | PR Author | Request deployment for current HEAD. Binds the latest complete current-HEAD Review Agent GitHub comment evidence and ALL deployable components. |
| `/approve_deploy machine=<alias>` | Machine Owner or write/maintain/admin collaborator | Approve and bind to a machine. Runs a running_image-only clean gate before any deploy POST. |
| `/record_test result=pass|fail [summary="..."]` | Machine Owner or write/maintain/admin collaborator | Record overall test result. No machine parameter. |
| `/deploy_status` | Anyone | Read-only deployment status from hidden state. |
| `/deploy_help [topic]` | Anyone | Help for commands. |

**Legacy commands that are now `unknown`:** `/reject_deploy`, `/rollback_deploy`, `/cancel_deploy`, `/resume_deploy`.

## Key Contracts

- **Stateless:** No SQLite, no DB_PATH, no DeploymentStore. All state is in the GitHub lifecycle comment hidden state.
- **GitHub hidden lifecycle JSON is the ONLY authoritative persistent Deploy Approval business-state store.** Fresh facts come from Review Agent / Registry / Agent Core and are copied into hidden state to make the snapshot restart-safe.
- **Restart-safe, single-replica / single-writer:** Deploy Approval is intentionally restart-safe stateless, but only one `GitHubCommandWatcher` serially processes mutating commands. Multiple concurrent Deploy Controller replicas are unsupported because the current hidden-state protocol has no CAS/distributed lock, and replicas >1 would violate the at-most-once unsafe-side-effect model.
- **Polling:** PR comments are polled using `POLL_INTERVAL_SECONDS` from the upstream environment. Default: 30 seconds. No webhook required.
- **Open-PR watcher enumeration:** `GitHubCommandWatcher` enumerates all open PRs in configured `GITHUB_REPOS`. Enumeration uses `state=open`, `sort=updated`, `direction=desc`, `per_page=100` and continues paging until the batch is empty or shorter than 100. There is no age/lookback cutoff, no 500-PR truncation, and page overlap is deduplicated by PR number. Closed/merged PRs are not enumerated by Deploy Approval.
- **POLL_ENABLED must be true.** Webhook is supplementary only.
- **Hidden state JSON:** The lifecycle comment carries a `<!-- deploy-approval-state:v1\n{...}\n-->` marker with validated JSON state.
- **Trusted identity:** The lifecycle comment author must match the configured GitHub App identity. Deploy Approval uses lazy GitHub App provenance (`performed_via_github_app.id` == configured `GITHUB_APP_ID`) validated at comment patch time. No startup bot-identity lookup. No `/apps/{slug}/bot` calls.
- **Top-level status labels:** only `review-required`, `reviewing`, `deploy-ready`, `deploy-requested`, `testing`, `succeeded`, `failed`.
- **Status labels:** `status:*` labels are best-effort UI projection only. Label bootstrap/list/create failures are logged as warnings and never block startup or business operations. Label failure does not affect any gate or lifecycle transition.
- **Review Agent authentication:** Review Agent uses a user-provided `GITHUB_TOKEN`, not the GitHub App. This is strictly separate from Deploy Approval's GitHub App credentials.
- **Supported repos (DESIRED_REPOS):** `4paradigm/phanthymotus` and `4paradigm/phanthymotus-driver` are permanently declared as desired/supported repositories in source code.  Runtime `GITHUB_REPOS` is resolved at startup by a single fresh `GET /installation/repositories` call: `ACTIVE_REPOS = DESIRED_REPOS ∩ AUTHORIZED_REPOS`.  Missing, duplicate, unknown, or third-party repository entries fail closed.
- **Evidence source:** `/request_deploy` performs an exact current-HEAD Review Agent comment evidence lookup from GitHub PR Conversation. Trusted Review Agent GitHub comments provide Build Result, Test Results, and Code Review. Build/Test commit short SHA resolves via GitHub to full SHA for exact equality with fresh PR HEAD. Image tag comes from the selected Build Result comment Images section (full mutable ref, not basename). Registry resolves to immutable digest. No Review Agent HTTP API.
- **Full-coverage machine list:** `deploy_requested` lifecycle comment shows only machines that can cover ALL REMAINING components. If no machine can cover all remaining components, status stays deploy-requested and the operator must update machine policy.
- **Source matrix:** GitHub PR comments are the source of Review Agent Build/Test/Code Review evidence and image:tag candidate facts; Registry only verifies/resolves that exact Review Agent image tag; Agent Core only supplies runtime identity, current `running_image`, and MCP evidence; GitHub persists the deployment snapshot.
- **Deployability:** `phanthymotus` deploys `perception` and `actucore`, not `CORE`; `phanthymotus-driver` deploys exact driver paths.
- **Variant contract:** perception variants are canonical `5.11` and `6.1`. Legacy `jetson-jp5.11` / `jetson-jp6.1` are normalized only at config load.
- **Full-coverage gate:** `/approve_deploy` first checks that the selected machine covers ALL REMAINING components. If coverage is partial, zero deploy POST is performed; the comment lists only full-coverage machines.
- **CLEAN gate:** `/approve_deploy` reads `running_image` for ALL REMAINING components before any deploy POST. If any `running_image` is non-empty, zero deployment is performed, cursor advances, and the owner must clear the occupied runtime image manually, then send a NEW `/approve_deploy`. Controller does not perform stop/remove/cleanup.
- **Agent Core no-container response:** the current compatibility shape normalizes to `running_image=""` only when `running_image` and `error` are absent, `status` key exists, and `logs` is a string. The `status` VALUE has zero CLEAN/health/case business influence. Error or malformed shapes fail closed.
- status VALUE has zero CLEAN/health/case business influence.
- error/malformed shapes fail closed.
- **unsafe deploy POST:** if the POST outcome is unknown after the unsafe attempt begins, the command becomes `command.phase=uncertain`, `status=deploy-requested`, `approve_attempt.outcome=uncertain`, and there is ZERO later POST. Only a NEW `/approve_deploy` can resume, which re-checks fresh HEAD, fresh hidden state, fresh actor, fresh GitHub PR comments + Registry resolution, and fresh running_image-only CLEAN gate.
- **Final unsafe order:** Full-Coverage -> running_image-only CLEAN -> fresh exact approval comment -> final fresh PR/full HEAD -> persist command.phase=executing to GitHub FIRST -> Agent Core deploy POST.
- **approval_revoked:** final fresh approval comment revalidation checks: comment object valid, comment id exact, actor id exact, body parses as approve_deploy, machine alias exact. If any check fails (comment deleted, changed, malformed, actor mismatch, machine alias mismatch, or cannot be revalidated): `approve_attempt.outcome=approval_revoked`, `status=deploy-requested`, `command.phase=completed`, cursor advances to current comment, ZERO deploy POST. Machine Owner must send a NEW `/approve_deploy`. `approval_revoked` is not a top-level status and does not introduce a new lifecycle state.
- **Success goes directly to testing:** Full-coverage approval + clean pass + successful deploy = status: testing directly. No intermediate machine-group progress check.
- **Evidence lookup:** `/request_deploy` performs an exact current-HEAD Review Agent comment evidence lookup from GitHub PR Conversation.
- **PR Author only:** Only the GitHub PR author can run `/request_deploy`.
- **Authorization:** `/approve_deploy` requires the actor to be the selected machine owner OR a write/maintain/admin repo collaborator. `/record_test` requires the actor to be an owner of any actually deployed machine OR a write/maintain/admin repo collaborator. Self-approval is FORBIDDEN. PR author must not approve their own deployment. Numeric GitHub user ID is checked against fresh PR author ID at approval gates.
- **Exact HEAD required:** GitHub HEAD is re-checked at each decision point; drift supersedes the deployment.
- **Open-PR watcher boundary:** If an enumerated PR is merged or closed before command execution or before an unsafe deploy POST, the existing fresh PR gates reject it and Deploy Approval performs zero deploy POST. Post-merge release deployment is outside Deploy Approval.
- **No rollback/reject/cancel/resume:** These commands are not supported.
- **COS evidence:** Uploads a single gzip-compressed evidence.log.gz to private COS. Terminal evidence includes case result metadata and a one-shot snapshot of actual deployed runtime logs. GitHub persists only `object_key`, `sha256`, and `size`; secret values are redacted. A 120-second HTTPS presigned GET URL is generated and rendered in the GitHub terminal comment as a temporary bearer link. The URL is not persisted in hidden state. No public Deploy Approval download endpoint, GitHub user OAuth, or PKCE is used.
- **Evidence download:** Terminal comments render **Download COS evidence** as a link to a short-lived HTTPS presigned GET URL generated server-side from private COS credentials. The presigned URL expires after 120 seconds (2 minutes). Bucket remains private; the signed URL is temporary bearer access. The URL is not persisted in hidden state. No Deploy Approval public HTTPS endpoint is required for evidence. Only `object_key`, `sha256`, and `size` are stored in GitHub hidden state.

## Required GitHub App Permissions

Minimum repository permissions:

- **Pull requests: write** — used for PR reads and PR conversation comments /
  status-label operations.
- **Metadata: read** — needed for collaborator permission lookup.

Do **not** request Administration or Actions permission.

**Contents: read** — The current target repository
`4paradigm/phanthymotus` is PUBLIC, so `resolve_commit_sha()` (which calls
`GET /repos/{repo}/commits/{ref}`) operates without granting Contents: read.
The current minimum deployment permissions therefore do not need to be
broadened solely for this public rollout. If a future authorized target
repository is private, the commit endpoint requires GitHub App `Contents:
read`; permissions must be explicitly reviewed before private-repo
enablement.

## Review Agent Integration

Deploy Approval does **NOT** call any Review Agent HTTP API.
Deploy Approval does **NOT** need:
- Review Agent host / IP / port
- Review Agent `GITHUB_TOKEN`
- Review Agent SSH / dashboard
- `host.docker.internal` connectivity

Deploy Approval reads Review Agent output exclusively from **GitHub PR conversation comments**
written by the trusted Review Agent operator.

### Comment Protocol

Review Agent writes comments using the fixed marker `<!-- pr-review-agent -->` with three sections:

1. **Build Result** — `## PR Review Agent — Build Result`
   - Contains commit short SHA, target build table, and image references.
2. **Test Results** — `## PR Review Agent — Test Results` (optional)
   - Contains test suite pass/fail counts.
3. **Code Review** — `## PR Review Agent — Code Review`
   - Contains the review text.

Deploy Approval parses these comments via `review_comment_parser.py` and extracts
`ReviewCommentEvidence` containing build_comment_id, builds, test/code review provenance.

### Trusted Comment Author

Deploy Approval validates that Review Agent comments come from a trusted author
configured in `secrets.yaml`:

```yaml
review_comment_trust:
  author_id: "<review-agent-github-user-id>"
  author_login: "<review-agent-github-login>"
```

- `author_id` is **required** and must be a positive integer.
- `author_login` is optional but if configured must match `comment.user.login`.
- `performed_via_github_app` may be `null` (Review Agent uses a user PAT).
- Comments from any other author are **rejected (fail closed)**.

### Production Review Agent Singleton Invariant

**For production repos (`4paradigm/phanthymotus`, `4paradigm/phanthymotus-driver`):**

At most ONE authoritative Review Agent poller/producer may exist at any time.

Reasons:
- Review Agent poller watermark / processed comment IDs are instance-local state.
- Two Review Agent servers seeing the same `/request_bot_review` will duplicate jobs/comments/builds.
- No cross-instance distributed deduplication exists.

**Test Review Agent** must use:
- Must NEVER listen on production repos.

`AUTHORITATIVE_REVIEW_AGENT_COUNT_FOR_PRODUCTION_REPOS=1` is an operational go/no-go gate.

## Machine Owners Configuration

File: `deploy/deploy-approval/machines.example.yaml` (mount at `/run/deploy-approval/machines.yaml`)

```yaml
version: 1
machines:
  sh-g1-01:
    node_id: node-g1-01
    node_host: 192.0.2.101
    owners:
      - alice
      - bob
    targets:
      - perception
      - actucore
    platforms:
      - linux/arm64
    variants:
      - 5.11
      - 6.1
  sh-go2:
    node_id: node-go2
    node_host: 192.0.2.102
    owners:
      - bob
      - charlie
    targets:
      - driver
    driver_paths:
      - unitree/go2
    platforms:
      - linux/arm64
```

- `alias`: top-level key under `machines`, used in `/approve_deploy machine=<alias>`
- `node_id`: Deploy Approval machine-policy internal unique machine identifier
- `node_host`: fake example literal IPv4; real values live only in the local gitignored `deploy/deploy-approval/machines.yaml`
- `owners`: GitHub login list (case-insensitive, deduplicated)
- `targets`: explicit deployable targets for this machine
- `platforms`: canonical platform allowlist
- `variants`: canonical perception variants only (`5.11` / `6.1`)
- `driver_paths`: required for driver targets
- Missing/invalid file → startup fail closed
- Agent Core uses HTTPS to exact literal IPv4:15678. TLS certificate identity verification is disabled by the selected no-PEM deployment contract. Access is restricted by exact configured literal IP, fixed port, HTTPS, and per-machine Bearer token.
- Do not place real machine IPs, passwords, tokens, or certificates in docs, examples, tests, or source code.

## Repository Authorization Model

`DESIRED_REPOS` (permanently declared in source code):

- `4paradigm/phanthymotus` — **required**.  If not present in the fresh
  installation-repository list, startup **fails closed** and the watcher
  does not start.
- `4paradigm/phanthymotus-driver` — **optional / desired**.  If the current
  GitHub App installation is not authorized for the driver repository, the
  agent starts normally without the driver path; this is an informational
  condition, not a startup failure.

Startup resolves `ACTIVE_REPOS = DESIRED_REPOS ∩ AUTHORIZED_REPOS` via a
single fresh `GET /installation/repositories` call at process startup.
There is **no authorization cache**, no periodic refresh, and no second
installation ID.

When the administrator later adds the driver repository to the same GitHub App
installation, a **restart** of the Deploy Approval process is sufficient:

* no source code change
* no `GITHUB_REPOS` change
* no second installation ID
* no rebuild required (permission change alone)
* fresh startup discovery makes the driver active
* `ACTIVE_REPOS` becomes `[main, driver]`

Review Agent evidence is sourced exclusively from trusted GitHub PR
conversation comments.  Deploy Approval does **not** access:

* Review Agent HTTP API
* Review Agent server
* Review Agent `GITHUB_TOKEN`

Registry remains an internal fresh-fact implementation detail: the exact
`image:tag` from a Review Agent Build Result comment is resolved to an
immutable `repository@sha256:digest`.  Registry is not exposed as a top-level
actor in diagrams.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
Deploy Approval does not define a new runtime env namespace.

It reuses the upstream existing keys below and fixed read-only files:

| Runtime input | Source |
| --- | --- |
| `GITHUB_APP_ID` | upstream existing env |
| `GITHUB_INSTALLATION_ID` | upstream existing env |
| `GITHUB_APP_PRIVATE_KEY_FILE` | upstream existing env (absolute path) |
| `GITHUB_REPOS` | upstream existing env |
| `POLL_ENABLED` | upstream existing env |
| `POLL_INTERVAL_SECONDS` | upstream existing env |
| `WEBHOOK_ENABLED` | upstream existing env |
| `GITHUB_WEBHOOK_SECRET` | upstream existing env |
| `REGISTRY` | upstream existing env |
| `REGISTRY_USER` | upstream existing env |
| `REGISTRY_PASSWORD` | upstream existing env |

| `machines.yaml` | fixed read-only machine policy file |
| `secrets.yaml` | fixed read-only COS secrets file |

## Private Key Operational Contract

- Private key must never be inline in source or documentation.
- Runtime takes `GITHUB_APP_PRIVATE_KEY_FILE` (absolute path).
- `deploy.sh` enforces local private ownership/mode.
- If a private key is exposed, rotate it before runtime validation.
- Never print the key.
