# Review rules — all components

These apply to every PR in either repository. Component-specific rules are
appended from the matching file.

## Infrastructure changes: be very strict

Treat any change to build or packaging infrastructure as needing an explicit
justification in your review. For each such file, say **why the change is
necessary** and **whether it makes the image bigger**. If the PR does not
explain it, that is itself the finding.

Blast radius is not equal. Check which tier the change lands in:

| File | Who is affected |
|------|-----------------|
| `phanthymotus/deploy/ros-base/Dockerfile` | **Everything.** All 13 drivers, agent-core and perception build `FROM` it. It also rewrites `/ros_entrypoint.sh` with `sed`, which every downstream image inherits. It lives in `phanthymotus` but the drivers in `phanthymotus-driver` depend on it, so a change here is a **cross-repo** change. |
| `phanthymotus-driver/dji/base/Dockerfile` | The three DJI drones (`psdk-base`). |
| `phanthymotus-driver/common/**` | Shared Python imported by every driver. |
| One component's `Dockerfile` | That component only. |

Flag as image growth, and ask whether it is avoidable:

- new `apt-get install` packages, especially without `--no-install-recommends`
- new `pip install` entries
- a changed or newly pinned base image
- a new `COPY` bringing a large path into the image
- a new build stage, or removal of `rm -rf /var/lib/apt/lists/*` style cleanup

Prefer: reusing what the base image already provides; adding a dependency to the
component's `requirements.txt` rather than inlining `pip install` in the
Dockerfile; downloading large artefacts at build time from COS via a Dockerfile
`ARG` instead of committing them.

If a PR touches a Dockerfile *incidentally* — reformatting, reordering, comment
churn — say so and recommend dropping it from the PR. Dockerfiles should change
only when the change is the point.

## File size: anything large belongs in COS, not the repo

Any added file over **500 KB** must be called out explicitly, and the review
should ask for it to be moved to Tencent COS and fetched by URL instead.

The bucket the project already uses:

```
https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public/
```

The established patterns, in rough order of preference:

1. **Dockerfile `ARG` pointing at COS** — how `unitree/g1`, `unitree/go2`,
   `unitree/r1` and `pnpbotics/adam` all obtain `cyclonedds-0.10.5.tar.gz`.
   This is the right answer for a driver needing an SDK tarball.
2. **A manifest entry downloaded at runtime** — `perception/utils/model_downloader.py`
   does this for every ASR/TTS/KWS/VAD model.
3. **A build-time fetch in the Dockerfile** — `perception/Dockerfile.jetson`
   pulls CLIP weights this way.

Flag these regardless of size, because they are the wrong *kind* of file to
commit: `.tar.gz`, `.zip`, `.so`, `.a`, `.pt`, `.onnx`, `.whl`, `.bin`.

Existing offenders to cite as precedent when explaining why this matters — all
of them are *under* 1 MB, which is why size alone is not the test:

- `dji/M300/third_party/psdk_lib-3.8.0-m300.tar.gz` — 720 KB, still in the tree
- a 3.2 MB tarball and a 1.7 MB JSON that were removed but permanently bloat
  every clone, because git history keeps them
- an **x86_64** `.so` committed to `unitree/go1` in an ARM64-only project
- a committed `.zip` of message definitions in `deep_robotics/lynx_m20`

Note that `.gitignore` blocks none of this — it only ignores `*.jpg`/`*.png`.
Nothing mechanical prevents this class of PR, so the review is the only gate.

Downloads in this codebase do not verify checksums. A new one that follows the
existing pattern is consistent, but mentioning the supply-chain gap is fair.

## Secrets and credentials

Flag any added or modified `.env`, key, certificate or credential file.
`*.example` and `*.sample` are templates and are fine. Look for hardcoded tokens,
passwords and registry credentials in code and in Dockerfiles.

## Correctness, in priority order

1. **Correctness** — bugs, races, unhandled errors, wrong logic. State the
   concrete consequence: what input produces what wrong behaviour.
2. **Security** — unvalidated input, unsafe subprocess or shell use, path
   traversal, secrets in logs.
3. **Architecture** — does the change respect the Agent Core / Perception /
   Driver separation, and the MCP boundary between them?
4. **Quality** — naming, dead code, duplicated logic, missing error handling.

## How to review

Read before judging. Use the tools: read the component's docs, read the files
the PR changes, and compare against an existing implementation of the same kind.
A finding you cannot point at a `file:line` for is usually a guess — either
confirm it by reading, or leave it out.

Do not restate what the diff does as though it were a finding. If the PR is
fine, say so plainly rather than manufacturing issues.
