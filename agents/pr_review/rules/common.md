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
   `unitree/r1` and `pndbotics/adam` all obtain `cyclonedds-0.10.5.tar.gz`.
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

## Container logs: stdout is a framed stream, not a console

A container's stdout/stderr **is** its Docker log, and the daemon frames every
write into a record. Two mistakes break that framing badly enough that
`docker logs` returns nothing at all for the life of the container:

```
Error grabbing logs: invalid character '\x00' looking for beginning of value
Error grabbing logs: error unmarshalling log entry: proto: illegal tag 0 (wire type 6)
```

Flag these:

1. **Any `dup2` / `os.dup` on fd 1 or 2.** `fd 1 == the container log` is an
   invariant. Redirecting it leaves two buffered writers on one pipe (writes above
   `PIPE_BUF` are not atomic, so lines interleave and tear a record) and costs
   every `multiprocessing`/`subprocess` child its stdout, silently. If a native
   library is noisy, silence it with its own env var, not with fd surgery.
2. **`truncate`/`> file` against anything under `/var/lib/docker/containers/`.**
   Truncating a live log resets the file size but not the daemon's write offset;
   the next write lands past EOF and the kernel NUL-fills the gap, producing the
   errors above. `docker restart <c>` is the correct way to reclaim log space.
3. **New `print`/log calls inside a per-frame or per-message callback** (audio
   chunk, camera frame, IMU, DDS handler, per-RPC). At 10–30 Hz these dominate
   the log on their own. Require the repo's throttle pattern: log the state
   *transition* unthrottled, then sample (`if n == 1 or n % 100 == 0`).
4. **Logging a value that could be `bytes` or attacker-controlled** — audio
   payloads, image buffers, DDS `response.data`, an HTTP request line. Log
   `len(...)` or escape and cap (`s.encode('unicode_escape').decode('ascii')[:200]`).
   With `network_mode: host` a request line is remote input going straight into
   the log framer.
5. **A new container or `docker run` without log rotation.** Every
   `deploy/service.yml` and compose fragment must carry
   `logging: {driver: local, options: {max-size: 10m, max-file: 3}}`. Note
   `json-file` with no options is **unbounded** — `local` at least defaults to
   20m x 5.

New long-running Python entry points, and every `multiprocessing` child entry
point, should install the atomic line writer (`logsafe.install()`); a spawned
child does not inherit the parent's `sys.stdout`. See
`phanthymotus/README.md` § Container Logs and
`phanthymotus-driver/README_dev.md` § Logging.

## Every DDS container carries the loopback profile

Applies to `agent-core`, `perception`, `actucore` and every driver — anything that
reaches Agent Core over FastDDS. The compose fragment (or
`agent-core/deploy/docker-compose.yml`) must have both:

```yaml
volumes:
  - /opt/phanthy-motus/dds-local.xml:/opt/phanthy-motus/dds-local.xml:ro
environment:
  - FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml
```

`ROS_DOMAIN_ID` is **42 everywhere**; there are no per-robot numbers to allocate.
`FASTDDS_BUILTIN_TRANSPORTS` must be gone — it is a process-wide switch that cannot
confine traffic to the local host, and the profile's `useBuiltinTransports=false`
overrides it anyway, so leaving it in only misleads the next reader.

This exists because `/remote_control/message` — a *command* topic — was reaching every
robot on the office LAN: an instruction typed on one robot was executed by a second one,
identical timestamp in both logs. DDS has no addressing and no authentication.

Three things to flag, none of which look like what they are:

1. **A missing mount does not merely leave the container unisolated — it cuts the
   container off.** With `useBuiltinTransports=false` everywhere else it ends up on a
   different transport from the rest of the machine and cannot reach Agent Core at all.
   The symptom is a device that registers over HTTP and shows in the dashboard while none
   of its topics carry data. If a PR describes that symptom, check the mount first.
2. **A mistyped mount path fails silently.** Docker creates a *directory* of that name,
   FastDDS falls back to every interface, nothing is logged. Compare the string exactly.
3. **Dropping `COPY <component>/deploy/ /deploy/` from a Dockerfile removes the profile
   too.** Without the fragment in the image, deployment degrades to the legacy
   `docker run` path, which carries no volumes from it. See the actucore rules.

For drivers, `phanthymotus-driver/scripts/check_service_yml.py` encodes this contract
plus its two exemption tables; `read_file` it rather than re-deriving the rules. Full
rationale: `CLAUDE.md` § "DDS is locked to the local host".

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
