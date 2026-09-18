# Review rules — Perception (`phanthymotus/perception`)

Authoritative reference: **`perception/README.md`**. It is short and almost
entirely about the ASR audio contract, which is the thing PRs break.

The same contract is restated from the driver side in
`phanthymotus-driver/README.md` §Audio Requirements for ASR Compatibility. If a
PR changes the contract, both documents need updating — check whether it did.

## The audio contract

ASR consumes `audio_msgs/AudioChunk`: **PCM 16 kHz mono S16LE**, chunks of at
least 1024 bytes.

Known failure modes to check for:

- A USB microphone delivering 48 kHz. It must be resampled to 16 kHz, not passed
  through — the symptom is recognition that silently produces nonsense rather
  than an error.
- Chunks smaller than 1024 bytes. These need buffering before publish; a driver
  that publishes per-callback without accumulating will trip this.
- Any change to sample rate, channel count or sample format is a **contract
  change**, not an implementation detail. It affects every driver that feeds ASR.

## Structure

- `main.py` — MCP server entry
- `plugins/` — `asr`, `tts`, `vop`, `ocr`, `kws`. Read a sibling plugin to
  learn the local convention before judging a new one.
- `config.yaml` — per-plugin enable/disable
- `utils/model_downloader.py` — the model manifest. **Models belong here, fetched
  from COS at runtime, not committed.** Eleven models are already listed; a new
  one should be added to this manifest in the same shape.
- `utils/model_progress.py` — the one place the download status line is worded.

## Model downloads

Authoritative: `perception/README.md` §"Model downloads: the two rules". If a PR
touches a download path — a new model, a new `ensure_*`, a plugin that fetches
weights, a changed manifest — check it against both rules and say which one it
misses.

**Rule 1 — pinned, and free to choose a source.** Every file carries `size` +
`sha256`, verified before acceptance. Flag: a new model added without pins; a
`check_file`-exists test used as verification (a truncated 780 MB transfer passes
it and then fails at session creation, undiagnosably); a hand-rolled loop over
sources instead of passing `base_url` as a list to `ensure_verified_bundle`, which
probes them and uses the fastest.

**Rule 2 — it must say how far along it is.** A download with no progress is
indistinguishable from a hang: one status line, and a cold fetch runs from seconds
to minutes. Flag any of these:

- an `ensure_*` call that omits `progress_cb` where the caller has a status
  channel — the parameter existing and not being passed is the usual shape of
  this bug, and it reads as wired when it is not;
- a new `ensure_*` front door that does not *accept* `progress_cb` (and `stage_cb`
  too, if it unpacks an archive) and forward it;
- the status line formatted by hand instead of via `utils/model_progress.fetch_status` —
  seven plugins render this and the wording must not drift;
- an archive path with no `stage_cb`: a percentage lies at the end of an archive
  download, and `100% (515/515 MB)` frozen while a tarball unpacks reads as a hang;
- **a callback wired to a status field nothing can read.** This is the one worth
  spending a round on, because the code looks complete: check the plugin's
  `info`/`state` can actually be answered *while the download is in flight*. Two
  real cases — TTS's `_loading` flag had no writer, so its "downloading" reply was
  unreachable code; the VLA card answered `idle` through a multi-gigabyte fetch
  because `_running` is only set afterwards.

Where a download genuinely cannot report progress (it runs in a child process
whose protocol is request/reply), the fix is to move the fetch, not to skip the
rule — see how face prefetches in the parent.

Perception ships `deploy/service.yml`, so it deploys the same way drivers do:
Agent Core extracts the fragment from the image and merges it into the host
compose file.

## One Dockerfile, Jetson only

`Dockerfile.jetson` is the only one — it builds from a prebuilt Jetson torch
image and downloads CLIP weights at build time. The CPU variant is gone: it
produced an image nobody deployed and had stopped building, so
`deploy/build_perception.sh` no longer takes `--variant`, only `--jp-version`
(5.11 / 6.1). A PR that reintroduces a CPU path needs to say who deploys it.

Build context is the **repo root**, so `COPY` paths inside the Dockerfile are
`perception/…`. A `COPY` written relative to `perception/` will fail the build.

Note `Dockerfile.jetson` hardcodes its registry rather than taking an `ARG`,
unlike every other Dockerfile in the project. Worth mentioning if a PR touches
that line anyway, not worth raising on its own.

It also does **not** `COPY perception/deploy/ /deploy/`, so the image ships
without the compose fragment and Agent Core silently falls back to the legacy
`docker run` path. That is a real bug — `actucore/Dockerfile.jetson` has the
correct `COPY`. Flag it if a PR is already editing the COPY block.
