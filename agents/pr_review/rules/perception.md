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
