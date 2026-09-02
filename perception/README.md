# Perception Stack

Modular ASR/TTS perception plugins running as an MCP HTTP server. Connects to Agent Core via MCP tool calls and exchanges audio/text over ROS2 DDS topics.

## Audio Requirements for ASR

The ASR plugin (VAD + speech recognition) has strict requirements on the audio stream it receives. Any mic driver that does not meet these requirements will produce no output.

### ROS2 Message Type

```
audio_msgs/AudioChunk
  std_msgs/Header header
  string format          # must be "audio/pcm-16k"
  uint8[] data           # raw PCM bytes (little-endian signed 16-bit)
```

### PCM Format

| Parameter | Required value |
|-----------|---------------|
| Encoding | 16-bit signed integer, little-endian (PCM_S16_LE) |
| Sample rate | **16 000 Hz** |
| Channels | **Mono (1 channel)** |
| `format` field | `"audio/pcm-16k"` |

### Chunk Size

| Parameter | Constraint |
|-----------|-----------|
| Minimum | **1 024 bytes** (512 samples, ~32 ms) |
| Recommended | 1 024 – 4 096 bytes (32 – 128 ms per chunk) |
| Maximum | No hard limit, but very large chunks increase latency |

Chunks smaller than 1 024 bytes are **silently discarded** by the VAD. This is the most common cause of "ASR receives audio but never outputs anything."

> **Why 512 samples?** The Silero VAD model requires at least one 512-sample window to compute a speech probability. WebRTC VAD requires 480-sample (30 ms) frames. Both backends use 512 samples as the minimum chunk size.

### Common Pitfalls

#### External USB mic (ALSA, 48 kHz native rate)

Most USB audio interfaces run at 48 000 Hz. After downsampling to 16 000 Hz, a 512-frame ALSA period becomes only **170 samples (340 bytes)** — below the VAD minimum.

**Fix (already applied in `phanthymotus-driver`):** Buffer resampled output until 512 samples are accumulated before publishing each `AudioChunk`.

If you are writing a custom mic driver, apply the same buffering pattern:

```python
TARGET = 1024  # bytes (512 int16 samples)
_buf = bytearray()

# Inside your capture loop, after resampling:
_buf += resampled_bytes
while len(_buf) >= TARGET:
    chunk, _buf = bytes(_buf[:TARGET]), _buf[TARGET:]
    publish(chunk)
```

#### Native G1 robot mic (UDP multicast)

Publishes raw 16 kHz PCM at 1 024 bytes per chunk. No resampling or buffering needed.

---

## VAD Tuning

The VAD parameters can be adjusted per ASR canvas card via the instance config (⚙ button):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `vad_threshold` | `0.5` | Speech probability threshold (0–1). Raise to `0.7`–`0.85` in noisy environments (e.g. robot motor noise). |
| `vad_silence_ms` | `400` | Silence duration (ms) required before an utterance is considered complete. |
| `vad_pre_roll_ms` | `500` | Audio retained from *before* the VAD tripped. Recovers clipped word onsets — without it the first syllable is often missing, which costs wake-word recall. |

---

## sherpa-onnx Device Selection

`device: cpu | gpu` (under `plugins.asr` and `plugins.tts` in `config.yaml`, and on
the dashboard's config form) selects where sherpa-onnx runs. It defaults to `cpu`.

**The model follows the device, not the other way round.** `ASR_MODELS` in
`plugins/asr.py` maps each (model, device) pair to the weights that pair loads,
because the best weights differ per device: quantised weights are right on the CPU
and wrong on the GPU. `device: gpu` therefore downloads a different bundle, not
just a different provider string.

| `asr_model` | `device: cpu` | `device: gpu` | gpu speed-up |
|-------------|---------------|---------------|--------------|
| `sensevoice-small` (default) | int8, 228 MB | **fp16, 448 MB** | **3.4x** per utterance ⚠️ |
| `paraformer-zh-en` (streaming) | int8, 226 MB | **fp32, 825 MB** | **1.77x** |
| `x-asr-zh-en` | int8 + fp32 | — not offered | 0.80x, i.e. slower |
| `paraformer-offline` | int8 | — not offered | unmeasured |
| `zipformer-en` | int8 | — not offered | unmeasured |

⚠️ **`sensevoice-small` on gpu drops some utterances entirely** — fp16 under the
CUDA provider returns an empty transcript for certain inputs, silently and
reproducibly, on both JetPack lines. Read § jp6.1, and a silent failure the gpu
path has always had before enabling it; the speed-up is real but so is the loss.

**gpu costs about 2 GB of RAM, and ~1.4 GB of that is unreturnable.** Measured with
only ASR resident: the cpu adapter adds 542 MB and drops back to 129 MB when
released; the gpu adapter adds 1968 MB and still holds 1516 MB after release,
because a process that has touched CUDA does not give its context and memory pool
back. On a 7.4 GB Orin already running vop (YOLO), OCR (TensorRT) and TTS, turning
on gpu ASR was enough to exhaust memory: perception was restarted in a loop, Agent
Core could not reach port 15720, and the dashboard rolled the project back and the
cards vanished. Budget for it before enabling.

TTS is simpler: Matcha is fp32 only, so both devices load the same files and
`device` only picks the provider (gpu measured ~4.3x). The `vits2_trt` engine
ignores `device` entirely — it is a TensorRT engine and never touches ONNX Runtime.

`device: gpu` also needs a CUDA sherpa-onnx wheel. Both Jetson images install one —
jp5.11 and jp6.1 each have their own build, because the wheel is tied to a
CUDA/cuDNN pair and a CPython ABI (see § The CUDA wheels). On x86 dev hosts, and on
any JetPack line we have not built a wheel for, it falls back to `cpu` with a
warning rather than failing to start.

### Latency per utterance, which is what an operator feels

Inference is not the latency. Captured from the plugin's own spans on a live
`device: gpu` instance, two consecutive wake-word utterances:

| span | utterance 1 | utterance 2 |
|------|-------------|-------------|
| `audio_end` → `asr_complete` (**felt latency**) | **7150 ms** | **10701 ms** |
| `asr_transcribe` | 120 ms | 91 ms |
| `kws_phonemize` | **5256 ms** | **5234 ms** |
| `kws_match` | 0 ms | 0 ms |
| unaccounted (queue wait + cutting the wake word off) | 1775 ms | 5376 ms |

Transcription was 91–120 ms. Almost all of it was `kws_phonemize` doing nothing
useful — see § asr_kws and espeak below — and the growing unaccounted figure is the
utterance queue backing up behind it, because the VAD emits a segment every 2–3 s
while each one took ~5.5 s to process. Latency accumulated over a session rather
than being constant, which is why it felt worse the longer you talked.

**Measure this from the spans, not from a `docker exec` one-liner.** The same
`_text_to_ipa` call cost 320 ms in a small test process and 5256 ms inside
perception, because the failure path forks `ldconfig` and forking a 3.6 GB
many-threaded process is expensive. Out-of-process timing hid the entire problem.

Inference alone, for comparison — real VAD segments (1–4 s), one at a time, rest of
perception running:

| | cpu (int8) | gpu (fp16) |
|---|---|---|
| first call, cold process | 165 ms | 1659–2269 ms |
| sustained p50 (40 calls) | 195 ms | **58 ms** |
| sustained p99 | 326 ms | **78 ms** |
| after 60 s idle | 159 ms | 73 ms |

No drift across 40 calls (both halves at a 57.7 ms p50) and no idle cliff. The gpu
p99 beats the cpu median, because vop and OCR contend for cores but not for the GPU.

`_warmup_adapter()` decodes a second of silence after building to keep CUDA's
first-inference cost off the first real utterance. Be aware it does not fully
absorb it: a cold process still measured 2269 ms on its first *real* segment after
warming on silence, so the warmup does not cover every input shape. It is worth
keeping — silence costs ~1.7 s inside the `loading` window either way — but the
first utterance after a model switch can still be slow. (An earlier measurement
that suggested warmup reduced this to 82 ms was an artifact of test ordering: an
unwarmed adapter earlier in the same process had already paid the CUDA init.)

The batch figures in § Measurements are larger (up to 23x) because they decode the
model's own `test_wavs`, which are 7 s each. Those compare dtypes with each other;
this table is what a user experiences.

### jp6.1, and a silent failure the gpu path has always had

Same measurement on orin6 (Orin NX, JetPack 6.1, CUDA 12.6, onnxruntime-gpu
1.18.1, 6 cores), SenseVoice, `num_threads=2`, through `_build_asr_adapter` so the
registry and provider selection are the production ones. `provider_for_device('gpu',
fp16)` returns `'cuda'`, and 120 calls per device:

| audio | cpu (int8) p50 / p99 | gpu (fp16) p50 / p99 | speed-up |
|---|---|---|---|
| rig's own VAD captures, 0.8–2.1 s | 119 / 194 ms | **49 / 58 ms** | 2.4x / 3.3x |
| KWS bundle test_wavs, 4.5–16.7 s | 462 / 1305 ms | **67 / 169 ms** | 6.9x / 7.7x |

The gpu p99 is below the cpu *minimum* in both rows, which is the useful sanity
check that CUDA is actually doing the work. Longer audio wins more, consistent with
the jp5.11 numbers. Building the gpu adapter took 3.7 s with weights already
local, 86.3 s including the 449 MB fp16 download. Whole-box `MemAvailable` fell
~605 MB while gpu was resident — well short of the ~2 GB the jp5.11 table reports,
so budget from a measurement on the box you are deploying to rather than from
either figure.

**But `device: gpu` can lose an utterance outright.** SenseVoice fp16 under the
CUDA provider returns an **empty** transcript for some inputs — deterministically,
5/5 attempts, on the KWS bundle's own `en_0.wav`: 6.6 s of clear English peaking at
0.535 FS, louder than the `en_1.wav` that decodes fine. Eight of the nine files in
that bundle match cpu exactly.

Isolated by varying one thing at a time, since dtype and provider normally change
together:

| | en_0.wav |
|---|---|
| `model.int8.onnx` + cpu | correct |
| `model.fp16.onnx` + cpu | correct |
| `model.fp16.onnx` + cuda | **empty** |

So the fp16 export is fine and the CUDA provider is at fault. Reproduced on **both**
lines — onnxruntime-gpu 1.16.0 on orin5 and 1.18.1 on orin6, byte-identical
`model.fp16.onnx` — so it is not a property of either wheel and not new. An earlier
note in `ASR_MODELS` claimed fp16 was "transcript-identical to fp32 on both
providers"; that was wrong, and the failure is silent. An empty transcript is
indistinguishable from silence, so the utterance is dropped with nothing in the
log to say so. Untested alternative: SenseVoice fp32 on CUDA, which measured
11.52x on jp5.11 and would still beat cpu — no fp32 bundle is published, so
switching the registry to it means building one first.

### Switching engine or device blocks for the bounded part

`action=config` on the `tts` tool waits up to `ENGINE_SWITCH_WAIT_S` (20 s) for the
new engine before answering `loading`. Only some of a build is open-ended — a cold
model download — while constructing the session afterwards took ~2 s on cpu and ~5 s
on gpu. Reporting `loading` for that made the dashboard send its `start` into a
facade with no engine, and the start was dropped: the engine then came up idle, and
Agent Core's loading watcher (`api/config.py` `_settle_loading_item`) reports "启动已
取消" the moment it sees `idle`.

Waiting is safe on that path — `mcp_call_tool` sets no client timeout and the
watcher polls for up to 900 s. (The 60 s often quoted in these plugins belongs to
the *LLM* tool path in `agent-core/src/mcp_client.py`, which does not send `config`.)
A build that outlives the bound still goes async, and `TTSPlugin` records any
`start` that arrives mid-build and replays it once the engine is resident, so the
card reaches `running` instead of being cancelled. A `stop` cancels a pending
replay. The bound is a time limit rather than a "does it need to download?" check
because the download is not the only slow phase: the gpu paraformer encoder is
636 MB and reading it cold takes seconds by itself.

### Measurements

orin5 (Orin NX, JetPack 5.11, CUDA 11.4, 6 cores), sherpa-onnx 1.13.6+cuda,
steady-state median, **on an idle box** — `embodied-perception` and `agent-core`
stopped, idle `GR3D_FREQ` verified 0% before each run. Both providers swept across
1/2/4 `num_threads`, because the ratio moves a lot with thread count:

| model | dtype | CPU t=2 | CUDA t=2 | ratio |
|-------|-------|---------|----------|-------|
| streaming paraformer (30.7 s audio) | int8 | 3295 ms | 8394 ms | **0.39x** |
| streaming paraformer | fp32 | 8702 ms | **1859 ms** | **4.68x** |
| streaming paraformer | fp16 | 42890 ms | 2077 ms | 20.65x ⚠️ broken, see below |
| X-ASR, beam search (28.7 s audio) | int8 | 2645 ms | 3294 ms | **0.80x** |
| offline SenseVoice (28.7 s audio) | int8 | 1996 ms | 2753 ms | **0.73x** |
| offline SenseVoice | fp32 | 4792 ms | 416 ms | **11.52x** |
| offline SenseVoice | fp16 | 8117 ms | **344 ms** | **23.60x** |
| Matcha TTS (13.3 s audio) | fp32 | 1784 ms | 416 ms | **4.28x** |

`num_threads: 2` is what `config.yaml` deploys. Threads matter on the CPU (4
threads buys roughly 1.2–1.5x) and not at all on the GPU for non-quantised weights.

### Why int8 loses on the GPU

ONNX Runtime's CUDA execution provider has no kernels for the quantised ops in an
int8 model. It partitions the graph and falls back to CPU node by node, inserting a
host↔device copy at every boundary.

Two independent confirmations, not just the int8/fp32 correlation:

- fp32 and fp16 on CUDA are **completely insensitive to CPU thread count**
  (417/416/415 and 343/344/346 ms at 1/2/4) with GR3D pinned at 95–97% — the graph
  really is on the GPU. int8 on CUDA instead **scales with CPU threads**
  (3793 → 2753 → 1905 ms for SenseVoice, 17861 → 11125 → 10524 for streaming
  paraformer), which only makes sense if much of it is executing on the CPU.
- Requantising at every partition boundary perturbs the output. Same model, same
  audio, `num_threads=2`, CPU vs CUDA: SenseVoice int8 differed on **3 of 4** clips,
  including `不然` → `主然` — a real word error, not punctuation. SenseVoice fp32 and
  fp16 differed on **0 of 4**.

GPU contention is not an alternative explanation: the idle `GR3D_FREQ` baseline was
0% for every run above. That is not a hypothetical — an earlier X-ASR run taken
while another container was building TensorRT engines read 0.50x instead of 0.80x.

`int16` is not a middle ground worth trying: ONNX Runtime's int16 quantisation
(`QInt16`/`QUInt16`, opset 21) is newer than int8, the CUDA provider has no kernels
for it either, and the CPU side lacks the dot-product paths that make int8 fast.

### Admitting a new (model, device) pair

The `gpu` column above is short because each entry had to earn its place. **A pair
is only added to `ASR_MODELS` after decoding real audio on the target device and
reading the text.**

That rule exists because of one result. Streaming paraformer fp16 on CUDA:

- created an ONNX Runtime session without complaint,
- ran in 2077 ms, 20x faster than the same file on CPU,
- produced byte-identical output across all three thread counts,
- and emitted nothing but `</s> </s> </s> …`.

The same fp16 file on CPU transcribed correctly, so the conversion was fine and the
CUDA+fp16+streaming *combination* is not. Session creation, speed, and
self-consistency were all green. Only reading the text caught it. (fp16 is also
slower than fp32 for that model, so there was nothing to gain by debugging it —
`paraformer-zh-en`'s gpu entry is fp32.)

Checklist:

1. Benchmark both devices across 1/2/4 threads on an idle box, and verify the idle
   `GR3D_FREQ` is 0% first.
2. Decode real audio on the target device and read the transcripts. Compare against
   the same weights on the other provider, and against the cpu entry.
3. Add the entry to `ASR_MODELS` with its `dtype`, and add the pinned bundle to
   `SHERPA_GPU_BUNDLES` in `utils/model_downloader.py`.
4. Add the model to the `device` field's `x-show-when` list in the configSchema.
   `tests/test_asr_device_registry.py` asserts that list matches the registry, that
   no gpu entry is int8, and that no cpu entry is fp16.

### Producing fp16 weights

`tools/convert_onnx_fp16.py` converts an fp32 sherpa-onnx model. Two flags are
load-bearing:

- `keep_io_types=True` — sherpa hands the session fp32 features, so only the graph
  interior may be fp16.
- shape inference must stay **on**. Ops that cannot take fp16 (`Range` above all)
  are already in onnxconverter-common's default `op_block_list` and get fenced with
  Cast nodes, but placing those Casts needs shape inference. Disabling it produces
  a file that saves fine and then fails at session creation with
  `Type 'tensor(float16)' of input parameter (…) of operator (Range) … is invalid`.

fp16 is a **GPU-only** choice: ONNX Runtime has no fp16 CPU kernels and casts
everything, which is why the streaming fp16 CPU row above is 42890 ms against
int8's 3295 ms. `provider_for_device()` logs an error if a cpu entry ever points at
fp16 weights, and the registry test rejects it.

### What does not follow `device`

- **KWS** is pinned to CPU. Its zipformer bundle ships int8 only, and int8 on CUDA
  lost on every model measured, so there is nothing to gain.
- **VAD** is pinned to CPU in both code paths (`_vad_worker` and
  `_vad_segment_sync`). silero infers one 512-sample window at a time — too little
  work to amortise a kernel launch plus two copies per 32 ms of audio — and in
  `_vad_worker` it would hold a second CUDA context in a child process.
- **`vits2_trt` TTS**, as above: TensorRT, not ONNX Runtime.

### GPU bundle distribution

`device: gpu` weights come from `SHERPA_GPU_BUNDLES` in `utils/model_downloader.py`
and are fetched with `ensure_verified_bundle`, which pins every file's size and
SHA256. The cpu bundles use `ensure_model`, whose only integrity check is "does
`check_file` exist in the archive" — acceptable for a 230 MB archive, not for a
780 MB one, where a truncated transfer would pass and then fail confusingly at
session creation.

Provenance: the fp32 weights come from `pengzhendong`'s ModelScope mirrors of the
k2-fsa model zoo, accepted only after that mirror's int8 files were confirmed
**byte-identical** (SHA256) to the copies we already deploy from COS. The fp16 files
are converted from those with `tools/convert_onnx_fp16.py`.

---

### The CUDA wheels

PyPI ships CPU-only `sherpa-onnx`, so `Dockerfile.jetson` downloads a wheel built
in-house from COS under `public/sherpa-onnx/<jp>/`, one per JetPack line, and falls
back to the PyPI CPU wheel for any `JP_VERSION` without one:

| | onnxruntime-gpu | CUDA / cuDNN | COS key |
|---|---|---|---|
| jp5.11 (focal, L4T R35) | 1.16.0 | 11.4 / 8 | `jp511/sherpa_onnx-<ver>+cuda-cp38-cp38-linux_aarch64.whl` |
| jp6.1 (jammy, L4T R36) | 1.18.1 | 12.6 / 9 | `jp61/sherpa_onnx-<ver>+cuda-cp310-cp310-linux_aarch64.whl` |

Neither the ONNX Runtime pairing nor the CPython ABI is portable, which is why
there are two wheels rather than one. `cmake/onnxruntime-linux-aarch64-gpu.cmake`
in sherpa-onnx pins the URL and SHA256 per version and names the target board for
each; 1.18.1 is the one it lists for L4T R36 + CUDA 12.6.

**The directory carries the JetPack, because the filename cannot.** Upstream's
`setup.py` tags the wheel `<ver>+cuda-cp<abi>`, so in a flat directory the only
thing separating the two builds is `cp38` vs `cp310` — a *Python* discriminator,
not a CUDA one. It selects correctly today, since each image ships exactly one
Python and pip refuses a wheel built for another, but a second cp310 build for
another JetPack 6.x on a different CUDA would collide. Renaming the file is not an
option: pip parses the version out of it and it has to match the wheel metadata.

**Check the cuDNN soname before committing to a build.** ONNX Runtime's aarch64 GPU
packages do not encode it in the filename past 1.18.0, and it is what decides
whether the provider loads at all. Two minutes with `readelf` beats an hour of
compiling:

```bash
readelf -d libonnxruntime_providers_cuda.so | grep NEEDED
```

1.18.1 needs `libcudnn.so.9` / `libcudart.so.12` / `libcublas.so.12`, all of which
the jp6.1 image resolves (cuDNN 9.4.0, CUDA 12.6). 1.18.0 would not — it is built
against cuDNN 8.9.4, which that image does not carry.

To build one, build **inside a container started from the perception image** for
the target JetPack — that image already carries cmake, g++, the matching CPython
headers and CUDA, so the pybind extension lands on the right CPython ABI and
glibc. Building on the host instead is what produces an unusable wheel: the
extension is tagged `cp<major><minor>` and will not import under a different
Python.

```bash
# on the build host (must match the target JetPack: jp5.11 → L4T R35, jp6.1 → R36)
docker run -d --name sherpa-build --runtime nvidia \
  -v /path/to/k2-fsa/sherpa-onnx:/src:ro -v /path/to/outdir:/out \
  --entrypoint bash <perception-image-for-that-jp> /out/build.sh

# inside, against a *writable* copy of the tree (setup.py appends __version__ to it).
# jp6.1 shown; for jp5.11 use 1.16.0 and python3.8.
export SHERPA_ONNX_ENABLE_GPU=ON
export SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.18.1   # jp6.1 / CUDA 12.6
export SHERPA_ONNX_MAKE_ARGS="-j2"                              # Orin has 6 cores but ~3 GB free
SHERPA_ONNX_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=ON \
  -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.18.1 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
  -DPython_EXECUTABLE=/usr/bin/python3.10" \
  python3.10 setup.py bdist_wheel
```

Notes:

- `setup.py bdist_wheel` builds its **own** cmake tree at
  `build/temp.linux-aarch64-cpython-<ver>/` with `SHERPA_ONNX_ENABLE_PYTHON=ON`. A
  tree left behind by `build-aarch64-linux-gnu.sh` has `ENABLE_PYTHON=OFF` and is
  not reusable — expect a full compile.
- To avoid re-downloading onnxruntime, drop
  `onnxruntime-linux-aarch64-gpu-<ver>.tar.bz2` in `/tmp/`;
  `cmake/onnxruntime-linux-aarch64-gpu.cmake` checks there before GitHub. Use that
  exact generic name even when the release asset is called something else — the
  hash it checks is the one for the asset it would have downloaded.
- Upload the result to `public/sherpa-onnx/<jp>/` (anonymous read, same `public/`
  prefix `utils/model_downloader.py` uses) and bump the matching
  `SHERPA_GPU_WHEEL_<jp>` in `Dockerfile.jetson` — the ARG holds the key *with* its
  directory, and the build downloads it to `/tmp/$(basename …)`. COS credentials
  live in `resource-center/deploy/values.env` (`COS_SECRET_ID` / `COS_SECRET_KEY`);
  the bucket is `agi-phanthy-dev-1252788780` in `ap-beijing`.

---

### `libnvdla_compiler.so` — the jp6.1 image carries its own

On JetPack 6, `import tensorrt` has `libnvdla_compiler.so` as a hard `DT_NEEDED`,
and no image can ship it at its real path: `/usr/lib/aarch64-linux-gnu/nvidia/` is
owned by nvidia-container-runtime, which bind-mounts the host's copies over the
zero-length placeholders the L4T base ships. A host BSP without
`nvidia-l4t-dla-compiler` has nothing to mount, the placeholder stays 0 bytes, and
every TensorRT plugin is down at once — VITS2 TTS, OCR, obstacle:

```
ImportError: libnvdla_compiler.so: cannot open shared object file
  → utils.tensorrt_runtime.TensorRTError: TensorRT is not available in this runtime
```

Bumi's Orin NX ships exactly that BSP (L4T R36.5.0, vendor-flashed): `drivers.csv`
line 85 names the library, `dpkg -l | grep dla` finds nothing, and
`find / -name 'libnvdla_compiler*'` on the host comes back empty. `torch.cuda` is
fine on that host, so the symptom is TTS-only and reads like a TTS regression.

So `Dockerfile.jetson` downloads a copy from COS
(`public/nvidia/jp61/libnvdla_compiler.so`, taken from `nvidia-l4t-dla-compiler
36.4.3`) into `/opt/nvidia/dla-fallback/` — **a path the CSV does not name**, or it
would be just another placeholder for the runtime to leave empty. `/etc/dla-fallback.env`,
sourced by `CMD`, appends that directory to `LD_LIBRARY_PATH` only when both hold:

| test | why |
|---|---|
| `-s .../nvidia/libcuda.so.1` | the nvidia runtime really took over. Under plain runc every file there is a 0-byte placeholder and the container has no GPU at all — masking the linker error would trade one clear failure at import for a baffling one later at engine build. |
| `! -s .../nvidia/libnvdla_compiler.so` | the host has no copy to mount. When it does have one, this is false and TensorRT uses it: the host's matches its kernel and DLA hardware, ours does not. |

Both test **size**, not existence — the placeholder always exists, so `-e` is true
even when nothing usable is behind it. And the directory is *appended*, never
prepended: `LD_LIBRARY_PATH` is searched before `ld.so.cache`, so going first would
shadow a correctly mounted host library on every other robot.

DLA is never used — our engines run on the GPU and this only satisfies the linker,
which is why an R36.4.3 copy works on an R36.5.0 host. Verified on Bumi (fallback
fires, `build_serialized_network` + `deserialize_cuda_engine` both succeed), on Bumi
under `--runtime=runc` (guard stays silent, import still fails loudly), and on orin6
where the host has the library (guard silent, `/proc/self/maps` shows the host copy
loaded, `LD_LIBRARY_PATH` untouched).

jp5.11 needs none of this: L4T R35 resolves both nvdla libraries out of
`/usr/lib/aarch64-linux-gnu/tegra/`, which its own base image populates, so
`DLA_COMPILER_SO` is empty for that line and the build step is a no-op.

**Corollary for debugging:** a `libnvdla_compiler.so` error is no longer proof that
`runtime: nvidia` is missing from the compose fragment — but with the fallback in
place it is again the *only* remaining cause, and the log line
`[dla-fallback] host BSP has no libnvdla_compiler.so` tells you which case you are in.

---

## asr_kws and espeak

`trigger_mode: asr_kws` transcribes every utterance and gates on a phoneme-level
fuzzy match against the wake word, so it needs IPA for both. That path had two
faults that together cost **5.2 s per utterance** and quietly degraded wake-word
accuracy.

**phonemizer could not find libespeak-ng.** It looks the library up with
`ctypes.util.find_library('espeak-ng')`, which on Linux shells out to `ldconfig -p`.
`Dockerfile.jetson` replaces `ldconfig` with a no-op during apt installs so the
libc-bin trigger cannot segfault under qemu, and the cache is never rebuilt — so the
package is installed (`/usr/bin/espeak-ng`, `libespeak-ng.so.1`, three debs) while
`find_library` returns `None` and `EspeakBackend` raises "espeak not installed".
`_text_to_ipa` caught that and fell back to comparing raw *characters*, which is
also why 「小康小康」 failed to match 「小范小范」: at character level 康≠范 is a full
mismatch, where the phonemes `kʰ ɑŋ` vs `f a n` still score a usable distance.

Fixed by pointing at the library directly, both as an `ENV` in `Dockerfile.jetson`
and as a runtime probe in `_point_phonemizer_at_espeak()` so an already-built image
recovers too. `PHONEMIZER_ESPEAK_LIBRARY` set by the deployment always wins.

**And every failure was retried.** `_text_to_ipa` splits an utterance into CJK /
non-CJK segments — four for 「小范小范，你好。」, since fullwidth punctuation is outside
`一-鿿` — and `_phonemize_safe` rebuilt the backend and retried on each one.
Eight `find_library` calls per utterance, each forking `ldconfig` from a 3.6 GB
many-threaded process. A construction failure is now remembered and never retried;
the retry-on-crash path remains for a backend that worked once and then died.

Measured in a deliberately large, threaded process:

| | first call | subsequent | phonemes? |
|---|---|---|---|
| library located | 511 ms | **0.3–0.5 ms** | real IPA |
| library missing | 385 ms | 0.0 ms (negative cache) | characters |

The persistent-backend cache the code always claimed to have only starts working
once construction succeeds — before this, the dict stayed empty and every segment
rebuilt.

**Cutting the wake word off the transcript used to guess.** The match returns a
*phoneme* index, but what gets forwarded to the agent is *text*, so the two have to
be related. The old code estimated: it re-phonemized the segment containing the
match and scaled, `round(offset_in_seg * chars_in_seg / seg_ipa_count)`. Chinese
characters are not a fixed number of phonemes each — 「小潘小潘，现在发生什么了？」
phonemizes to 29 phonemes over 12 characters and the wake word ends at phoneme 12,
where the estimate gives `round(12 * 11 / 29) = 5` and eats 现, so the agent was
asked 「在发生什么了？」. Off-by-one on a syllable also silently changes the utterance
in mixed script, where latin and CJK have very different phonemes-per-character and
one ratio was applied to both.

`_text_to_ipa(text, with_positions=True)` now also returns, per phoneme, the
character offset in the original string that phoneme ends at — built from growing
prefixes of each segment, phonemized through the same function that produced the
phonemes. Segments are `(start, end, is_cjk)` offsets rather than `strip()`ed
substrings, so punctuation and whitespace stay accounted for. `_text_after_phoneme`
is then a lookup, which also removes the second phonemization pass. Positions are
opt-in: callers that only need to match pay nothing for the prefix passes.

---

## Plugin Concurrency

**`dispatch()` is not single-threaded.** `main.py` serves MCP over
`ThreadingHTTPServer`, so every `tools/call` runs on its own thread. `start`,
`stop`, `config`, and `speak` on the *same* plugin can genuinely run at once —
the canvas does exactly this (config → start, then stop, then config → start).

This has already caused a production incident, so the rules below are not
theoretical.

### The failure mode

Any plugin that keeps per-instance state in a dict is exposed to this shape:

```python
# ❌ WRONG — check-then-act with no lock
node_key = instance_id or input_topic
if node_key not in self._nodes:          # ← two threads both pass here
    node = _ASRNode(...)
    self._executor.add_node(node)
    self._nodes[node_key] = node         # ← only the last one survives
return self._nodes[node_key].start()
```

Both threads build a node with the *same* ROS node name, both add it to the
executor, and the dict keeps only the second. The first is now an **orphan**: its
subscription, its VAD subprocess, and its transcription thread are all still
running and still publishing to the same output topic, but it is not in
`self._nodes`, so `stop` can never reach it. It survives until the process exits.

Observable symptoms: every utterance recognised and published twice, duplicate
files in `/models/vad_segments` with byte-identical content, an extra
`vad_worker` child process that `stop` does not reap, and this from rclpy:

```
Publisher already registered for provided node name. If this is due to multiple
nodes with the same name then all logs for that logger name will go out over the
existing publisher.
```

### The rules

**1. Make the dict access atomic.** One `threading.RLock` per plugin, guarding
every read-modify-write of the state dict:

```python
# ✅ CORRECT — atomic get-or-create
with self._nodes_lock:
    node = self._nodes.get(node_key)
    if node is None:
        node = _ASRNode(...)
        try:
            self._executor.add_node(node)
        except Exception:
            node.destroy_node()          # don't leak a half-registered node
            raise
        self._nodes[node_key] = node
    else:
        self._sync_cfg(node)
```

**2. Never hold that lock across `node.start()`, `node.stop()`, or a model
load.** `_ASRNode.start()` blocks for up to 15 s waiting for the first audio
chunk. If `stop` is queued behind the lock for those 15 s, it cannot set the
cancellation flag in time, `start` sails through to `running`, and you are left
with a pipeline nobody asked for. Register the node inside the lock, then release
it and call `start()` outside.

**3. Register the node *before* starting it.** That is what lets a concurrent
`stop` find it and cancel the in-flight start. Loading a model or otherwise
blocking *before* the node is in the dict means `stop` finds nothing, returns
`{"state": "idle"}`, and silently no-ops — while the start it was meant to cancel
completes anyway.

**4. `stop` signals first, locks second.** Give the node a non-blocking
`request_stop()` that sets its cancellation events, call that before taking any
lock, and only then tear down:

```python
def stop(self) -> dict:
    self.request_stop()                  # non-blocking; unblocks an in-flight start
    with self._lifecycle_lock:
        self._teardown()
        self.state = "idle"
        return {"state": "idle"}
```

**5. Guard the node object too, and treat "starting" as taken.** A per-node
`RLock` plus `if self.state in ("running", "starting")` — otherwise two threads
that resolve to the *same* node object can both enter `_start_inner()` and build
two subscriptions and two subprocesses on one node.

**6. `destroy_node()`, not just `remove_node()`.** `remove_node` detaches the
node from the executor; it does not release the rclpy node, its publishers, or
its node name. Skip it and every start/stop cycle leaks a topic endpoint, and a
later start on the same key collides with the still-registered ghost:

```python
def _dispose_node(self, node, key=""):
    node.stop()
    self._executor.remove_node(node)
    node.destroy_node()                  # ← required
```

**7. Snapshot before iterating.** `info` is a heartbeat probe called constantly.
Iterating the live dict can raise `RuntimeError: dictionary changed size during
iteration` in the middle of a start. Copy under the lock, then iterate the copy.

### Where this applies

Every MCP server in the project uses `ThreadingHTTPServer` — `perception/main.py`
and each robot driver's `main.py`. Any plugin holding a `self._nodes` /
`self._instances` / `self._streams` dict needs the treatment above.

### The other way to orphan a node: nobody sends `stop`

The same symptoms — two nodes publishing the same kind of output, one of them on a
topic no card accounts for, and rclpy's duplicate-node-name warning — also appear
when the plugin is correct and the `stop` simply never arrives.

Agent Core's `stop-project` walks the cards in the *saved canvas layout*. A card
deleted from the canvas is no longer in that list, so from that moment nothing can
reach its instance again: it keeps its ROS node, its subscription, its subprocess
and (on `device: gpu`) ~1.4 GB of CUDA context until perception exits. On orin5 a
deleted TTS card left `vits2_trt_card_mt4rkb752py8` publishing to the topic of a
connection that had already been deleted, while the live card published elsewhere —
the dashboard read "running", and there was no sound.

`api/canvas.py` and `api/solutions.py` now diff the card set on every layout write
and `stop` whatever left it (`api/config.py: stop_removed_cards`). So when you see
an orphan, check both sides: the plugin's locking *and* whether a `stop` for that
instance_id ever showed up in the log.

---

## Topic Naming

| Direction | Topic pattern | Format |
|-----------|--------------|--------|
| Input (mic) | `/{namespace}/mic/audio` or `/{namespace}/ext_mic/{id}/audio` | `audio/pcm-16k` |
| Output (ASR result) | `{input_topic}/asr` | `data/json` |

ASR result JSON:
```json
{
  "text": "recognized speech text",
  "audio_start_ts": 1234567890.123,
  "audio_end_ts":   1234567891.456,
  "asr_complete_ts": 1234567891.789
}
```

## Sensitive Config Fields

Perception plugins hold real credentials (ASR/TTS API keys). Canvas configuration
gets packaged into shareable **Solutions** and uploaded to the Resource Center,
so every credential field must declare itself sensitive in its `configSchema` —
packaging blanks declared fields only, there is no field-name blocklist:

```python
"configSchema": {
    "type": "object",
    "properties": {
        "api_key": {"type": "string", "format": "password"},   # masked input + never packaged
        "app_key": {"type": "string", "x-sensitive": True},    # visible input + never packaged
        "model":   {"type": "string"},                         # packaged as-is
    },
}
```

An unmarked credential is uploaded in clear text and readable by anyone who
downloads the solution. Full spec: `phanthymotus-driver/README_dev.md`
§ "Marking sensitive fields".

---

## Running the tests inside a perception image

`python3 -m pytest tests -q` from a checkout works anywhere. Running the same
suite **inside a perception container** — which is how you confirm a fix behaves on
the Python the device actually ships — needs one flag:

```bash
docker cp tests <container>:/work/
docker exec <container> bash -c 'source /opt/ros/humble/install/setup.bash && \
  source /ros_ws/install/setup.bash && cd /work && \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q -p pytest_mock'
```

Without `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, **`caplog` captures nothing** and every
test that asserts on a log record fails while the log line is plainly there in the
captured stderr. The image inherits ROS 2's own pytest plugins from the base
(`launch_testing`, `launch_testing_ros`, six `ament_*`, `colcon-core`) alongside
`pytest-cov 3.0.0` / `pytest-timeout 2.1.0`, and something in that set breaks
`_pytest.logging`'s capture handler under pytest 8. Verified by reduction: a
two-line test logging to a plain `logging.getLogger("probe")` fails the same way,
and passes the moment autoload is off. It is not the tests, and not a Python 3.10
difference.

Measured on jp6.1 (Python 3.10.12, pytest 8.3.3): 175 passed / 1 failed with
autoload on, **176 passed** with it off.
