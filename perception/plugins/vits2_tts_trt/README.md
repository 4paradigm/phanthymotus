# VITS2 TensorRT TTS

Chinese-English VITS2 speech synthesis (16 kHz, single speaker) for Jetson
Orin. It is one implementation of the standard `tts` plugin, not a second MCP
plugin — `plugins/tts.py` owns the tool contract and picks the engine.

## Build

`ENABLE_VITS2_TRT` defaults to 1, so both JetPack lines build with the VITS2
frontend installed:

```bash
cd phanthymotus/deploy
./build_perception.sh --jp-version 6.1     # TensorRT 10
./build_perception.sh --jp-version 5.11    # TensorRT 8
```

The image installs the WeText/Kaldifst runtime; Chinese text normalization
executes the pre-compiled FSTs shipped in the model release, so neither
OpenFST nor Pynini is built on the device. Model files are not embedded in the
image.

numpy is pinned once, in `Dockerfile.jetson`, to the version each JetPack's
torch/cv2/rapidocr stack was built against (jp61 1.26.4, jp511 1.24.4) and that
same pin is passed to this package's install — `requirements.jetson.txt`
deliberately does not list numpy. See the comment there before adding a
dependency that wants a different one.

### JetPack 5.1.1 / TensorRT 8

Both lines build from the same source; `ENABLE_VITS2_TRT` defaults to 1. What
differs on jp511, and what to check when bringing a device up:

- **Python 3.8.** The frontend needs `importlib.resources.files`, which arrived
  in 3.9 — `requirements.jetson.txt` installs the `importlib_resources`
  backport under a marker and `frontend/wetext_compat.py` patches it in before
  WeText loads. Nothing else in the package uses 3.9+ syntax.
- **One source build.** Every dependency resolves to a cp38 aarch64 wheel
  except `pyahocorasick` (wetext → contractions → textsearch), which has never
  published one. The Dockerfile installs `python3-dev` only if `Python.h` is
  missing, so this compiles during the build — slow under qemu, fine natively.
- **The target device is Orin (8.7).** The JP5 release uses TensorRT 8.5.2.2
  plans; its manifest records compute capability as `unknown`, so the
  runtime does not check that field for this archive. This is not a promise
  that the plans work on another GPU. A Xavier device (7.2) needs its own
  compatible plans. Check the actual device with
  `python3 -c "import torch;print(torch.cuda.get_device_capability())"`.
- **The TensorRT API in use is 8.5+**: `num_io_tensors`, `get_tensor_name`,
  `get_tensor_mode`, `set_input_shape`, `set_tensor_address`,
  `get_tensor_shape`, `execute_async_v3`. All present in 8.5.2 — the same
  name-based API `utils/tensorrt_runtime.py` relies on for OCR.
- **Memory is the practical limit.** jp511 devices are usually Orin NX 8/16 GB
  and perception already keeps ASR, VOP (YOLOv8-World) and OCR (TensorRT)
  resident. VITS2 adds ~54 MB of engines plus a CUDA context. Measure on 8 GB
  before enabling everything at once.
- `runtime: nvidia` is required, as on jp61: the L4T base ships zero-length
  placeholders for the driver libraries and only that runtime mounts the real
  ones over them (`deploy/service.yml` sets it).

### A landmine worth knowing

`g2p_en` calls `nltk.download()` at **module import** if the corpora are not on
`nltk.data.path`. `frontend/english.py` therefore inserts the release's
`nltk_data` directory before importing it, and nothing may import `g2p_en`
earlier — including build-time checks, which is why the Dockerfile only asserts
the package is installed rather than importing it.

## Configure

The engine is a `configSchema` field, so it is visible and switchable in the
device panel:

```yaml
plugins:
  tts:
    enabled: true
    engine: vits2-zh-en     # also settable as tts_engine via action=config
    backend: trt           # VITS2 requires trt
    model_dir: /models/vits2
    speed: 1.0
    reply_language: auto   # auto / zh / en; VITS2 only
    warmup: true
```

`action=config` with `tts_engine` switches engines at runtime: the outgoing
engine's nodes are disposed first (two live publishers on one audio topic is
the duplicate-audio failure mode), then the incoming one is built in the
background. `speaker_id` must stay 0 — the model has one voice.

`speak` keeps the standard action-completion contract: a `speak-*` action id,
`interrupt` support, and the end-of-utterance marker on every terminated
utterance, including one abandoned before it reached a node.

### Chinese / English preference

The existing TTS settings expose a `reply_language` selector for VITS2:
`auto` (自动), `zh` (中文), and `en` (English). This chooses readings and, when
requested, returns a language instruction for the dialogue caller. It does
not translate text or connect ASR to a dialogue model by itself.

The existing UI renders the translated labels through `oneOf`. With no saved
selection it can initially show a placeholder; select and save “自动” if
needed. The backend defaults to `auto` when the setting is absent.

Direct `speak` remains supported without a preceding `language` call. In
automatic mode it uses the speech text; ambiguous numeric input falls back
to Chinese. A fixed selection gives numeric-only text and supported names
the corresponding Chinese or English reading. For example, `2026 10/ 20`
is read as a date in the selected language. Chinese-English mixed text is
allowed; a complete reply must contain the selected language's script.

To follow the user's current turn rather than infer from generated speech:

```json
{"action":"start","instance_id":"conversation","input_topic":"/robot/reply"}
{"action":"config","reply_language":"auto"}
{"action":"language","instance_id":"conversation","asr_result":{"text":"Hello, please check the battery.","lang":"en"}}
```

Pass the returned `instruction` to the dialogue model, then use its reply
and the returned `turn_id` in `speak`:

```json
{"action":"speak","instance_id":"conversation","text":"The battery is ready.","turn_id":"<returned turn_id>"}
```

The precedence for an explicit `language` request is:

1. `requested_reply_language` for this turn, if supplied by the caller.
2. The fixed panel selection (`zh` or `en`).
3. A valid `input_language` or recognized `asr_result.lang`.
4. Rules applied to the current user text.
5. Chinese when the text is ambiguous.

`asr_result.language` is not used as a recognition result. Missing or unknown
`lang` values fall back to text rules. Previous-turn language is not used as
the fallback. Text rules are heuristics for this bilingual application, not
a general language detector.

After an explicit `language` call, that instance requires its current
`turn_id` with speech. A newer turn rejects a late reply; it does not cancel
already queued audio. Use `interrupt` to stop audio. Language context is kept
in memory, and `stop` clears it. Instance-specific API configuration is
supported; the panel field is tool-wide, as declared by its shared scope.

### Pronunciation frontend

The frontend incorporates fanyi's 13,083-entry English Gold lexicon with
`CUSTOM → Gold → CMU → G2P` lookup for English and mixed text. The lexicon
ships with the plugin source, so changes are versioned with the code.

Context-aware normalization covers telephone numbers, dates and times,
order IDs, last-four digits, model numbers, fractions, scientific notation,
units, temperatures, mathematical expressions, URLs and email addresses.
Examples include `020-87654321`, `订单 ID：8492610`, `RTX-4090`, `2/3`,
`lr=1e-4`, and `SNR≥28dB`. Ambiguous numbers still depend on their surrounding
text; these rules do not claim to infer every identifier's meaning.

The integration preserves the current engine's background loading, ACP
completion, interrupt generations, audio EOF, and paced PCM publishing. It
does not change the acoustic adapter or TensorRT execution code.

## Model release

`utils.model_downloader.ensure_vits2_model()` installs one archive per JetPack
family from the public
[lanxijiang/TTSV4_Ch_Eng](https://www.modelscope.cn/models/lanxijiang/TTSV4_Ch_Eng)
repository at revision `e67fccc9fa0c2bd8848dcf93221e18bb3fa4f205`, pinned by size
and SHA256, using the same lock /
staging / retry path as the other verified models. It returns the engine
directory for the TensorRT that is actually importable — TensorRT plans are not portable across
TensorRT majors or GPU architectures.

The JP6 archive is 160,688,390 bytes and the JP5 archive is 162,677,801 bytes.
They retain the existing `lc1_male_v6_frontend_146k` release from the
Liaoqianqian-based implementation, including ONNX build inputs and reference
samples. Runtime loads only the TensorRT plans and frontend resources; there
is no on-device engine build in this installation path.

Each archive contains a release directory with `config.json`, `engines/`,
`frontend_data/`, `tn_cache/`, and `nltk_data/`. The loader points both engine
and frontend to that same release. The NLTK corpora ship with it so the
container does not download them at runtime. These archives have different
engine and Chinese TN hashes from the former COS bundles; they are pinned
explicitly rather than treated as interchangeable copies.

Installation happens in a background loader on the first `start`/`speak`, which
returns `{"state": "loading"}` immediately — the download plus three TensorRT
engines plus warmup takes far longer than the 60 s Agent Core allows a
processor `tools/call`. Instances that were requested while the model loaded
come up on their own when it is ready, and utterances queued in the meantime
play then. `info` never blocks and never triggers a load.
