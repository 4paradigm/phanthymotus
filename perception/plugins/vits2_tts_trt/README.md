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
- **Compute capability must be 8.7.** The engines are built for Orin
  (`trtexec 8.5.2.2`, sm87) and `_validate_manifest` refuses a mismatch with
  `GPU compute capability mismatch`. Orin AGX/NX/Nano are 8.7; a Xavier device
  (7.2) needs its own plans built on it. Check with
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
    engine: vits2_trt      # or sherpa_onnx; also settable via action=config
    backend: trt           # VITS2 requires trt
    model_dir: /models/vits2
    speed: 1.0
    warmup: true
```

`action=config` with `tts_engine` switches engines at runtime: the outgoing
engine's nodes are disposed first (two live publishers on one audio topic is
the duplicate-audio failure mode), then the incoming one is built in the
background. `speaker_id` must stay 0 — the model has one voice.

`speak` keeps the standard action-completion contract: a `speak-*` action id,
`interrupt` support, and the end-of-utterance marker on every terminated
utterance, including one abandoned before it reached a node.

### Reading language

Every utterance has a reading language that decides how **notation** is read —
numbers, dates, times, units, symbols, model names such as `Qwen3`. Words are
always spoken in the script they are written in, so mixed Chinese–English text
works either way; nothing is translated.

| Value | Behavior |
| --- | --- |
| `auto` (default) | Follow the sentence around the notation. A bare `2026 10/20` with no sentence follows the previous utterance on that instance, else Chinese |
| `zh` | Read notation in Chinese |
| `en` | Read notation in English |

| Input | `zh` | `en` |
| --- | --- | --- |
| `2026 10/20` | 二零二六年十月二十日 | October twentieth, two thousand twenty six |
| `Qwen3` | 千问三 | Qwen three |
| `RTX 4090` | R T X 四零九零 | R T X forty ninety |
| `2/3` | 三分之二 | two thirds |
| `y = 2x-3` | y 等于二乘以 x 减三 | y equals two times x minus three |
| `精度是5μm` | 精度是五微米 | 精度是 five micrometers |

It is **not** a card setting. The only way to choose is `speak`'s optional
`reading_language` argument, for one utterance; everything that does not pass
it — `on_notify`, the input topic, an LLM that leaves it out — reads `auto`,
which is right for normal replies. Set it only when a user explicitly asks for
numbers in one language. `action=support_language` returns the live engine's
language codes (`["zh", "en"]` here) and works on every engine, even while one
is loading. Other engines ignore `reading_language`.

It is a reading choice, never a gate: text the model cannot fully say (kana,
Cyrillic, emoji) is spoken with the unsayable characters dropped, as before,
and if the language step itself fails the plain reading is used. The logic is
`frontend/language.py`; it is stateless and resolved per utterance on the
node's worker.

### Pronunciation dictionary

English and mixed text use the model's CMU dictionary first, then a compiled
external dictionary, then G2P. The external dictionary is
`frontend_data/external_lexicon.json` **inside the model archive** — pinned by
the archive's own SHA256, with no separate download. It holds:

- 13,198 English entries: the fanyi Gold source plus reviewed business and
  computing terms. Entries that overlap CMU keep the CMU reading; 6,266
  letter-spelled abbreviations match exact uppercase only, so ordinary
  lowercase words are untouched.
- The Chinese phrase annotations `frontend/heteronym.py` used to hold in code
  (105 custom phrases, 10 jieba phrases, 9 robot/UI phrases such as 长按、重试).
- 3,558 terminology text mappings (577 active rules) and two isolated-synthesis
  plans (`CUDA`, `SageAIOS`).

At load the adapter checks the dictionary was compiled against this release's
CMU data and fails the load otherwise — a mismatched dictionary mispronounces
silently. It is rebuilt offline with `perception/tools/vits2_prepare_external_lexicon.py`
and shipped in a new archive; it is never compiled on a robot.

For deployment-local terms, `VITS2_HOTWORDS_PATH` (+ optional
`VITS2_HOTWORDS_SHA256`, `VITS2_HOTWORD_DOMAIN`) adds a small reviewed manifest;
`VITS2_LEXICON_PATH` (+ `VITS2_LEXICON_SHA256`) replaces the dictionary. Both
are read at load, not per word. The manifest format is validated in
`frontend/hotwords.py`.

## Model release

`utils.model_downloader.ensure_vits2_model()` installs one archive per JetPack
family (~60 MB) from COS, pinned by size and SHA256, using the same lock /
staging / retry path as the other verified models. It returns the engine
directory for the TensorRT that is actually importable, so `engines/jp61` and
`engines/jp511` can never be confused — TensorRT plans are not portable across
TensorRT majors or GPU architectures.

The archive unpacks to `config.json`, `engines/<family>/`, `frontend_data/`
(with `external_lexicon.json`), `tn_cache/` and `nltk_data/`. The NLTK corpora
ship with it precisely so the container never calls `nltk.download()` at
runtime. The engines and TN graphs are the `lc1_male_v6_frontend_146k` release,
upstream at [lanxijiang/TTSV4_Ch_Eng](https://www.modelscope.cn/models/lanxijiang/TTSV4_Ch_Eng)
revision `e67fccc9fa0c2bd8848dcf93221e18bb3fa4f205`, with the dictionary from
revision `275f343e89d9cef520c829e9fff3e950354b9b12`. Only runtime files were
repacked and mirrored to COS — the ONNX graphs and reference samples upstream
are build inputs. This is a different checkpoint from the earlier Starlight777
release (different engines, `config.json` and Chinese TN graphs); the archive
names carry `v6` so the two never collide, and an upgrading robot re-extracts
over the same directory.

Installation happens in a background loader on the first `start`/`speak`, which
returns `{"state": "loading"}` immediately — the download plus three TensorRT
engines plus warmup takes far longer than the 60 s Agent Core allows a
processor `tools/call`. Instances that were requested while the model loaded
come up on their own when it is ready, and utterances queued in the meantime
play then. `info` never blocks and never triggers a load.
