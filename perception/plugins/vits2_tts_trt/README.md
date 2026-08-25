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

## Model release

`utils.model_downloader.ensure_vits2_model()` installs one archive per JetPack
family (~60 MB) from COS, pinned by size and SHA256, using the same lock /
staging / retry path as the other verified models. It returns the engine
directory for the TensorRT that is actually importable, so `engines/jp61` and
`engines/jp511` can never be confused — TensorRT plans are not portable across
TensorRT majors or GPU architectures.

The archive unpacks to `config.json`, `engines/<family>/`, `frontend_data/`,
`tn_cache/` and `nltk_data/`. The NLTK corpora ship with it precisely so the
container never calls `nltk.download()` at runtime. Upstream is
[Starlight777/VITS2-ZH-EN-Male-16k](https://www.modelscope.cn/models/Starlight777/VITS2-ZH-EN-Male-16k)
at revision `14954122c4baf4e80b44436c4b2b167e38db4103`; the runtime-required
files of that revision were repacked and mirrored to COS.

Installation happens in a background loader on the first `start`/`speak`, which
returns `{"state": "loading"}` immediately — the download plus three TensorRT
engines plus warmup takes far longer than the 60 s Agent Core allows a
processor `tools/call`. Instances that were requested while the model loaded
come up on their own when it is ready, and utterances queued in the meantime
play then. `info` never blocks and never triggers a load.
