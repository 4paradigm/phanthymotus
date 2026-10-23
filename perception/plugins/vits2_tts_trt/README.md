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

### Container ports for parallel evaluation

The Jetson image starts through `deploy/run_perception.py`. When `MCP_PORT`
or `WS_PORT` is supplied, it copies the source YAML (including a custom
`CONFIG_PATH`) into a private file under `/run/perception`, overrides only
the supplied ports, and launches the unchanged `main.py` with that file as
`CONFIG_PATH`. Without port overrides it uses the original configuration.

For Darvin's three host-network containers, supply `15720/15721`,
`15820/15821`, and `15920/15921` respectively. Each container gets its own
configuration; plugin settings, models, and source YAML are preserved.
Deployment scripts that replace the image's command should also invoke
`python3 /deploy/run_perception.py /work/main.py` to apply these overrides.

### TTS engine

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
    prepare_on_config: true  # optional VITS2 preparation before speech
```

`action=config` with `tts_engine` switches engines at runtime: the outgoing
engine's nodes are disposed first (two live publishers on one audio topic is
the duplicate-audio failure mode), then the incoming one is built in the
background. `speaker_id` must stay 0 — the model has one voice.

With `prepare_on_config: true`, `action=config` also starts model loading and
warmup, waiting up to 20 seconds by default. It returns `model_state: ready`
when ready, or `state: loading` while preparation continues in the background.
`info` remains available during preparation, and `start` remains non-blocking.
Set this option to false to defer loading until `start` or `speak`.

`speak` keeps the standard action-completion contract: a `speak-*` action id,
`interrupt` support, and the end-of-utterance marker on every terminated
utterance, including one abandoned before it reached a node.

### Text chunking and synthesis

The frontend normalizes the complete input, including numbers and units,
before choosing synthesis chunks. The default `max_chunk_tokens: 256` in
`config.yaml` limits each chunk by the model's input ID count: phoneme IDs
plus blank IDs when enabled by the model. The number of written characters
and the resulting audio duration vary with the text.

The shared [text splitter](frontend/chunking.py) applies these steps:

1. Split at punctuation boundaries, retaining punctuation and closing quotes.
2. For a unit over the token limit, try Chinese/English transition boundaries,
   then jieba word boundaries, then remaining permitted character positions.
   Each step chooses a fitting prefix and recursively handles the remainder.
   Recognized Chinese-number/KB/MB/GB-style expressions are protected from
   internal cuts. The character fallback can split an unusually long word;
   this is not a semantic or prosody-aware sentence parser.
3. Greedily join adjacent valid units while the combined token count fits.
   Several short sentences, including mixed Chinese/English text, can therefore
   share one synthesis chunk. Reject input that cannot fit safely rather than
   silently discard its remainder.

The [adapter](adapter.py) calls `engine.synthesize` once per text chunk and
receives that chunk's complete PCM before yielding its audio frames. The engine
may decode long latent sequences internally with overlapping windows; those windows are
reassembled before the adapter receives PCM. Streaming to the caller therefore
begins after the first text chunk has been synthesized.

`MIX_VITS_CHUNK_PAUSE_MS` defaults to `0`: no extra silence is inserted between
synthesis chunks. Punctuation-related pauses can still be present in the
generated waveform. Smaller text chunks can reduce the first synthesis wait,
but add inference calls and prosody boundaries; increasing or decreasing this
limit needs listening and performance checks on the target device.

### Audio delivery

Audio frames are transport units created from the synthesized PCM. With the
default 16 kHz, mono, signed 16-bit output:

| Setting | Default | Behavior |
| --- | --- | --- |
| `MIX_VITS_CHUNK_BYTES` | `3200` bytes | Each full frame carries 100 ms of audio; the final frame may be shorter. |
| `MIX_VITS_PREBUFFER_FRAMES` | `5` | Send an initial burst of up to 0.5 s of available audio, then continue paced delivery. This does not insert silence. |
| `MIX_VITS_FRAME_INTERVAL_MS` | `100` ms | Pace full frames at their playback duration. |
| `MIX_VITS_SYNTH_QUEUE_FRAMES` | `200` | Bound the frame queue between synthesis and publishing; the producer waits when it is full. |

Synthesis and publishing run in separate threads so queued audio can continue
to be sent while the next text chunk is being generated. The queue bounds the
handoff buffer; it does not cap TensorRT workspace or the complete PCM held for
an individual text chunk. Scheduling and initial prebuffering are per utterance.

`MIX_VITS_FRAME_INTERVAL_MS` cannot select an interval below the PCM duration
in live mode. Faster delivery requires **both** an explicit smaller interval
and the existing `MIX_VITS_ALLOW_FAST_DELIVERY=1` option. This opt-in is only
for an offline consumer that stores every PCM frame; do not enable it with the
web player or robot speakers. No default config enables it, and the plugin
never detects or changes delivery mode based on a benchmark name or input topic.

Measure generation time separately from time to receive the last PCM frame:
realtime delivery normally makes the latter RTF approach 1 for long speech,
even when generation is faster. Report the timing boundaries with each RTF;
changing delivery intervals does not demonstrate faster model inference.
Build and unit-test results do not establish device playback or Darvin
performance. Validate complete audio reception, consecutive queued replies,
interrupts and the required concurrent instances on the deployment device.

### Image test dependency

Perception declares `jsonschema==4.23.0` in its component `requirements.txt`.
Its direct consumer in Perception is the existing music schema test, not TTS
inference. The current PR test runner installs pytest and pytest-asyncio but
does not consume a separate component test requirements file. Keeping this
declared package in the image lets that existing suite run without changing
the shared runner or music assertions.

This installation adds image content. The verified dependency wheels contain
1,987,034 uncompressed bytes for JP5/Python 3.8 and 2,136,403 bytes for
JP6/Python 3.10 (about 1.90 and 2.04 MiB). These totals include dependencies
that a base image may already provide; they are not measured final image
growth, and do not include install-generated bytecode. Full target installed
size and image delta have not been measured. Once the shared runner installs
this dependency in its test container, it can be removed from the production
image. This PR does not claim zero dependency or image growth.

### Chinese / English preference

The existing TTS settings expose a `reply_language` selector for VITS2:
`auto` (自动), `zh` (中文), and `en` (English). This chooses readings and, when
requested, returns a language instruction for the dialogue caller. It does
not translate text or connect ASR to a dialogue model by itself.

The existing UI renders the translated labels through `oneOf`. With no saved
selection it can initially show a placeholder; select and save “自动” if
needed. The backend defaults to `auto` when the setting is absent.

#### Direct card audition

Direct `speak` remains supported without a preceding `language` call. In
automatic mode it uses the speech text. Neutral numbers/dates/symbols and supported
standalone names retain the same instance's most recent explicit language;
without that history they fall back to Chinese. For example, `Hello` followed by
`2026 10/23` reads the date in English; `你好` then changes subsequent neutral
readings to Chinese. Ordinary ambiguous bilingual prose keeps its conservative
Chinese fallback. This does not automatically subscribe to ASR. A fixed selection gives
supported dates, numeric fields, quantities, symbolic expressions and standalone
names the corresponding Chinese or English reading, including fields within
mixed prose. For example, `2026 10/ 20` is read as a date in the selected language.
For manual speech, this selection is a reading preference: Chinese-English
mixed text and standalone names or acronyms are allowed without requiring
characters from the selected language. It does not translate the input.
`Qwen3` and `Qwen 3` both read as `千问三` in Chinese mode and `Qwen three` in
English mode. Empty input and unsupported scripts are still rejected.

Standalone formulas, unit quantities, recognized model/identifier formats,
URLs and email addresses use the resolved language too: variable letters,
unit/model prefixes and address characters alone are not evidence of English
prose. For direct auto speech, the existing parsers must cover the whole input
to treat it as neutral; ordinary sentences retain their local contextual routing.
This is deterministic text normalization, not semantic interpretation of
arbitrary notation or translation of an English sentence.

| Input | 中文 (also direct auto without prose evidence) | English |
| --- | --- | --- |
| `y = 2x‑3` | `y 等于二乘以 x 减三` | `y equals two times x minus three` |
| `RTX‑4090` | `R T X 四零九零` | `R T X four zero nine zero` |
| `1.2A` | `一点二安培` | `one point two amperes` |
| `2/3` | `三分之二` | `two thirds` |
| `$1,234.50` | `一千二百三十四点五零美元` | `one thousand two hundred thirty four dollars and fifty cents` |
| `21st` | `第二十一` | `twenty first` |
| `v1.2.3` | `V 一点二点三` | `V one point two point three` |
| `-.5V` | `负零点五伏特` | `minus zero point five volts` |

For standalone addresses, Chinese mode uses `点`, `艾特`, `冒号` and `斜杠`
where English mode uses `dot`, `at`, `colon` and `slash`. Address letters are
spelled and digits are read individually; URL paths are not reinterpreted as
fractions or dates. Direct auto falls back to Chinese for these addresses when
the instance has no remembered language.
In fixed Chinese/English modes, addresses and recognized numeric fields within
prose also use the selected reading. `2026 10/23 message` in Chinese mode reads
the date as `二零二六年十月二十三日` and retains the English word `message`.
`2026 10/23 星期五` in English mode reads the date as
`October twenty third, two thousand twenty-six` and retains `星期五` in Chinese.
Explicit newlines remain separate; dates do not span multiple lines. Ordinary
words are not translated. Auto prose keeps the existing clause-local readings.
The fixed preference is captured when the request is accepted, so a later
panel change does not alter queued speech. An explicit `language` turn uses
its resolved language for these fields as well.

English standalone fractions use fraction nouns (`one half`, `two thirds`,
`three quarters`). An explicit arithmetic context such as `Calculate 2/3` or
`x=2/3` retains the division reading. Dates and URL paths keep priority over
fraction parsing.

Currency parsing recognizes complete `$`, `£` and `€` amounts with valid comma
grouping. English uses major/minor units for up to two decimal places; greater
precision is retained as a decimal amount. Ambiguous currency symbols are not
assigned a currency by guessing. English ordinal suffixes must be valid (`11th`,
`21st`); generated word hyphens are normalized to spaces to avoid artificial
pauses. Version readings require `v`/`V` or a `version`/`版本` label and preserve
leading zeros in each component. Explicit IDs have priority and read their
digits separately, including dotted IDs. Unlabelled unknown names such as
`abc123` retain the existing lexical route. Prerelease/build suffixes are not
interpreted as a partial version. Leading-dot decimals such as `.5` and `-.5`
are supported in quantities, formulas and measurements, including the additional
`V`, `mV`, `kV`, `W`, `mW` and `kW` units. This does not add or replace a lexicon,
change acoustic parameters, or alter audio delivery timing.

A complete ASCII-labelled field such as `ID:abc123` or `Order ID:abc123` is
also neutral notation for auto mode: its digits follow the remembered language,
or Chinese in a new instance. Full sentences such as `The order ID is abc123`
and Chinese labels retain their language evidence. This does not merge labels
with values across line breaks.

#### ASR-driven dialogue integration

The `language` action is an integration API. The dialogue caller must forward
the **current user utterance's** ASR result, use the returned instruction when
generating a reply, and attach the returned turn ID when speaking. Selecting
“自动” in the panel alone does not create this connection. This plugin does not
subscribe to ASR results or insert instructions into the dialogue model itself.

```text
ASR text + lang
    -> tts(action=language)
    -> reply_language + instruction + turn_id
    -> dialogue model generates the reply using instruction
    -> tts(action=speak, text=reply, turn_id=returned_turn_id)
```

Configure automatic selection and start a conversation instance, then supply
the recognized user utterance before generating its reply:

```json
{"action":"config","reply_language":"auto"}
{"action":"start","instance_id":"conversation","input_topic":"/robot/reply"}
{"action":"language","instance_id":"conversation","asr_result":{"text":"Hello, please check the battery.","lang":"en"}}
```

For this request the result includes `reply_language: "en"` and
`language_source: "asr"`. Pass its `instruction` to the dialogue model, then
use the generated reply and the returned `turn_id` in `speak`:

```json
{"action":"speak","instance_id":"conversation","text":"The battery is ready.","turn_id":"<returned turn_id>"}
```

The precedence for an explicit `language` request is:

1. `requested_reply_language` for this turn, if supplied by the caller.
2. The fixed panel selection (`zh` or `en`).
3. A valid `input_language` or recognized `asr_result.lang`.
4. Rules applied to the current user text.
5. The same instance's remembered language for neutral notation or a supported
   standalone name, if available.
6. Chinese when no rule above resolves the input.

With panel mode `auto` and no per-turn override, examples are:

| Current ASR text | ASR `lang` | Reply language | Source |
| --- | --- | --- | --- |
| `Hello, please check the battery.` | `en` | English | `asr` |
| `你好，Qwen3 is ready，然后继续。` | `zh` | Chinese, with mixed content allowed | `asr` |
| `Hello, please check the battery.` | absent | English | `text` |
| `2026 10/20` | `en` | English | `asr` |
| `2026 10/20`, no earlier language | absent | Chinese | `default` |
| `2026 10/20`, after explicit English in this instance | absent | English | `history` |

`asr_result.language` is not used as a recognition result. Missing or unknown
`lang` values fall back to text rules. Neutral input can inherit the most recent
explicit language in the same instance; this is reported as `history` with
`detected_language=null`, never as a fresh ASR detection. Text rules are
heuristics for this bilingual application, not
a general language detector. A recognized `lang` takes priority over text
inference; this layer does not validate ASR confidence or correct an incorrect
tag. Forward the original `text` and `lang` together when available: a
text-only forwarding path can only use the text fallback. If the caller also
supplies `input_language`, it must agree with a recognized `asr_result.lang`;
conflicting labels are rejected.

After an explicit `language` call, that instance requires its current
`turn_id` with speech, and the generated reply must include its selected
primary language (mixed content remains allowed). The manual-speech exception
above does not bypass this dialogue check. A newer turn rejects a late reply; it does not cancel
already queued audio. Use `interrupt` to stop audio. Language context is kept
in memory, and `stop` clears it. Changing the configured mode also clears the
remembered language; repeating the same configuration preserves it. A generated
reply passed to `prepare` does not update input-language history. Stateless
HTTP audition through `/tts/test` has no instance identifier and does not share
the MCP card's history. Instance-specific API configuration is
supported; the panel field is tool-wide, as declared by its shared scope.

### Pronunciation frontend

English and mixed text share the model's original 129,464-entry CMU cache
and Liaoqianqian's three CUSTOM entries: `CUDA`, `HAMI`, and `IELTS`.
Two additional targeted corrections for `Agentic` and `NVIDIA` use the
checked terminology annotations. These corrections are separate from the
original dictionary; an existing fanyi Gold file is not loaded as an overlay.
The complete uppercase abbreviations `US`, `IT`, `ID`, and `IP` are spelled
as letters before normalization; ordinary lowercase words keep their readings.

The terminology layer retains 3,558 text mappings and 828 IPA reference
records from jinghong's `frontend_res` collection, whose source fields name
zhangjinghong and i-wangtianshuo. The generated data matches those annotations;
577 generated rules are active after filtering unchanged readings, bare
numbers and curated overrides. IPA records are reference metadata, not direct
runtime phoneme overrides. This layer is distinct from the CMU dictionary.

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

The CMU dictionary and its runtime cache are included in the size- and
SHA256-verified model archive. No separate Gold lexicon download is required.
Old installations may retain `frontend/en_gold_pronunciations.json`; it is
ignored and does not affect loading or pronunciation. The installation uses
the existing `/models` cache and needs no extra volume or image layer.
The public ModelScope source follows the PR submission procedure; a later
approved COS mirror can use `VITS2_MODEL_BASE_URL` with the same relative file
paths and checksums.

Installation happens in a background loader, triggered by the optional config
preparation above or the first `start`/`speak`. Starting an unloaded instance
returns `{"state": "loading"}` immediately — the download plus three TensorRT
engines plus warmup takes far longer than the 60 s Agent Core allows a
processor `tools/call`. Instances that were requested while the model loaded
come up on their own when it is ready, and utterances queued in the meantime
play then. `info` never blocks and never triggers a load.
While loading, `info.desc` reports byte progress, extraction, initialization and
warmup. Superseded loaders cannot overwrite the current load's status.
