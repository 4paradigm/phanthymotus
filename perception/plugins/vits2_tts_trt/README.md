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
| `RTX‑4090` | `R T X 四零九零` | `R T X forty ninety` |
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

Direct speech updates automatic language memory only after frontend validation
and successful queue acceptance. A filtered/invalid request, such as an empty
result after normalization, cannot change the language of a following date.
Speech waiting for model loading reports `language_pending`; deferred items
resolve in FIFO order using preceding accepted language evidence. Old work
cannot overwrite a newer ASR turn, changed setting, or restarted instance.
Explicit `action=language` still establishes current-user context immediately.

English standalone fractions use fraction nouns (`one half`, `two thirds`,
`three quarters`). An explicit arithmetic context such as `Calculate 2/3` or
`x=2/3` retains the division reading. Dates and URL paths keep priority over
fraction parsing.

Date and currency normalization validates complete fields before assigning a
natural reading. Recognized malformed shapes are still speakable: extra date
components, invalid calendar dates, incorrect currency grouping and repeated
decimal points fall back to digit-by-digit and symbol readings in the resolved
language. For example, `$1,23.45` becomes `美元符号一逗号二三点四五` or
`dollar sign one comma two three dot four five`. This preserves leading zeros
and decimal digits instead of guessing a corrected amount or dropping a suffix.
Addresses, explicit identifiers, labelled versions and explicit arithmetic keep
their own parsing priority.
The Chinese tone reader also distinguishes lexical words ending in `额`
(such as `金额`, `余额` and `巨额`) from interjections using the existing POS
tags. Lexical words keep the dictionary tone; standalone particles retain
their existing behavior. This changes frontend tone selection, not the model.

Explicit ranges such as `0.5–1.5A` retain the range relation (`到` / `to`),
and `KB/s`, `MB/s`, `GB/s` are complete byte-rate units. Unit case matters:
`MB/s` is not treated as `Mb/s`. An ASCII subtraction sign in a formula keeps
its arithmetic meaning.

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

Calendar ranges such as `2026.12.30–1.3` keep both endpoints, including across
New Year. Yearless dates such as `1/1–1/3` require a nearby calendar label or
explicit holiday duration; bare fractions and decimals retain their existing
readings. Calendar validation checks leap years without inventing a spoken end
year. Parenthesized date fields retain their local context in automatic mode.
Energy units `J`, `mJ`, `kJ`, `MJ` have Chinese/English readings; numeric `KJ` and
`kj` are explicit aliases of `kJ`, while `mJ` and `MJ` remain distinct. Numeric
`km`/`KM` both read 公里 in Chinese and kilometers in English. These aliases do
not rewrite words, model identifiers, URLs or email addresses.

#### ASR-driven dialogue integration

The configuration panel retains 自动 / 中文 / English. The public tool contract
also retains `language`, `instance_id`, `turn_id` and ASR input fields so existing
integrations can discover and validate them. Core binds the instance ID for
connected cards; ordinary speech does not require a manually supplied turn ID.
Blank routing fields on ordinary calls are treated as omitted, including when
caching shared configuration. After an explicit `language` call, speech still
requires its current turn ID; missing or stale turns remain errors.

Card fields and LLM action parameters share the current Core schema mechanism.
This update preserves that contract rather than removing API parameters to hide
card fields. No Core renderer change is included.

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
original dictionary. The fanyi source and reviewed business/computing entries
are compiled into one Gold dictionary with CMU conflicts repaired.
The complete uppercase abbreviations `US`, `IT`, `ID`, and `IP` are spelled
as letters before normalization; ordinary lowercase words keep their readings.

#### External pronunciation dictionary

The default frontend combines all **13,083** entries from the pinned fanyi
source with **120** reviewed supplements: **115** new keys and five identical
existing keys. The resulting **13,198-entry** dictionary is compiled to
`frontend_data/lexicon/20261027-v4-7c0e657e/external_lexicon.json` outside Git:

- 225 entries overlapping CMU retain the released CMU reading by default.
  The original Gold phones remain in the data for auditing.
- 6,266 letter-spelling entries require exact uppercase spelling; this includes
  114 entries with separators, reducing unintended changes to ordinary names.
- 6,707 other OOV entries use case-insensitive matching. Their source phones
  are retained, not certified as universally correct pronunciations.

The v3 update adds six whole-token, case-insensitive product names:
`ModelSphere` (Model Sphere), `AutoTune` (Auto Tune), `LLMBench` (L L M Bench),
`SageOne` (Sage One), `SageGPT` (Sage G P T), and `HyperCycle` (Hyper Cycle).
Name provenance is recorded from the [ModelSphere project](https://github.com/modelsphere/modelsphere),
[LLM AutoTune](https://github.com/modelsphere/llm-autotune), and Fourth Paradigm's
[company](https://www.4paradigm.com/about/index.html),
[computing platform](https://www.4paradigm.com/product/advanced.html), and
[operations platform](https://www.4paradigm.com/product/cess.html) pages.
Phones reuse the released CMU component words/letter names; this is not a claim
of official brand-specific stress or audio approval. All prior 13,192 rows,
Chinese annotations, ordered normalization rules and acoustic resources are unchanged.

Seventy reviewed technical abbreviations, including `VLLM`, `LLM`, `ASR`,
`TTS`, `NLP`, `RLHF`, `GPT`, `TCP`, `SSH`, `GRPC`, `CLI`, `LLVM`, and `SQL`,
match case-insensitively. For example, `vLLM`, `VLLM`, and
`vllm` use the same source letter-name pronunciation in English and mixed
speech. This policy does not globally spell uppercase words or change source
phones. Policy edits belong to the external resource release. Publishing updated data
requires regenerating and pinning the resource; an explicit
`VITS2_LEXICON_PATH` remains caller-managed.

English and Chinese-English mixed text share the same lookup. CUSTOM and
existing verified corrections remain first; ordinary CMU words retain their
reading. Eligible external whole words precede G2P fallback. Full known
hyphenated tokens are protected during final punctuation cleanup; structured
math and addresses keep priority. Ambiguous `M-S` still follows math parsing.

The public frontend resource is one `external_lexicon.json` in the same
ModelScope repository as the model, under `frontend/lexicon/20261027-v4-7c0e657e/`, pinned to revision
`275f343e89d9cef520c829e9fff3e950354b9b12`.
It contains the merged Gold runtime entries and their original annotations,
matching policies, reviewed business-source records, text replacements,
ordered frontend rules, Chinese pronunciation annotations and IPA reference data. Vocabulary data is not embedded in generated
Python modules or duplicated as separate Git dictionaries.

The v4 resource also records the Chinese reading of `华为` and optional
isolated-synthesis parts for `CUDA` and `SageAIOS`. The latter are data in
the unified lexicon, not hardcoded names in the adapter. The original path is
retained for addresses, file names, identifiers, ambiguous matches, and
requests with more than one marked occurrence; those cases do not receive the
isolated-synthesis behavior.

The ordered rules retain their original stages: 80 pre-normalization rules,
seven post-normalization product spellings and one lexical-number rule.
Their regex boundaries, flags and order are preserved. The interpreter only
accepts literal replacements or the existing, allowlisted Ascend digit handler;
the downloaded resource is never executed as Python. The original CUSTOM
pronunciations and the early Qwen/version language handling remain in code.
The maintenance compiler lives in `perception/tools/vits2_prepare_external_lexicon.py`;
it is not part of the inference plugin. When rebuilding the dictionary, carry
forward and validate the unified resource's `chinese` and `isolated_synthesis`
sections; the Gold compiler does not generate those sections itself.

Before frontend startup, the adapter downloads/verifies this one size/SHA256-
pinned JSON into `frontend_data/lexicon/20261027-v4-7c0e657e/external_lexicon.json`. The existing downloader
provides process locking and atomic installation; valid cached bytes are reused.
Cache validation and replacement of corrupt downloads use the unchanged mainline
`ensure_verified_bundle` implementation during resource preparation, not per utterance.
The index is compiled when publishing a resource version, rather than on the
robot. Its CMU source hash must match the model; English startup also checks
its full nested CMU identity and overlapping pronunciations. IPA remains
reference data and is not used as runtime phoneme input. The acoustic archives,
engines, speaker, speed and model weights are unchanged.

Ordinary deployment uses the combined hash index directly; it does not scan
the supplement list for each token. CMU and original CUSTOM/VERIFIED priority
remain intact. A configured domain/deployment override or an older explicit
Gold resource uses indexed compatibility candidates with the original domain,
priority and CMU guards. File reads and validation remain cached at startup.

Deployments can provide a maintained external file with `VITS2_LEXICON_PATH`
and optionally `VITS2_LEXICON_SHA256`. Missing/corrupt resources or a different
CMU fail explicitly. The full dictionary has its own 4 MiB / 50,000-entry limit;
it is not subject to the optional small-manifest limit below. Lookup is cached,
not reloaded for every word. Use `reload_external_lexicon()` or restart after
replacing the same path, and update a configured checksum. Do not hand-edit
the auto-generated default cache; maintain the source/compiled deployment file.
No large dictionary is committed to Git, and the acoustic model is unchanged.

English startup binds the actual release CMU once. Each newly loaded external
file, including a manual reload or explicit path, is checked against that
binding before lookup can use it. Preparation before this binding reports
`cmu_verified=false`; schema validation alone is not a ready pronunciation
resource. An incompatible reload raises an error, and lookups of that invalid
file remain blocked until a valid file and its configured checksum are restored.
CMU hashing and full validation are cached, not repeated for every word.

#### Chinese annotations in the shared resource

The v2 resource introduced a `chinese` section to the same JSON, with its
revision, resource directory, size and SHA256 pinned together. The older v1/v2 caches
remain separate so older code can continue to use its own resource. An old
English-only JSON is insufficient for the new shared-resource loader; missing
Chinese data raises an initialization error.

The section has `version: 1` and preserves three existing registration groups:

| Field | Existing data | Representation and registration |
| --- | --- | --- |
| `custom_dict` | 105 inherited phrase readings | One pinyin-alternative list per character, numbered tones; original `tone2` loading style |
| `jieba_phrases` | 10 segmentation phrases | Ordered strings; original default jieba frequency |
| `robot_phrase_pinyin` | 9 reviewed robot/UI readings | One tone-mark pinyin-alternative list per character; jieba frequency remains at least 100 |

These are the effective Python mappings, including existing last-definition
behavior for duplicate literals. Migration changes neither annotations nor
registration order. The model's broad phrase dictionary loads first, then
`custom_dict`, then the reviewed robot phrases. Existing higher segmentation
frequencies remain intact. `chinese.source` records the original file, commit
and SHA256 for provenance.

`frontend/heteronym.py` now loads and validates this section instead of holding
word data. Reads happen during frontend initialization, never per speech token.
Chinese data uses the shared release resource; an older explicit English
`VITS2_LEXICON_PATH` remains an English override and cannot replace this section.
Restart the frontend after updating Chinese annotations because pypinyin and
jieba registrations are process-global. Do not use an English-only cache reload
to apply or remove Chinese phrase entries.

Maintain names and phrase pronunciation exceptions in these data fields.
Edit the complete maintained JSON and retain its English/rule sections; the
English-only compiler output is not a replacement for the complete resource.
Contextual particle handling, tone sandhi, language selection and numeric/text
normalization remain code. No new acoustic model or TensorRT engine is needed.

#### Reviewed business supplements and optional domain overrides

The resource's `business_lexicon` section retains the reviewed maintenance
input for the combined Gold compiler. Its 114 enabled whole-token entries supply 109 missing
words and preserve five existing Gold
readings. They do not override released CMU ordinary words, CUSTOM or VERIFIED.
Examples include Phancy, PhancyCloud, vGPU, AIOS, ModelHub, OpenMLDB and
HuggingFace. PyPI uses the documented "pie-pea-eye" reading and is distinct
from PyPy. Airbnb and ASML still come from the full external dictionary.
Computing additions include HTML, VPN, MVCC, CRDT, OLTP, OLAP, YAML and
Gunicorn. Gunicorn follows its [author's G + Unicorn reading](https://github.com/benoitc/gunicorn/issues/139);
YAML follows its [specification's rhyme with camel](https://yaml.org/spec/1.2.2/).
Unconfirmed project names are not inferred
from spelling alone.

The whole token `FOLLOW-UP` reuses the released CMU `FOLLOWUP` pronunciation,
including its secondary stress on `UP`. Ordinary spaced `follow up` and other
hyphenated words keep their existing readings. This entry also applies when
the frontend treats `follow-up` inside a bare filename or path as ordinary text;
complete URLs and email addresses retain their existing address handling.

The user-confirmed `PhancyGroup` reading is `Phancy Group`, using the same
phones as the existing Phancy and CMU Group components. Uppercase, lowercase
and mixed-case forms share that whole-token entry. Other supported fused
aliases include FourthParadigm, ModelHubXC, TokenFactory and AIStudio; these
are spelling variants of existing components, not new brand pronunciations.

The manifest retains the complete 148-term company handoff inventory and a
computing catalog with coverage and source conflicts. Catalog records
are not active rules. Unconfirmed brand readings (including PhanthyMotus),
conflicting annotations and unsupported phrases stay disabled. Sources and
component phones are retained for review; the runtime does not read private
handoff paths. Existing separated phrases retain their normal component path;
the specific company name `Frost & Sullivan` also keeps its spoken "and".
`pending_review` disables only the proposed phoneme override, not speech of
the word. For example, `Phanthymotus` still uses the existing `Phanthy Motus`
text expansion. Its two handoff transcriptions disagree on the final vowel
in Motus (`UW0` versus `AH0`); EngineX differs on its initial vowel (`IH1`
versus `EH1`). JIT's letter-by-letter versus word reading remains unconfirmed.
Structural/phoneme validation does not replace device listening.

Deployments may add a local version-1 JSON manifest using
`VITS2_HOTWORDS_PATH`, optionally pinned by `VITS2_HOTWORDS_SHA256`.
`VITS2_HOTWORD_DOMAIN` defaults to `general`; selecting a domain is explicit,
not inferred from a sentence. For example, a reviewed financial ticker may
override a word only in a financial deployment. Ordinary `Wen` and `car` retain
CMU readings in the default deployment. Token language is `en` even when the
English word appears inside Chinese speech; panel language does not translate it.

Each manifest entry must declare `text`, `arpa` (stressed ARPAbet), `case`
(`exact`/`fold`), `languages` (`["en"]`), `domain`, integer `priority`,
`enabled`, `reviewed`, `source`, and `source_sha256`. `override_cmu` defaults
to false; enabling it requires exact case and a specific non-general domain.
Higher priority wins between eligible bundled/deployment entries; ambiguous equal priorities,
invalid phones, checksum errors, and raw Gold files are rejected. The local
deployment manifest is bounded to 64 KiB. The SHA-pinned bundled business
manifest has a separate 128 KiB bound for its retained provenance and catalog.
At most 128 active entries are allowed across both manifests, with no downloads.
Changing a deployment path/SHA selects a new cached manifest. Editing the
same deployment file requires `reload_hotwords()` or a service restart; speech does not
reread the file for each token. This adds no UI, new model, or runtime package
dependency.
Changes to the bundled maintenance source require updating its pin and
rebuilding the combined dictionary before publication. Resource preparation
only downloads and validates the published JSON; it does not compile vocabulary
on the robot. Explicit compiled paths remain owner-managed.

The terminology layer retains 3,558 text mappings and 828 IPA reference
records from jinghong's `frontend_res` collection, whose source fields name
zhangjinghong and i-wangtianshuo. The generated data matches those annotations;
The resource's `text_replacements` section retains the mapping order; 577
rules are active after filtering unchanged readings, bare
numbers and curated overrides. IPA records are reference metadata, not direct
runtime phoneme overrides. This layer is distinct from the CMU dictionary.

Known lexical versions are resolved before later normalization can reinterpret
their digits. Fixed Chinese/English preferences also apply inside opposite-
language prose: `StyleTTS2` uses 二/two and `Velo3D` uses 三 D/three D.
Generated phrase matches preserve line boundaries and do not match inside
longer Latin words, including accented letters. `OpenAI`, `SkyAI`, and
`MBody AI` retain both letter names in AI; ordinary words such as `Hello`
keep the published CMU pronunciation and stress.

The Chinese frontend keeps its inherited 105 effective custom phrases and
adds nine reviewed robot/UI phrases with matching segmentation support:
长时间、长按、重试、重连、重发、调高、调低、调参、这行代码。
The particle 地 is handled in its grammatical context, separately from lexical
dì in 土地/目的地. These are targeted frontend corrections; unknown names and
ambiguous readings still require pronunciation review and device listening.

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
SHA256-verified model archive. The separate, unified frontend JSON described
above is downloaded into `frontend_data/lexicon/20261027-v4-7c0e657e/` and retains the
CMU priority rules. Older Gold copies inside model archives are not used as
global CMU overrides. Installation reuses `/models`, with no new volume or
image layer.
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
