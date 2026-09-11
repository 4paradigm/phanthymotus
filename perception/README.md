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

## TTS Engines

`tts_engine` (configSchema on the `tts` tool, and `plugins.tts.engine` in
`config.yaml`) selects the voice. Engines are named **`<model>-<languages>`** —
the same shape `asr_model` uses (`x-asr-zh-en`, `parakeet-en`, `sensevoice-small`),
because the dashboard renders the raw enum string, so these two dropdowns sit side
by side in front of the same operator. Language codes, not country codes: `zh`, not
`cn`. Naming an engine after its *runtime* was the previous mistake — `matcha-zh-en`
and `mms-th` both run on sherpa-onnx, so `sherpa_onnx` identified neither.

| `tts_engine` | model | languages | runtime | model dir |
|---|---|---|---|---|
| `vits2-zh-en` (default) | VITS2 16 kHz | 中 / 英, code-switching | TensorRT | `/models/vits2` |
| `matcha-zh-en` | matcha-icefall-zh-en + vocos | 中 / 英 | ONNX Runtime | `/models/sherpa-onnx/tts` |
| `mms-th` | MMS-TTS-THAI-MALE-NARRATOR | ไทย only | ONNX Runtime | `/models/mms-th` |
| `kokoro-multi` | Kokoro-82M v1.0 | 英 / 中 / 日 / 西 / 法 / 意 / 葡 / 印地, code-switching | ONNX Runtime | `/models/kokoro-multi/<device>` |

`vits2_trt` and `sherpa_onnx` still resolve, via `ENGINE_ALIASES` in
`plugins/tts.py`, and underscores fold to hyphens first so `vits2_zh_en` works too.
The aliases are not decoration: both old names are already persisted in ConfigDB
rows and in `config.yaml` on every deployed robot, and `_select_engine` raises on an
unknown engine — so dropping them would put every existing TTS card into
`state: error` on the next restart.

Only one engine is resident at a time. Switching disposes the outgoing one's nodes
first, because two live publishers on one audio topic play both voices at once.

### Chinese number normalisation is applied by us, never by `rule_fsts`

Affects both sherpa-onnx engines that can speak Chinese (`matcha-zh-en`,
`kokoro-multi`). Both pass **`rule_fsts=""`** and call `plugins/zh_text_norm.py`
instead. Do not "simplify" that back.

sherpa-onnx applies every FST in `rule_fsts` to the **whole text before its frontend
decides what is Chinese** (`offline-tts-kokoro-impl.h`: the `tn_list_` loop runs, and
only then `ConvertTextToTokenIds`). The frontend routes on `[一-鿿]`. So handing it
the ZH number/date/phone FSTs rewrites every digit into Chinese characters *first*,
and the frontend then reads them in Chinese — regardless of `lang`, in every language:

| input | with `rule_fsts` | fixed |
|---|---|---|
| `...opened in 2026.` | `...opened in 二千零二十六.` | unchanged, espeak reads it |
| `We have 25 exhibits today.` | `We have 二十五 exhibits today.` | unchanged |
| `Hola, tenemos 25 exposiciones hoy.` | `Hola, tenemos 二十五 exposiciones.` | unchanged |

The FSTs are not the problem and must not be dropped — for Chinese they are
load-bearing and good: `2026年` → `二零二六年`, `2026年1月15日` →
`二零二六年一月十五日`, `第25个展品` → `第二十五个展品`, `延迟200毫秒` →
`延迟二百毫秒`.

So `zh_text_norm` runs them in Python, before sherpa sees the text, and only on the
Chinese parts. **A digit run is Chinese when either neighbour is** — which gets both
directions right in a way that gating on `tts_language` could not:

```
"延迟 200 毫秒"        -> Chinese on both sides   -> 二百
"共25 items"           -> Chinese only before     -> 二十五
"In 2026 我们开业"     -> Chinese only after      -> 二零二六
"we have 25 exhibits"  -> Chinese on neither      -> left to espeak
```

Two details are load-bearing and pinned by `tests/test_zh_text_norm.py`, which needs
no model and no `kaldifst`:

- **Whole segments go to the FST, never the bare digits.** `date-zh.fst` has to see
  the `年` to produce `二零二六年` instead of the quantity form `二千零二十六`.
- **The CJK range is the same one sherpa routes on** (`一-鿿`). If the two
  disagreed, text could be normalised here and then routed to espeak anyway.

`matcha-zh-en` has no language field at all, which is the other reason the decision is
made per number rather than per engine setting.

`kaldifst` reaches the image through `plugins/vits2_tts_trt/requirements.jetson.txt`,
installed when `ENABLE_VITS2_TRT=1` (the Dockerfile default). Without it this degrades
to a **warning, not a refusal**: Chinese digits get read by espeak in the selected
voice — wrong, but audible and confined to numbers. Contrast the Thai frontend, which
*does* refuse without pythainlp, because there the digits vanish from the audio
entirely.

Known limitation, sherpa's FSTs rather than this code: `电话13800138000` becomes
`一百三十八亿零一十三万八千` — `phone-zh.fst` does not match a bare number, so
`number-zh.fst` reads it as a quantity. And a bare `2026.` with no `年` reads as
`二千零二十六`; the year form needs the context character.

### `kokoro-multi`, the multilingual voice

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) **v1.0**, added because
`vits2-zh-en` is a Chinese male voice whose English is the weaker half, and there was
no engine good enough to put in front of an English-speaking audience. 82 M
parameters, Apache-2.0, and it needs no new framework: sherpa-onnx already ships
`OfflineTtsKokoroModelConfig`, so this engine added **no Dockerfile change** — only
Python, a COS artefact and tests.

Named `kokoro-multi`, not `kokoro-zh-en`: it has voices for nine languages, and a
name claiming two would mislead every operator reading the dropdown. `multi` is also
what upstream calls the release (`kokoro-multi-lang-v1_0`).

**Language is a runtime dropdown, and switching it is free.** `tts_language`
(`plugins.tts.language` in `config.yaml`) takes `en-us`, `en-gb`, `zh`, `ja`, `es`,
`fr`, `it`, `pt-br` or `hi`. sherpa-onnx reads it from
`GenerationConfig.extra["lang"]` on *every* `generate()` call, so it is a scale on
the resident model exactly like `speed` — it is deliberately **absent from
`_session_keys`**, because putting it there would tear down and reload a 310 MB fp32
CUDA session every time someone changed the dropdown. Verified on the shipped wheels
for both Python ABIs (cp38/jp5.11 and cp310/jp6.1): `extra` accepts a plain dict and
round-trips.

**What the field actually selects is the espeak-ng voice for the *non-Chinese* runs
of the text.** Chinese is phonemised from `lexicon-zh.txt` on every setting, because
that branch of sherpa-onnx's frontend never consults `lang`. Two consequences:

- Mixed zh/en in one sentence code-switches on its own, with no Python-side
  detection — `KokoroMultiLangLexicon` splits the text on Chinese vs non-Chinese
  character runs and phonemises each separately.
- **`zh` maps to an English espeak voice on purpose**, and is not redundant with
  `en-us`: it only decides how *embedded Latin* in a Chinese sentence is read, and
  English is the right answer. Digits are already converted to Chinese characters
  upstream by the ZH rule FSTs.

**The espeak voice names in `LANGUAGE_VOICES` are measured, not guessed.** Getting
one wrong fails in two ways, both silent on the robot:

| configured | espeak voice | what a guess did |
|---|---|---|
| `en-gb` | `en-gb-x-rp` | plain `en-GB` is **not a voice espeak-ng ships** → "Failed to set eSpeak-ng voice", **no audio at all** |
| `pt-br` | `pt-BR` | — |
| `fr` | `fr` | `fr-FR` **silently truncates**: 18280 samples where `fr` gave 63277 on the same sentence, nothing raised, nothing logged |

espeak matching is case-insensitive, which is why `en-us` resolves to `en-US` and
made `en-gb` look like it ought to work. Hence a closed map rather than passing the
configured label through, and a construction-time probe that refuses to load if the
resolved voice produces no audio.

**`speaker_id` is an index within the selected language, not a global voice number.**
`speaker_id: 0` is the first voice of whatever `tts_language` is set to, so switching
language moves the voice with it and the two cannot end up disagreeing. The model's
own numbering is not learnable — Japanese starts at 37 and Spanish is 28, 29 **and**
53 — and exposing it made it easy to pick an American voice, switch to Japanese, and
wonder why the result sounded wrong. That class of mistake is now unrepresentable
rather than warned about.

| `tts_language` | voices | `speaker_id: 0` is |
|---|---|---|
| `en-us` | 20 | `af_alloy` (`af_heart` is 3, `af_bella` 2 — the highest-graded) |
| `en-gb` | 8 | `bf_alice` |
| `zh` | 8 | `zf_xiaobei` |
| `ja` | 5 | `jf_alpha` |
| `hi` | 4 | `hf_alpha` |
| `es` / `pt-br` | 3 | `ef_dora` / `pf_dora` |
| `it` | 2 | `if_sara` |
| `fr` | 1 | `ff_siwis` |

Out of range at load is a `ValueError` naming the language and its count. Out of
range *after* a language switch clamps to voice 0 with a warning instead — language
is a free per-utterance setting and must not be able to fail, and `fr` has exactly
one voice. The requested index is remembered, so switching back restores it. Logs and
`info` report the resolved name (`af_heart`), not the number.

> **Only `en-us`, `en-gb`, `zh` and `ja` have been exercised on device.**
> sherpa-onnx's own documentation says of this model "it is a multi-lingual model,
> but we only add English and Chinese support for it" — Kokoro was trained with
> misaki G2P, and for `es` / `fr` / `it` / `pt-br` / `hi` the espeak phoneme set is
> not guaranteed to be the one the acoustic model learned. All of them produce audio
> of a plausible length; treat those five as best-effort until someone who reads the
> language has heard them.

#### Measured on Orin 6 (jp6.1), fp32 on gpu

| lang | sid | audio | synth | RTF | resample |
|---|---|---|---|---|---|
| `en-us` | 3 | 8.46 s | 0.78 s | **0.092** | 26 ms |
| `en-gb` | 26 | 10.62 s | 0.85 s | 0.080 | 36 ms |
| `zh` | 47 | 9.02 s | 0.89 s | 0.099 | 27 ms |
| `ja` | 37 | 6.44 s | 0.54 s | 0.085 | 18 ms |
| `es` / `fr` / `it` / `pt-br` / `hi` | — | ~4-6 s | 0.31-0.44 s | 0.070-0.083 | 11-22 ms |

Session build 2.6 s (well inside `ENGINE_SWITCH_WAIT_S = 20`), warmup probe 0.93 s,
and the resampler costs 3-4% of synthesis — not worth a polyphase rewrite.

**`cpu` is not viable for streaming.** int8 on the same box measured **RTF 1.6** —
slower than real time, so the audio cannot keep up with the frame clock. That is why
`ENGINE_DEVICE_DEFAULTS` puts this engine on **gpu**; `cpu` exists for hosts with no
CUDA wheel and for offline synthesis, not for live speech. (Note the dashboard form
still renders the schema's `cpu` default, since JSON Schema cannot express a
per-engine one.)

**It is the only engine whose sample rate is not the pipeline's** — Kokoro is
24 kHz, everything downstream is 16 kHz, and `AudioChunk` has no way to be told
otherwise (the rate lives inside the `audio/pcm-16k` format string, and the
publisher's pacing derives from `SAMPLE_RATE`). So the adapter downsamples
internally, in `utils/resample.py`: 24000 → 16000 is exactly 2:3, so it upsamples by
2, lowpasses at 8 kHz with a 97-tap Blackman-windowed sinc, and decimates by 3 —
numpy only, because scipy would pull its own numpy pin and every plugin
requirements file here exists partly to prevent that. Nothing outside the adapter
knows Kokoro is 24 kHz.

Checked against `scipy.signal.resample_poly` on a real 8.5 s utterance: correlation
**0.999997**, peak error −39.6 dB, and the residual confined to the 7–8 kHz
transition band (error/signal 3.8e-3 there against ~1e-6 below 6 kHz). The ~2-5% DC
offset in the output is **Kokoro's own** — the raw 24 kHz is +0.0236 and the
resampled 16 kHz is +0.0236, which is what a unity-DC-gain lowpass should do.

Two details in that resampler are load-bearing and are pinned by
`tests/test_resample.py`: the whole utterance is resampled *before* being framed
(doing it per 3200-byte frame restarts the filter every 100 ms, which is ten clicks
a second), and clipping happens in float before the int16 cast (`astype(np.int16)`
on an out-of-range value wraps, turning an overshoot into a full-scale *opposite*
polarity tick on the loudest part of the utterance). A third was found by those
tests rather than by ear: `np.convolve(mode="same")` returns
`max(len(signal), len(taps))`, so an utterance shorter than the 97-tap filter came
back padded out to 97 samples of filter tail — which a punctuation-only chunk hits.

**`device` selects different weight files here, not just a provider.** `gpu` gets
fp32 and `cpu` gets int8, because `provider_for_device` refuses int8 on CUDA (the
CUDA provider falls back to CPU per quantised node). So the two are separate pinned
archives that unpack into `/models/kokoro-multi/gpu` and `.../cpu` — flipping the
field downloads the other one and leaves the first in place.

The release is repacked from sherpa-onnx's `kokoro-multi-lang-v1_0` by
`tools/repack_kokoro_v1_0.py` and mirrored to COS, pinned by size + SHA256 like
every other model here. Two things upstream ships are dropped:

- `lexicon-us-en.txt` / `lexicon-gb-en.txt` (11.6 MB) — **unreachable.** For
  non-Chinese runs sherpa-onnx short-circuits to espeak whenever `lang` is
  non-empty, and `lang` falls back to the model's own `meta_data.voice` (`"en-us"`)
  when unset, so it never is. English pronunciation here comes from espeak-ng, not
  from a dictionary.
- `dict/` (14 MB, the jieba dictionary) — ignored since sherpa-onnx v1.12.15, which
  logs "you don't need to provide dict_dir" if you pass one.

`lexicon-zh.txt` stays: the Chinese branch does *not* consult `lang`, so it is
genuinely used. The three ZH rule FSTs stay for number/date/phone normalisation.
Result: **~358 MB unpacked on gpu, ~157 MB on cpu** (from 384 MB / 183 MB upstream).

`_validate_kokoro_manifest` checks the release *before* `OfflineTts` is
constructed, and one of those checks is not optional: for a `version >= 2` Kokoro
model with **both `lexicon` and `lang` empty**, sherpa-onnx's `InitFrontend` calls
`SHERPA_ONNX_EXIT(-1)` — a **process exit**, so `main.py`'s try/except around the
TTS plugin cannot turn it into a card in `state: error`; it takes ASR, VOP and OCR
down with it. Same hazard `_validate_thai_manifest` exists for. It also refuses a
`model_version` of 1 (v0.19 ignores `lang`, so the dropdown would be visibly present
and silently inert) and any `sample_rate` other than 24000.

### `ja`: why it is hard, and how far this gets

Two separate problems, and the second is only partly solved.

**1. Raw Japanese is read in Mandarin.** sherpa-onnx's Kokoro frontend splits text on
`[一-鿿]` and sends that range to `ConvertChineseToTokenIDs`, which reads
`lexicon-zh.txt`. **That function never receives the requested language.** Japanese
kanji are inside the range (`今` U+4ECA, `私` U+79C1, `何` U+4F55), so kanji are
pronounced in Mandarin whatever `tts_language` says. Measured on Orin 6:

| text | `ja` | `en-us` | `es` | `fr` |
|---|---|---|---|---|
| `今日私何` (kanji only) | 30682 | 30682 | 30682 | 30682 |
| `こんにちは` (kana only) | 28908 | 114986 | — | — |

Identical sample counts for kanji under every language; kana change 4x. `lang` has no
effect on kanji at all.

**2. Kana are read with holes.** Kokoro's 114-token table *is* the misaki phoneme
inventory — it contains `ʣ ʥ ʦ ʨ ᵝ`, so the model was trained to speak Japanese. But
it has no `ʑ`, and espeak-ja emits exactly that for じ, plus combining diacritics
`U+0308` and `U+031E`. sherpa phonemises with espeak and looks the result up in a
misaki-derived table, silently discarding what is missing: **12 phonemes dropped from
one sentence**, audible as gaps, and unintelligible.

**The root cause of (2) is that upstream Kokoro and sherpa-onnx use different G2P.**
Upstream drives Kokoro with misaki, whose phonemes are the table the model was trained
on. sherpa-onnx uses espeak-ng, whose inventory does not match. This is not a
limitation of the model.

`plugins/ja_text_norm.py` works around both: kanji→kana via Janome, dates and counters
by rule, then **kana→Hepburn romaji**, so nothing is left in the CJK range and the
text is phonemised by a Latin-script voice whose output Kokoro can represent.
`LANGUAGE_VOICES["ja"]` is therefore `"it"`, not `"ja"` — the voice is chosen for its
**phoneme inventory**, not its language. Measured on the reported sentence:

| | dropped | duration |
|---|---|---|
| kana + espeak `ja` | **12** | 12.11 s |
| romaji + espeak `ja` | 0 | 14.78 s — spelled out letter by letter |
| **romaji + espeak `it`** | **0** | **7.85 s** |

Two details cost more than they look:

- **Word boundaries.** Japanese has none, but the romaji is read by a Latin-script
  voice, and one unbroken `kyoowanisennijuurokunen…` is a single enormous word whose
  stress espeak has to guess: 7.62 s unspaced against 4.60 s spaced. Janome has
  already found the morpheme boundaries, so they are kept — except before a lone
  `ウ`/`ー`, since Janome splits `マショウ` into `マショ`+`ウ` and a space there gives
  `masho u` rather than `mashoo`.
- **Long vowels are doubled, not written as digraphs.** `キョウ` becomes `kyoo`, never
  `kyou`, which Italian and Spanish read as two syllables.

Also corrected here: the particle `は` is *read* `ハ` but *pronounced* わ — `今日は` is
`kyoo wa`, never `kyoo ha` — and full-width `。！？` become ASCII so espeak can find
sentence boundaries.

> **Honest limit: this sounds like a non-native speaker reading Japanese.** Verified by
> ear, not inferred. Zero phonemes are dropped and it is intelligible, which the kana
> version was not, but Italian phonemes are not Japanese ones. Making it sound native
> needs misaki-compatible phonemes reaching the model, which sherpa-onnx offers no path
> for today — its only phoneme-level entry point is the lexicon, and
> `ConvertNonChineseToTokenIDs` bypasses the lexicon whenever `lang` is non-empty,
> which it always is because it falls back to the model's own `meta_data.voice`.
>
> An earlier version of this section claimed the output was "unmistakably Japanese".
> That was inferred from "no kanji left in the text" without listening, and it was
> wrong.

#### Superseded: Japanese now bypasses sherpa entirely (`plugins/kokoro_direct.py`)

The romaji route above is what the accent limit was measured on. It is no longer the
shipping path — the limit was the *phoneme alphabet*, so the fix is to stop letting
espeak choose it. `kokoro_direct.py` loads the same `model.onnx` with onnxruntime and
feeds it misaki phonemes directly, the way upstream Kokoro is driven:

```
kana --(plugins/ja_phonemes.py, vendored misaki mora table)--> phonemes --> tokens.txt ids --> ONNX
```

The mora table is pinned to misaki at **`fdc9c5e5e` (2025-01-13)**, the commit that
matches Kokoro v1.0. This pin is the whole point and `pip install misaki` is the wrong
thing to do here — current misaki targets a newer Kokoro with a larger vocabulary:

| misaki `ja.py` | distinct phonemes | missing from our `tokens.txt` |
|---|---|---|
| **2025-01-13 (pinned)** | 31 | **0** |
| 2025-04-05 (what pip gives) | 39 | 12 — `G K g ƫ ᶀ ᶁ ᶃ ᶄ ᶆ ᶈ ᶉ` |

A test asserts every phoneme the table can emit exists in `tokens.txt`. The absence of
that assertion is what let the original 12-drop bug ship.

**This session runs in a process of its own, and that is not a tuning decision.** Two
ONNX Runtimes live in the perception process — sherpa's bundled one and the standalone
`onnxruntime` wheel that also serves face recognition — and they cannot both hold a CUDA
session on the Kokoro graph. The reason is in the dynamic linker, not in either library:

`libonnxruntime_providers_shared.so` is an 8 KB library exporting exactly
`Provider_GetHost` and `Provider_SetHost` — a process-global slot holding **one** pointer
to **one** runtime's `ProviderHost`. It carries a SONAME, and `ld.so` deduplicates a
`dlopen` by matching the requested *basename* against already-loaded objects, so the
second runtime's copy is never mapped: both get the first one. Each runtime writes its own
host into that slot as it loads a provider, **last writer wins**, and the next session
built runs against the other runtime's framework objects:

| order | result |
|---|---|
| sherpa's CUDA session first | the standalone one fails, error names sherpa's build path |
| the standalone one first | **sherpa** fails, error names the standalone build's path |

both with `Error mapping output names: Could not find OrtValue with name
'/Squeeze_2_output_0'`. Verified by watching `Provider_GetHost()` change value with only
ever one bridge mapped. **On jp5.11 the same collision is a SIGSEGV that kills the whole
perception process**, taking ASR, VOP, OCR and face with it — worse than jp6.1, and it had
never been exercised there because only Japanese reaches this path.

Order does not save it, and neither does anything short of a process boundary. Four
narrower fixes were tried and measured: a version-matched 1.18.1 build (still collides),
renaming the bridge's SONAME (`ld.so` matches the *other* object's SONAME, so changing
ours does nothing), renaming its symbols (the second file is then never mapped at all),
and finally renaming both the bridge and the CUDA provider *files* — which does work, but
only by forking ONNX Runtime's ABI and maintaining a build per JetPack line.

So `plugins/kokoro_worker.py` spawns a child that is the only ONNX Runtime in its address
space. `spawn`, never `fork`: `fork` copies already-`dlopen`ed libraries, so a forked child
would inherit sherpa's runtime and the isolation would be worthless.

**What that buys, measured on Orin 6** — one 9.3 s utterance, 121 tokens:

| | first call | steady state |
|---|---|---|
| in-process, cpu (the previous shape) | RTF 0.555 | RTF 0.525 |
| **worker, cuda** | RTF 0.861 (8.03 s, cold kernels) | **RTF 0.063** |

**What it costs**, whole-box `MemAvailable`, measured end to end through the real adapter
on Orin 6:

| | while resident | after `close()` |
|---|---|---|
| in-process cpu session (the previous shape) | 820 MB | **never returned** |
| worker holding a cuda session | 1585 MB | ~330 MB residual (1191 MB reclaimed) |

So it is more expensive *while speaking Japanese* and cheaper once it is done, which the
in-process version could never be. A spawned child's own baseline is only 32–40 MB, so the
extra interpreter is not the expensive part; the CUDA context is. An earlier note here
quoted "+372 MB net" from a differently-sequenced measurement — the table above is the one
taken through the adapter and is the one to trust.

The process boundary itself is free enough to ignore. Only a phoneme string goes in and
the float32 waveform comes out; the parent still does the resample, the 3200-byte framing,
the pacing and the DDS publish, so it is **one round trip per utterance, not per frame**.
Measured with a real 0.89 MB payload (9.3 s at 24 kHz): **2.7 ms mean, 335 MB/s** — 0.45%
of the 590 ms the GPU synthesis itself takes.

Three consequences worth knowing before turning it on:

- **Nothing is spent unless the card's language is `ja`.** `_direct()` is reached only from
  `_synthesize_japanese`, which is behind `if self._language == "ja"`. A card configured
  for any of the other eight languages never spawns the child. A card configured *as* `ja`
  pays at card start, because the construction-time warmup goes through the same branch.
- **The first CUDA call costs 8.03 s**, so the child is warmed during startup and kept
  alive. A per-utterance child would be far worse than the CPU path it replaces.
- **It is reaped after two minutes idle**, which also fixes a leak the in-process version
  had: `_direct_runtime` was assigned once and never released, so a card that spoke
  Japanese once kept the session for the adapter's whole life even after switching back to
  English. Reaping on idle rather than on the language switch is deliberate — the cold path
  is ~15 s measured end to end and an alternating tour would otherwise pay it on every
  switch. Card stop and engine switch close it immediately.

Verified through the real adapter on Orin 6: no worker exists until Japanese is used;
`en-us`/`zh` keep RTF ~0.11 with the worker resident; Japanese alternates with them at
RTF 0.067–0.070; `kill -9` on the child recovers; `close()` reclaims ~1.2 GB and a later
Japanese utterance rebuilds.

#### Fixed: every standalone-ORT session lives in one child process

The hazard this section used to describe — face on gpu before Kokoro on gpu, which
could not build the TTS engine at all — is gone, and so is the whole class it belonged
to. `plugins/ort_worker.py` spawns **one** child that owns every standalone-ONNX-Runtime
session, so the perception process holds only sherpa's runtime and the two can no longer
reach each other.

Not one process per card: one per **runtime**, which is the boundary the bug has. CUDA
contexts therefore do not increase — the parent used to hold sherpa's *and* the
standalone one's; now it holds sherpa's and the child holds the standalone one.

`spawn`, never `fork`. `fork` copies the address space including already-`dlopen`ed
libraries, so a forked child would inherit sherpa's runtime and the isolation would be
worthless. A test asserts it, because that one word is the entire guarantee.

**Where the boundary is, and why not further in.** The first attempt split at
`InferenceSession.run` and left decoding, letterboxing, alignment and the quality gate
in the parent. It worked and it was the wrong cut:

| | in-process | split at `run()` | **whole pipeline in the child** |
|---|---|---|---|
| a frame, jp6.1 | ~46 ms | ~78 ms | **48.1 ms** |
| a frame, jp5.11 | — | — | **26.4 ms** |

because the parent was sending a 2.93 MiB normalised blob and getting **0.96 MiB of
pre-threshold candidates** back — SCRFD returns 16 800 of them and a frame keeps nought
to eight — then a 147 kB crop per face for a 2 kB embedding. What the parent actually
holds is the `CompressedImage` JPEG, ~232 kB, and what it wants is a few boxes and a
512-float vector each. So the frame crosses **once, compressed**, and
`plugins/face_service.py` runs decode → detect → decode-head → align → sharpness →
**the quality gate** → embed inside the child. The gate has to come along: it sits
between alignment and embedding and decides which faces are worth embedding at all.

What stays in the parent: the FaceDB (`matrix @ embedding` on a 2 kB vector, its lock,
its files), subject selection, the payload, ROS publishing, the enrolment window. None
of it touches ONNX.

`FaceAnalyzer` now **refuses to be constructed outside the child** rather than
documenting that it should not be. A test also walks the plugin tree for
`ort.InferenceSession` calls and fails on any outside the three files allowed to have
one, because a new caller adding one would reintroduce a SIGSEGV silently.

**Verified on both lines**, in the order that fails on main: face's sessions up first,
then sherpa's Kokoro engine builds; Japanese and face share one child and interleave;
dropping one service leaves the other running.

**Two things that are not settled**, recorded rather than smoothed over:

- **A dropped session does not return memory to the OS.** Measured: unloading Kokoro
  from the child moved `MemAvailable` by **+0 MB**. The CUDA pool is returned on process
  exit, not on session destruction, so unloading frees space *inside* the child for
  reuse and stops the headroom guard mis-reading the box — it does not give memory back.
  An earlier version of this section claimed it did.
- **The face path has only been exercised on frames with no faces in them.** The
  embedding, the aligned-crop return and the registration paths have unit coverage and
  no on-device run.

#### jp5.11 runs Japanese on the CPU, for two independent reasons

Both were measured on Orin 5, and both had to be guarded, because they fail at different
moments.

**1. The CUDA path computes durations wrongly.** The graph *is* stochastic — 4
`RandomNormalLike` and 7 `RandomUniformLike` nodes — so the waveform differs run to run
and between providers **by design**, and "cpu output must equal cuda output" is not a
valid check. The **duration** is not stochastic, and that is where the defect shows:

| | cpu | cuda |
|---|---|---|
| jp6.1, 8 runs | 8.35 s, stdev 0.000, 1 distinct length | 8.35 s, stdev 0.000, 1 distinct length |
| **jp5.11**, 8 runs | 8.35 s, stdev 0.000, 1 distinct length | **6.05 s**, stdev 0.053, 3 distinct lengths |

27.5% short is audibly rushed speech. The probe used for warmup gates this: 10 tokens
give exactly 47400 samples on CPU **on both lines**, jp6.1's CUDA matches exactly, and
jp5.11's CUDA returns 16800–21000. Gating on the ORT version number would be the wrong
fix — it would not catch the next line with the same defect — so the gate is the
measurement, and it is free because the probe already ran.

**2. A CUDA child does not fit.** sherpa's own Kokoro GPU adapter takes **3.2 GB** on
jp5.11 against ~950 MB on jp6.1, so a CUDA child's allocation OOM-killed the rig —
`dmesg`: `Out of memory: Killed process … (python3) anon-rss:2420028kB` — **before the
probe could run**. So the duration gate alone is not enough; a headroom check has to come
first, because it is the one that can take the process down.

**No silent substitution.** An earlier revision quietly used the CPU when the GPU was
unavailable, and that produced the worst state available: a card configured for `gpu`,
running at RTF 0.52 instead of 0.07, with the reason in a log line nobody reads. It
took a measurement to explain why Japanese "felt slow". The card now goes
`state: error` and the message names **which** of the two problems it hit, because they
have different fixes:

| | what it means | what to do |
|---|---|---|
| *not enough memory …* | the GPU is fine, the box is full | free memory, or set `japanese_worker_device: cpu` |
| *… computes the duration path wrongly* | the GPU works and gets the wrong answer | set `japanese_worker_device: cpu`; freeing memory will not help |

The second verdict is **recorded on the machine**, keyed by the ONNX Runtime version,
because the attempt is not free: one rejected CUDA session took Orin 5's MemAvailable
from 5754 MB to 1935 MB and **kept it** — unloading does not return it. A crash restarts
perception, so an in-memory verdict would be lost and the next start would pay again.
That is how face's later GPU load tipped that box into the OOM killer. Delete
`.cuda-duration-verdict` in the model directory to force a re-evaluation.

The headroom figure depends on **who else is in the child**, because the first CUDA
session there pays for the context and the rest do not:

| | measured | threshold |
|---|---|---|
| Kokoro alone, bringing its own context | 1585 MB | 2500 MB |
| Kokoro beside the face service | **967 MB** | 1400 MB |

A single conservative number refused the second case on the first case's evidence: with
2158 MB free on an otherwise idle jp6.1 box, a 967 MB allocation was declined. With the
guard asking `ort_worker.has_cuda_session()` first, face and Japanese now **both run on
the GPU in the same child** on jp6.1 — Japanese at RTF 0.089 while face recognises at
29.7–37.8 ms. An earlier version of this section said a 7.4 GB box could not fit both;
that was true of two CUDA *contexts* and not of two sessions.

jp5.11 needs `japanese_worker_device: cpu` set explicitly, and the error says so if it
is not. Two independent reasons, either one disqualifying:

- **its CUDA renders this graph wrongly** — confirmed by ear, and the cause is not
  known. Ruled out: the two-runtime collision, the model file, TF32, the ONNX Runtime
  version, and the CUDA provider binary. The full record, so nobody repeats those five
  experiments, is **[docs/jp511-cuda-kokoro.md](docs/jp511-cuda-kokoro.md)**;
- and separately, the box rarely has room.

Worth being precise about what was *not* broken, because "GPU works on jp5.11" is also
true: sherpa's own engines and face both run on the GPU there and always have.
`KokoroDirect` is the only thing affected, it exists only for Japanese, and it was
pinned to the CPU from the day it was written, so this path had never been exercised on
that line until now.
**The residual risk is stated rather than hidden**: the thresholds are heuristics, and
there is a window where CUDA would be attempted on jp5.11 — the duration gate catches it
if the build survives, and does not if the box OOMs first. `japanese_worker_device: cpu`
removes the window on that line.

Both CPU configurations remain safe, and are safe for the same reason: a CPU-only session
loads no CUDA provider, so it never touches the bridge. `japanese_worker_device: cpu` keeps
the process boundary at RTF 0.525; `japanese_worker: false` goes back to the in-process CPU
session entirely. That fallback also happens automatically if the child cannot start or
dies — Japanese losing 8× is acceptable, Japanese breaking is not.

The other eight languages are unaffected either way. Measured in one process with the
Japanese session live throughout:

| language | runtime | RTF |
|---|---|---|
| en-us / en-gb | sherpa, cuda | 0.31 / 0.21 |
| zh | sherpa, cuda | 0.14 |
| es / fr / it / pt-br / hi | sherpa, cuda | 0.10 – 0.11 |
| **ja** | **worker, cuda** | **0.063** |

On the CPU path, threads are the only lever and they matter — `intra_op_num_threads`
defaults to one per core (RTF 0.52); the ORT default of 2 gives 1.17, slower than real time.

**Pitch accent is not implemented and cannot be with this model.** The pinned misaki has
no accent code at all (it arrived in 2025-04, alongside the larger vocabulary above),
and sherpa's v1.1 token table is a symlink to v1.0's. Japanese here is correctly
*phonemised* but flat. Fixing it needs a newer Kokoro, which is a different change.

### Chinese: use `vits2-zh-en`, and why Kokoro's Chinese is not broken

Chinese takes a different path from Japanese and loses nothing. `Skip unknown phonemes`
counts on Orin 6:

| text | dropped |
|---|---|
| Chinese, ordinary | **0** |
| Chinese with embedded English | **0** |
| Chinese, rare characters (`饕餮纹鼎鬲甗簋簠盨匜盘`) | **0** |
| English | **0** |
| Japanese kana (for contrast) | **12** |

`ConvertChineseToTokenIDs` reads `lexicon-zh.txt`, whose entries are already misaki
phonemes (`七 ʨ ʰ i →`), and never touches espeak — so there is no bug here to fix.
What remains is that Kokoro is one 82 M model covering nine languages, while
`vits2-zh-en` is a **purpose-trained 16 kHz Chinese voice** and is already the default
engine. Prefer it for Chinese; `kokoro-multi` exists for English.

### `mms-th`, and why it cannot be handed raw text

The Thai voice is an ONNX export of
[VIZINTZOR/MMS-TTS-THAI-MALE-NARRATOR](https://huggingface.co/VIZINTZOR/MMS-TTS-THAI-MALE-NARRATOR),
a fine-tune of Meta's MMS VITS — 36 M parameters, **natively 16 kHz** (which is why
this voice and not the more popular `FEMALEV2`, which is 22.05 kHz and would need a
resampler), end-to-end so there is no vocoder, and character-level so there is no
lexicon and no espeak-ng data. Produced by `tools/export_mms_thai_onnx.py`.

**Licence: CC-BY-NC-4.0, inherited from `facebook/mms-tts`. Non-commercial.**
Re-evaluate before shipping it in a product. Every small Thai model with usable
quality has the same constraint — the MMS family and Piper's `th_TH-tsync2` are all
CC-BY-NC — and the only permissively-licensed alternative found was
`VachaSpeech-0.6B` (Apache-2.0), a 0.6 B autoregressive model that is not viable on
CPU and marginal on an Orin GPU.

The tokenizer is character-level over exactly **71 characters** and skips
everything else. sherpa-onnx records the loss as a C++ stderr line that never
reaches the Python logger or the dashboard; the audio simply comes back short. So
`plugins/thai_frontend.py` is **not optional**, and three of the omissions are not
guessable:

- **`ำ` (U+0E33 SARA AM) is not in the table**, but `ํ` (U+0E4D) and `า` (U+0E32)
  are. MMS was trained on text spelling the vowel as that pair, so every `ำ` is
  rewritten — otherwise น้ำ, ทำ, คำ, สำหรับ and a large part of the language lose
  their vowel. Measured: "น้ำ ทำ คำ สำหรับ น้ำหนัก" logs five skipped U+0E33 and
  synthesizes 1.34 s; rewritten it is 1.51 s and those five vowels are audible.
- **`ๆ`** (maiyamok) is absent, so `ต่างๆ` is expanded to the repeated word.
- **Of the Arabic digits only `0 1 2 4` are present** — `3` and `5`-`9` are not, and
  neither are the Thai digits `๐`-`๙`. No numeral can be passed through; every
  number becomes words.

Mixed text is transliterated into Thai script rather than routed to another engine,
so the deployment keeps one voice: numbers via `pythainlp`, Chinese via
`pypinyin` → `wunsen`, Latin via a hand lexicon then a `khanaa`-based rule
fallback. **`wunsen` does not handle English** — it covers Japanese, Korean,
Mandarin and Vietnamese only — so the Latin path is the rule table in
`thai_frontend.py`, and a name it gets wrong belongs in `_LATIN_LEXICON`, not in
the rules. The consequence to be clear about: English inside a Thai sentence is
spoken with a Thai accent by the Thai voice, not natively. Native pronunciation
would need both engines resident at once, which the facade forbids.

`tests/test_thai_frontend.py` guards one property above all — every character the
frontend emits is one the model can pronounce. That is the assertion that catches a
silent drop.

### The dry-run probe is per adapter

`_TTSNode.start()` refuses to report `running` until the adapter has synthesized a
short probe. That probe used to be a literal `"."` for every engine — and this
frontend normalises punctuation to a space and then strips it, so the probe reached
the Thai model as an empty string and **every** start on the robot answered
`TTS dry-run produced no audio` while the model itself was fine.

It is now `TTSAdapter.dry_run_text`, overridden to `ก` for `mms-th` (measured
0.384 s of audio, 108 ms to synthesize — the cheapest probe that still proves the
model runs). `MmsThaiTTSAdapter.__init__` also refuses to construct if the frontend
normalises its own probe away, so a future frontend change that swallows it fails at
load, naming the cause, instead of at every start.

A related silence surfaced in the same session: a start deferred during an engine
build is replayed afterwards, and `dispatch` **reports** failure rather than raising
it — `start()` returns `{"state": "error"}`. The replay loop only caught exceptions,
so a replayed start that failed logged nothing at all and looked like a successful
one. It now inspects the result and logs the engine's own message.

### `config` must not rebuild the session it just built

`SherpaOnnxTTSPlugin`'s config action used to call `_build_tts_adapter`
**unconditionally** and then dispose every node. On Orin5 an identical, no-op
config measured **2.6–2.8 s** for `mms-th` — a full ONNX session teardown, reload
and archive re-verification — and the card then had to `start` again. The dashboard
re-applies a card's config around a speak, so that landed on *every* utterance:
"every speak takes 5 s".

It now rebuilds only when a key the session is built from actually changed
(`device`, `speaker_id`, `model_dir`, `thai_phrase_spacing`), comparing normalised
values so the dashboard's `speed: 1` does not read as a change against a stored
`1.0`. `speed` is applied with `set_speed()` on the resident model — which is what
`vits2` has always done ("avoids tearing down a resident model for a slider
change"), and why only the two sherpa-onnx engines were slow. **`matcha-zh-en`
benefits identically; `vits2-zh-en` never had the bug.**

Measured after the fix: repeated identical config **2 ms**, and speak → first
`AudioChunk` on the topic **226–333 ms**.

Two things this exposed:

- `TTSPlugin._config` carried a hardcoded `("speaker_id", "speed", "device")` into
  a newly built engine and so dropped `thai_phrase_spacing`. The engine came up
  without it and the next config saw a change and rebuilt — one extra 2.7 s per
  switch. The list is now derived from `configSchema` (`SHARED_CONFIG_KEYS`).
- The config filter dropped every falsy value, so `thai_phrase_spacing: False` and
  `speaker_id: 0` were unsendable — phrase spacing could be turned on and never
  off. `_session_keys()` now decides presence per key.

### Warmup applies to the sherpa-onnx engines too

`plugins.tts.warmup` in `config.yaml` existed all along and only the VITS2 plugin
honoured it. Unpaid, the first utterance carried two separate costs, both measured
on Orin5:

| cost | measured | note |
|---|---|---|
| CUDA kernels + memory pool | 1695 ms | first utterance 2361 ms vs 342 ms for the second |
| pythainlp lazy corpus load | **3183 ms** | `normalize()`'s first call; 0.2 ms after |

The frontend is the larger of the two and is pure CPU. `TTSAdapter.warmup()`
synthesizes `dry_run_text`, which goes through the adapter's own frontend and so
covers both; measured 1.29 s at load on device.

### Throughput, and why `device: cpu` is not viable for Thai

Measured on Orin5 for a 5 s Thai utterance:

| device | RTF | first frame (warm) |
|---|---|---|
| `gpu` (cuda) | **0.07 – 0.13** | 342 – 665 ms |
| `cpu` | **0.96 – 1.04** | ~5000 ms |

On CPU this model is barely realtime — no headroom, and first audio arrives about
when the utterance would have ended. Use `device: gpu` for `mms-th`. (An earlier
note here cited RTF 0.31 for CPU; that was measured on an x86/arm64 dev laptop, not
on an Orin, and is not representative.)

Total wall time from speak to the *last* frame is necessarily ≥ the audio duration:
the node paces publication at exactly realtime, deliberately — see
`FRAME_INTERVAL_S`, where over-delivering by 30 ms per frame made the browser player
rewind its schedule and play overlapped at 1.43x.

### The Thai deps are pinned per Python version, and `requires_python` lies

jp6.1 is cp310, **jp5.11 is cp38**, and both `pythainlp` and `khanaa` declare
`requires_python >= 3.7` while shipping code that cannot be imported on 3.8. Each
annotates a module-level name with a PEP 585 builtin generic — `_THAI_DICT:
dict[str, list]`, `def find_same_sound_consonant(...) -> list[str]` — and those are
evaluated at runtime, so the import raises:

```
TypeError: 'type' object is not subscriptable
```

pip installs them happily first. This is why the jp5.11 build failed on the first
attempt with `pythainlp==5.2.0`, and why reading a changelog is not verification
here — the metadata is simply wrong.

| package | cp38 (jp5.11) | cp310+ (jp6.1) | newer versions on cp38 |
|---|---|---|---|
| `pythainlp` | `5.0.4` | `5.3.7` | 5.0.5+ all raise the TypeError |
| `khanaa` | `0.0.6` | `0.1.1` | 0.1.0+ raise it |

Both older pins also have **different APIs**, and both differences are absorbed in
`plugins/thai_frontend.py` rather than pushed onto callers:

- `khanaa` 0.0.6 spells with `SpellWord().spell_out(...)` where 0.1.1 uses
  `Kham(...).form`. `_load_khanaa` returns one uniform callable for either. Verified
  on cp38 that they agree — `สต+เอะ+ก+tone 3` → `เสต๊ก`, `บ+อู` → `บู`.
- `pythainlp` 5.0.4 has no `expand_maiyamok`, and its `maiyamok` takes a token list
  where 5.3.7's takes a string (passing a string raises `IndexError`). So the
  frontend expands `ๆ` itself, which also fixes a leading `ๆ` that 5.0.4's helper
  crashes on.

The loaders catch `Exception`, not `ImportError`: a pure-Python package failing at
import time with a `TypeError` is exactly the case here, and an `ImportError`-only
guard let it escape `normalize()` and kill the utterance.

Verified by running the real frontend inside the actual jp5.11 perception image
(Python 3.8.10) — output is byte-identical to cp312 for every case, including
`Bumi` → `บูมิ`, `WiFi` → `ไวไฟ`, `你好` → `หนี ห่าว` and `ต่างๆ` → `ต่างต่าง`. There
is no degraded JetPack line.

The image therefore carries a build-time self-check *after* the source COPY that
runs `normalize()` on the shipped interpreter and asserts the vocabulary invariant.
Importing the dependencies is a weaker test: it passes while an API difference is
still waiting to crash on one line only.

The adapter also refuses to construct without `pythainlp`. The frontend's
per-transliterator fallbacks are deliberately quiet, but losing number conversion
is not survivable — the digits `3` and `5`-`9` are not in the token table, so they
vanish from the audio and the card would report `running` while mispronouncing every
utterance carrying a number.

### Adding a Thai voice, or replacing this one

```bash
pip3 install torch transformers onnx onnxruntime soundfile   # not in the image
python3 tools/export_mms_thai_onnx.py --repo <hf-repo> --out /tmp/thai-tts
```

The script refuses a checkpoint that is not 16 kHz or not single-speaker, and its
self-check fails if the ONNX output is silent, disagrees with torch on duration, or
ignores `length_scale` (which would make the card's `speed` field decoration).

Two metadata keys decide whether the result loads at all, and getting either wrong
is worse than an ordinary error:

- **`frontend` must be exactly `characters`.** sherpa-onnx's
  `OfflineTtsVitsImpl::InitFrontend` dispatches on that string; anything else
  reaches the lexicon branch, which logs "Not a model using characters as modeling
  unit" and calls `SHERPA_ONNX_EXIT(-1)` — a **process exit**, so `main.py`'s
  try/except around the TTS plugin cannot turn it into a card in `state: error`,
  and ASR, VOP and OCR go down with it. `_validate_thai_manifest` checks the
  release's `manifest.json` before constructing `OfflineTts` so this fails as an
  exception instead.
- **`comment` must not contain `piper`, `coqui` or `Inflect`.** Those select
  different, shorter ONNX input layouts in `OfflineTtsVitsModel::Run`.

Then tar `model.onnx`, `tokens.txt`, `manifest.json` and `LICENSE`, upload to
`public/` with credentials from `resource-center/deploy/values.env`
(`prisma/articles/upload-figs.js` cannot — it only accepts image extensions and
forces its own key shape), **re-download from COS and hash that copy**, and paste
the verified `size`/`sha256` into `THAI_TTS_ARCHIVE`. Hashing the local file you
uploaded defeats the point of the pin, which is to catch a bad transfer.

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
| `x-asr-zh-en` | int8 + fp32 | — not offered | 0.80x, i.e. slower |
| `parakeet-en` | int8, 104 MB | **fp32, 437 MB** | **4.9x** long / **2.0x** short |

### Removed models

`paraformer-zh-en`, `paraformer-offline` and `zipformer-en` were dropped from the
registry for accuracy. The two bilingual paraformers were worse than
`sensevoice-small` on the same audio; `zipformer-en` is LibriSpeech — 960 h of
clean read audiobook speech — which is the wrong distribution for a robot whose
microphone always carries cooling-fan noise.

`REMOVED_ASR_MODELS` in `plugins/asr.py` maps each of them onto a replacement
(the paraformers → `sensevoice-small`, `zipformer-en` → `parakeet-en`) and logs a
warning. That map is not politeness: a card's `asr_model` lives in agent-core's
config DB **on each robot**, so an upgrade cannot rewrite it. A removed name that
resolves to nothing makes `config` return an error and the card comes up
`state: error` after the next restart, on every deployment that had picked one.
Do not delete an entry from that map when you remove a model — add one.

**`parakeet-en` is the English model.** NeMo FastConformer CTC 110M, trained on
~1.7 M h of diverse audio with non-speech material deliberately mixed in to
suppress hallucination, and it emits punctuation and capitalisation, which
`zipformer-en` did not. It is also the smallest offline English archive here
(104 MB int8) and needs no new runtime: `OfflineRecognizer.from_nemo_ctc` has been
in the pinned sherpa-onnx 1.13.6 all along.

Measured inside the perception image on both JetPack lines, cpu provider,
`num_threads=2` — RTF 0.039/0.054 on Orin 6 (jp6.1) and 0.040/0.055 on Orin 5
(jp5.11) for the bundle's 7.4 s and 1.0 s samples. Roughly 25x realtime on either
line, with punctuation and capitalisation in the transcript. CPU was the point of
picking this model, and cpu remains the sensible default for it.

A gpu pair is offered as well, and it went through the admission rule below on
both lines: 6 runs each, first discarded as warmup, against the int8 cpu entry.
The 7.43 s clip takes 56/57 ms on cuda against 275/271 ms on cpu (~4.9x); the
0.99 s clip takes 26/29 ms against 52/51 ms (~2.0x). The short clip wins less
because fixed per-call overhead dominates, and short is the shape most robot
utterances have — budget for ~2x, not ~5x. Transcripts were read, not just timed:
all six configurations returned byte-identical text, stable across repeats, with
none of the silent empty-transcript failure sensevoice fp16 shows on CUDA. The
weights are fp32 because no fp16 variant is published upstream and int8 on CUDA
is slower, not faster. Cold start on cuda is ~2.1 s on jp5.11 (466 ms on jp6.1)
against ~290 ms on cpu, absorbed by the load-time warmup. The ~2 GB of RAM a CUDA
context costs applies here as much as anywhere — see below.

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

TTS is simpler: Matcha and the Thai MMS model are fp32 only, so both devices load
the same files and `device` only picks the provider (Matcha measured ~4.3x). The
`vits2-zh-en` engine ignores `device` entirely — it is a TensorRT engine and never
touches ONNX Runtime.

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
slower than fp32 for that model, so there was nothing to gain by debugging it.
That model has since been removed from the registry for accuracy; the finding is
kept because it is the reason every gpu entry has to be listened to, not timed.)

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
- **`vits2-zh-en` TTS**, as above: TensorRT, not ONNX Runtime.

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

`trigger_mode` has two values: `vad` (transcribe and forward everything) and
`asr_kws` (forward only what follows a wake word). `asr_kws` is the default.

There used to be a third, `kws`, which ran a second sherpa `KeywordSpotter` on
the raw audio with its own zipformer bundle and its own `waiting_wake` state in
the VAD worker. It is gone: that is an extra model, an extra download and an
extra state machine to do a job `asr_kws` already does on a transcript the ASR
produces anyway.

`REMOVED_TRIGGER_MODES` in `plugins/asr.py` migrates `kws` → `asr_kws` at load.
Unlike a removed *model* name, leaving this unmapped would not surface as a card
error — an unrecognised `trigger_mode` falls through to `vad`, so a robot that
was wake-word gated would silently start answering every utterance in the room.
The migration also carries the wake word across, taking the display form after
`@` in the old `keywords` spec (`"x iǎo f àn x iǎo f àn @小范小范"` → `小范小范`),
because `asr_kws` with no keyword degrades to `vad` — the same silent failure by
another route. A keyword with no `@` part yields nothing and logs at error
level: the token side is a spotter lexicon, and de-spacing it would invent a
wake word nobody can pronounce.

`asr_kws` transcribes every utterance and gates on a phoneme-level
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

## Face Recognition

`plugins/face.py` — a `processor` card named **`face_recognition`**. Subscribes to an
`image/jpeg` topic, publishes identities on `{input_topic}/face` as `data/json`, and
enrols new people from a photo, the live stream, or a batch package.

### Models

InsightFace **buffalo_sc**, the smallest pack that still ships landmarks (alignment
needs them):

| File | Size | Role |
|------|------|------|
| `det_500m.onnx` | 2.5 MB | SCRFD-500M-BNKPS — detection + 5 keypoints |
| `w600k_mbf.onnx` | 13 MB | ArcFace MobileFaceNet — 512-d embedding |

Both run on the **standalone `onnxruntime`**, which is a second, independent ONNX
Runtime from the one compiled into the sherpa-onnx wheel (ASR/TTS reach only that
one; there is no supported way to run our own models on it).

`device: auto | cpu | gpu`, default **auto** — use the GPU when the installed
onnxruntime offers a CUDA or TensorRT provider, else CPU. `auto` resolving to cpu
is not warned about, because it is the expected outcome on this image: it ships
the **CPU wheel**, so today `auto` always means cpu. An explicit `gpu` that cannot
be honoured warns and degrades. A per-JetPack `onnxruntime-gpu` wheel on COS is
the follow-up, mirroring `SHERPA_GPU_WHEEL` in `Dockerfile.jetson`; nothing else
has to change when it lands, because `auto` will pick it up.

#### The onnxruntime version is pinned, and 1.19.x must not be used

**One version for both JetPack lines: 1.18.1.** That is what sherpa-onnx bundles on
jp6.1 (`sherpa_onnx/lib/libonnxruntime.so.1.18.1`), so on that line the two mapped
runtimes are ABI-identical. It runs on jp5.11 too — its cp38 aarch64 wheel is
manylinux_2_28 and focal has glibc 2.31 — verified on Orin5 (Ubuntu 20.04, glibc
2.31, Python 3.8) building a session and inferring. A per-line split was tried
first and dropped: nothing required it, and one version is one thing to reason
about.

`ORT_VERSION` is an `ARG` on the onnxruntime step itself, **not** in
`/etc/jetpack.env`, because that file is layer 2 — see § Where this layer sits.

**onnxruntime 1.19.2 abort()s the whole perception process** during
`InferenceSession()` on a Jetson where some cores are parked. It enumerates
`/sys/devices/system/cpu/present` and pins threads to every core in it; on Tianyi
in MODE_30W (`present` 0-11, `online` 0-7) `pthread_setaffinity_np` returns EINVAL
and a `std::vector` index then goes out of range:

```
pthread_setaffinity_np failed for thread: 31, index: 1, mask: {9, }, error code: 22
stl_vector.h:1123 ... Assertion '__n < this->size()' failed.  Fatal Python error: Aborted
```

Measured on Tianyi: 1.19.2 aborts with **every** `SessionOptions` combination,
including none at all, so no amount of configuration avoids it; 1.18.1, 1.17.3 and
1.16.3 each build a session and return all 9 SCRFD outputs. The build asserts the
installed version is not 1.19.x, and `warn_on_parked_cores()` logs the
present/online mismatch at load time — an abort leaves no Python traceback, so the
precondition has to be in the log *before* the session is created.

Orin6 has `present == online`, so this never reproduces there. Judge it on a robot
whose power mode parks cores.

#### Where this layer sits, and why that matters more than it looks

The onnxruntime step is the **last of the dependency layers**, below everything
ASR/TTS/VOP/OCR share and above only the `COPY` of application code.

That placement is the whole safety story. A layer inserted higher up invalidates
the Docker cache for every layer below it, and several of those install
**unpinned** — `ultralytics` and `phonemizer` have no version constraint — so they
silently re-resolve to whatever is newest on the next build. An earlier revision of
this change put `ORT_VERSION` in `/etc/jetpack.env` near the top of the file, and
that alone moved `ultralytics` 8.4.138 → 8.4.142 in the built image, with nothing
to do with face recognition. Measured by diffing `pip freeze` between the old and
new images.

So the rule for this layer: it must stay below every shared layer, and anything it
needs must be resolved *in* it. Against `main` the Dockerfile diff is a single
additive hunk — 74 lines added, **0 removed** — so no shared layer's inputs change
and no other algorithm's dependencies can drift because of it.

#### Its dependencies are deliberately not installed

`pip install --no-deps`. A resolved install pulls in protobuf, coloredlogs,
humanfriendly and flatbuffers — and measured on the jp6.1 image, **protobuf was
not present at all** beforehand, so a plain install would introduce protobuf
7.36.1 into an image where rapidocr, TensorRT and ROS2 all live. None of it is
needed for inference: verified on Tianyi that with `--no-deps` and none of those
packages present, `InferenceSession` builds and `run()` returns all 9 outputs. So
this layer adds exactly one package and touches nothing else — numpy included,
which matters because the torch/cv2/rapidocr stack is built against a specific
numpy C-ABI.

The `insightface` package is deliberately not a dependency: it wants onnx,
scikit-image, scikit-learn and Cython to wrap ~200 lines of pre/post-processing.
Those 200 lines are in `plugins/face_runtime.py` instead — the SCRFD decode there
(strides 8/16/32, 2 anchors per location, distance-coded boxes and keypoints) is a
wire format, not a design choice, and was verified against the real model:
`det_500m.onnx` has 9 outputs and `12800 = 80x80x2` rows for stride 8 at 640px.

Alignment uses an explicit Umeyama similarity fit, **not**
`cv2.estimateAffinePartial2D`: that runs RANSAC/LMEDS, and on exactly five
correspondences a robust estimator can discard a point and return a different
transform run to run, which would make one photo produce different embeddings.

#### Re-hosting the models

`FACE_MODEL_BASE` points at COS, not the upstream GitHub release: the release URL
redirects to a signed, expiring `release-assets.githubusercontent` URL that cannot
be pinned, and the robots have no reliable route to GitHub. To refresh, download
`buffalo_sc.zip` from the insightface v0.7 release, upload the two `.onnx` files to
`public/face/buffalo_sc/` with credentials from `resource-center/deploy/values.env`
(`prisma/articles/upload-figs.js` cannot do it — it only accepts image extensions
and forces its own key shape), then re-download from COS and paste the *verified*
`size`/`sha256` into `FACE_MODEL_FILES`.

### The person record: `name` vs `profile`

The split is about **what gets published**:

| field | type | in the per-frame payload | what it is |
|---|---|---|---|
| `id` | str | yes | `p-N` (named) or `unknown-N` |
| `name` | str | yes | 姓名, structured. A non-blank name is what makes an entry *named* |
| `profile` | object | yes | 非结构化: gender, appearance, notes, tags - whatever the operator wants the agent to have in context |
| `registered_at` | float | no | when the identity was created |
| `last_seen_at` | float | no | most recent sighting |

The timestamps are deliberately out of the payload: they change every frame (or
never), and "when was this person around" is a question the **visit log** answers
properly and a per-frame field cannot.

A `profile` sent as a plain string is stored as `{"note": ...}` rather than
rejected - an LLM will occasionally send prose where an object is expected.

A database written by the pre-split build migrates on load: the old free-text
`profile` becomes `name`, the old structured `meta` becomes `profile`, and
`created_at` becomes `registered_at`. `persons.json` carries `version: 2`.

### 访问记录表 - the visit log

`visits.jsonl`, one line per **visit**, where a visit is a contiguous presence:

```json
{"person_id": "p-1", "name": "小王", "first_seen": 1788780000.0,
 "last_seen": 1788783600.0, "sightings": 3417, "topic": "/cam/rgb"}
```

`list_visits` takes `person_id`, `since`, `until`, `limit`, `offset`. `since` and
`until` accept epoch seconds **or** ISO-8601 (`2026-09-07T15:00`), because an
operator asking "who was here at 3pm" thinks in wall-clock and an LLM will send a
string. Filtering is by **overlap, not containment**: somebody present
14:50-15:10 *was* there at 15:00, and a query for 15:00-15:05 has to say so.

Three design points that are not obvious:

**One row per visit, not per frame.** At the default 1 detection/second a
per-frame log would be 86 400 writes a day per person onto eMMC, carrying no
information a visit does not.

**`visit_gap_s` is 10 minutes.** A visit closes only after the person has been
unseen that long. A short gap would fragment one afternoon in the office into
dozens of rows every time somebody turned their head.

**Open visits are checkpointed, because that 10-minute gap is a data-loss
window.** Somebody present all afternoon is a single visit held in memory for
hours, and a robot that loses power would lose the whole record - not just the
tail. So `visits-open.json` is rewritten at most every `visit_checkpoint_s`
(60 s), bounding the loss to a minute of `last_seen`/`sightings`. On startup a
checkpointed visit is **resumed** if its subject was seen recently, or closed and
appended if they left while the process was down. The checkpoint is written
*after* the append when a visit closes, so a crash in between replays a closed
visit rather than dropping it - a duplicate is recoverable, a loss is not.
Stopping an instance force-closes its open visits for the same reason.

`list_visits` also returns visits still in progress, flagged `open: true`, so the
10-minute close latency does not hide who is in the room right now.

### Actions: register vs recognize

Symmetric by suffix, and the two halves differ in more than direction:

| | photo | stream | corpus |
|---|---|---|---|
| **register** (writes) | `register_by_photo` | `register_by_stream` | `register_by_corpus` |
| **recognize** (read-only) | `recognize_by_photo` | `recognize_by_stream` | - |

`recognize_*` is **read-only**: it neither auto-enrols the stranger it failed to
match nor records a sighting. Asking "who is this" must not quietly change the
answer. It also does **not** apply `subject_dominance`: that gate exists because
enrolment has to resolve to exactly one person, whereas a query can just report
everyone it sees. So a two-person photo is answered with two identities by
`recognize_by_photo` and refused with `ambiguous_subject` by `register_by_photo` -
same input, opposite handling, both correct.

An unmatched face comes back as `person_id: null` with a `best_score`, which is
the number an operator needs to decide whether `match_threshold` is too strict.

`recognize_by_stream` scans the last **1 s** by default (not the 3 s enrolment
window - the question is "who is in front of me now"), reports each person once at
their best score across those frames, and falls back to reporting the clearest
unidentified face so the answer is "someone I do not know" rather than "nobody".

### Identity database

`plugins/face_db.py`, default `/models/face_db` — `/models` is the only host-mounted
writable path this container has (`deploy/service.yml`).

Two files, and **`persons.json` is the commit point**: it names the
`embeddings-<n>.npy` it belongs to, is replaced last, and the superseded matrix is
unlinked only after that succeeds. Writing `embeddings.npy` in place instead means
two `os.replace` calls with a window where the row count and the owner list
disagree — which silently misattributes every identity after the missing row.

Matching is one `matrix @ embedding`: both sides are L2-normalised, so the dot
product *is* the cosine and no `sklearn` is needed. Rows are per **sample**, and a
person's score is their best sample — a mean-vector centroid would blur the pose
variation that several enrolment photos exist to capture, and could push a real
match below threshold when a second photo is added.

Ids are `p-N` for named people and `unknown-N` for strangers, from monotonic
counters that never decrease: a retired id must not resolve to a different person
later, because it may already be on the activity stream and in the agent's history.

### Recognising

Published payload (`{input_topic}/face`):

```json
{"ts": 1788777509.34, "count": 1, "latency_ms": 28,
 "faces": [{"person_id": "p-1", "name": "小王", "profile": {"gender": "male"},
            "known": true, "score": 0.61, "bbox": [207, 186, 149, 206],
            "det_score": 0.811, "blur": 1484.2, "min_side_px": 149,
            "quality": "ok"}]}
```

A stranger who clears the quality gate is auto-enrolled and reported as
`unknown-N`, with the **same id on every later sighting and after a restart** —
which is what makes `register_by_stream` able to name them retroactively.

Detection runs at **`detect_fps`** — 检测频率, detections per second, default
**1.0**, fractional allowed (`0.5` = once every two seconds, `0` = every frame the
camera delivers). It is per-instance, so two cameras can run at different
cadences. Expressed as a frequency rather than a minimum interval because that is
what an operator reasons about, and it stays meaningful when the camera's own rate
changes. (`min_interval_ms`, the knob this replaced, is still honoured when
`detect_fps` is absent, so a canvas saved by an earlier build keeps working.)

A face that *fails* the gate is reported with `person_id: null`, `quality: "low"`
and a `reason`, and is neither matched nor enrolled. Matching a blurred 30 px face
is a coin flip, and enrolling one would spend an `unknown-N` slot forever on a
smear that never matches anything again. Relax `min_face_px` / `blur_min` if you
want identities at greater distance.

`match_threshold` defaults to **0.35**. Measured separations on this model, same
photo transformed: same face at half resolution **0.983**, same face +35
brightness **0.979**, a different person **-0.108**. Real same-person /
different-photo scores sit well below the first two, so **tune this against your own
faces and record what you saw** — 0.35 is a starting point, not a measurement.

### Registering, and why it fails

Any channel can fail for mundane physical reasons, so all three return
`{"ok": false, "reason": ..., "detail": ...}` rather than raising:

| `reason` | Condition |
|----------|-----------|
| `no_face` | no detection anywhere in the input |
| `low_quality` | best face fails `det_thresh` / `min_face_px` / `blur_min`; the detail names each gate with the measured value |
| `ambiguous_subject` | ≥2 faces pass the gate and the primary is not `subject_dominance`x the runner-up; every candidate's bbox and score is returned |
| `no_frames` | no running instance, or nothing in the window |
| `bad_input` | undecodable image, unreachable URL, unreadable package, path outside `image_roots` |

A real `low_quality` detail, from the group photo in the end-to-end check:

```
no clear face: face 53 px < 64 px (move closer); sharpness 41.4 < 60.0 (hold still)
```

Prominence is area discounted 40% for being off-centre — pure area picks the
bystander standing nearer the lens edge, pure centrality picks a distant face
framed dead-on.

| Action | Input |
|--------|-------|
| `register_by_photo` | `image_path` — uploaded from the card, or written to `/uploads` (see below) — plus `name` and `profile` |
| `register_by_stream` | `instance_id` + `name`; analyses **every frame in the last `enroll_window_s`** (default 3 s, up to `enroll_max_analyzed` of them, newest first) |
| `register_by_corpus` | `package`: a directory, `.zip` or `.tar.gz`, by path or URL |

#### Getting a photo *into* this container

Not obvious, and it produced two real failures before being fixed.

perception and agent-core share **no filesystem**: agent-core mounts
`/opt/phanthy-motus` and `/opt/phanthy-motus/data`, perception mounts `/dev` and
`/opt/embodied/models`. The intersection is empty, and each container's `/tmp`
and `/work` is its own. So an LLM that downloads a photo inside agent-core and
passes `image_path: /work/daiwen.jpg` names a file that genuinely exists — just
not here. That was failure one. Its next attempt, `image_b64`, failed too: the
photo was 43 800 base64 characters, which does not survive being carried through
a model's own context, so what arrived was truncated.

**base64 input has been removed**, and files now move through a proxy:

```
browser / LLM ──upload──▶ agent-core  POST /api/mcp/{mcp_id}/file/upload
                              │  looks the target's address up in the MCP
                              │  registry, streams the body on in 1 MiB chunks
                              ▼
                          perception  POST /file/upload
                              │  writes to file_intake.dir (/models/uploads)
                              ▼
                     ◀──reply── {"path": "/models/uploads/2026-09-08/alice.jpg"}
                          that path is used verbatim as image_path
```

The reply carries the path **in the receiving container's own namespace**, so
there is one viewpoint and nothing to translate. No shared mount is involved,
which also means no container has to be recreated to enable it.

The address comes from the MCP registry — every service reports `url` when it
registers — so this one route covers perception, actucore and every driver even
though their ports all differ. `utils/file_intake.py` is stdlib-only for the same
reason: the drivers run a bare `ThreadingHTTPServer`, so they can adopt the
identical endpoint in about five lines.

On the card, `image_path` is declared `"format": "file"` with
`"uploadTo": "mcp"`, which is what routes the canvas file picker through the
proxy instead of agent-core's own `/api/file/upload`. Omitting `uploadTo` keeps
the old behaviour, which is correct for a tool agent-core serves itself
(`remote_image`, `remote_audio`).

Details worth knowing about the receiving end:

* **Size is capped while writing**, not after, so an oversized body is never
  fully committed to disk or held in memory. Files land under a `.part` name and
  are renamed, so a reader listing the directory never sees half a file.
* **Filenames are reduced to one component** and keep CJK characters — the
  people using this name their files in Chinese — after NFC normalisation, so a
  macOS upload and a Linux one produce the same name rather than two files that
  look identical.
* **`subdir` is refused rather than sanitised** if it contains a separator.
  Sanitising turned `../escape` into `.._escape`, a valid name, so the write
  succeeded somewhere the caller did not ask for and nothing said so.
* **Uploads are pruned after `retention_days`** (7). They are a transfer buffer,
  not storage: enrolment keeps the *embedding* and never the photo, so without
  this the directory grows forever on a 57 GB eMMC.
* **Auth is by reachability, not a secret.** The endpoint honours `ACCESS_TOKEN`
  if the service has one, but perception is not given one today (verified on
  Tianyi) — and the port already serves `tools/call`, so anything that can reach
  it can already drive the plugin.

Verified on Tianyi: 200 KB of random binary round-trips byte-identically through
the real endpoint with a Chinese filename, returning `/…/2026-09-08/戴文渊.jpg`.

Passing `image_b64` now returns a `bad_input` naming the mechanism that works,
and a path outside `image_roots` says to upload through the proxy or use
`register_by_url` — an LLM told only "cannot read" retries with another
invisible path, which is exactly what happened.

`register_by_stream` averages the agreeing frames rather than trusting one
grab, and refuses with `ambiguous_subject` when fewer than half the usable frames
agree with each other — two people taking turns being the dominant face would
otherwise be enrolled as one identity matching neither.

Enrolment never silently duplicates. A new face matching an existing **named**
person is added as another sample (`merged: true`); matching an **`unknown-N`**
promotes that entry **keeping its id** (`promoted: true`), so earlier sightings stay
attributable.

#### Batch package layout

Images plus an optional `manifest.json`, in either shape:

```json
[{"file": "alice.jpg", "name": "Alice from ops", "person": "alice", "profile": {"badge": "A7"}}]
{"alice.jpg": "Alice from ops"}
```

Fallbacks in order: a sidecar `alice.json` / `alice.txt`, then the filename stem. A
shared `person` key merges several photos into one identity. Archive members that
are absolute, contain `..`, or are not regular files are skipped and logged.

The result carries **one record per photo**, so a 40-person batch says exactly which
people registered and why each of the rest did not:

```json
{"ok": true, "total": 40, "registered": 37, "failed": 3,
 "results": [{"file": "alice.jpg", "ok": true, "person_id": "p-9"},
             {"file": "bob.jpg", "ok": false, "reason": "low_quality", "detail": "..."},
             {"file": "team.jpg", "ok": false, "reason": "ambiguous_subject", "candidates": [...]}]}
```

Synchronous, capped at `max_batch` (200); over the cap it returns `bad_input` naming
the count rather than hanging the MCP client.

### Roster CRUD

`list_persons` (`named` = all/named/unknown, `query` over id+name+profile,
`limit`, `offset`), `get_person`, `update_person` (`name`, `profile`,
`profile_delete[]`, `merge` — default merges, `merge: false` replaces), `forget`
(or `named: "unknown"` to clear every anonymous entry), and `list_visits`.
Setting a non-blank `name` on an `unknown-N` names it in place, keeping the id.
Reads never return embeddings.

`unknown_capacity` (default 500, editable on the card) bounds automatic enrolment
only; lowering it evicts the excess immediately, oldest `last_seen_at` first, and
reports how many went. Named people are never candidates.

### Tool-name dispatch

`face_recognition` is the first `PREFIX` containing an underscore.
`PerceptionBundle.dispatch`/`owns` used to split on the first `_` and compare that
to `PREFIX`, which made such a name undispatchable — it resolved to a plugin called
`face`, matched nothing, and reported the tool as unknown. Both now go through
`_plugin_for`, which matches the **longest** prefix; `tests/test_bundle_dispatch.py`
pins that and that the two functions can never disagree.

### Verifying

The pytest suite fakes the analyzer — a host-side suite must not need models or
onnxruntime — and covers the lifecycle, all five reason codes, `unknown-N`
stability and promotion, id retirement, capacity eviction, profile CRUD, the
visit log (sessionisation, checkpoint recovery, overlap queries), batch
per-item results and zip traversal (`tests/test_face_plugin.py`,
`tests/test_face_db.py`).

What that cannot cover is the decode itself, so it was checked separately against
the real models: a known similarity transform recovered to 4e-6, all 5 landmarks
inside their bbox on real photos, and the identity separations quoted above. On
hardware, confirm `info` goes `loading → ready`, that `{topic}/face` publishes,
and — because two ONNX Runtimes share the process — that an ASR utterance is still
transcribed and TTS still speaks with the card running.

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
