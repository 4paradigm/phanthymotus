# phanthy_music card

A music and ASMR/助眠音频 player, written for the two callers it has: a person
speaking, and the LLM relaying them. It is an **actuator** that emits audio PCM
on its own topic and nothing else — mixing and "duck the music while the robot talks" belong to
[`audio_mixer`](audio-mixer.md).

```
phanthy_music ──► audio_mixer ──► Speaker
TTS           ──►
```

An earlier version of this card *was* the mixer: it took TTS in on its own
input, ducked the song under it and published the combined stream. If you have
a canvas wired `TTS → phanthy_music → Speaker`, `start` will refuse with
`legacy_wiring` and name the fix. That refusal is deliberate — silently ignoring
the input would leave a **mute robot reporting 已就绪**, because TTS would be
publishing into nothing.

## Two kinds of content, two sets of actions

The catalogue holds songs *and* ASMR behind one endpoint and one key (the
protocol side is in [`music-catalog.md`](music-catalog.md)). Here that is two
pairs of actions, not a `kind` parameter:

| | Songs | ASMR |
|---|---|---|
| Play | `play` | `play_asmr` |
| Look without playing | `search` | `search_asmr` |
| Filters | `query`, `genre`, `language`, `artist`, `tags` (+`vocal` on `search`) | `query`, `category`, `tags` |
| Shared | `next`, `previous`, `pause`/`resume`, `stop_music`, `now_playing`, `set_volume`, `set_autoplay`, `list_genres` | same |

**Why separate actions rather than `play(kind=…)`.** An ASMR entry has no genre,
no voice type and no artist, and the catalogue returns 422 when those are sent
with it. A single `play` carrying all eight filters would therefore be offering
the model five distinct ways to earn a 422 — and the one thing it cannot do is
learn the rule from the parameter list. Because `_coerce_filters` reads the
parameter set from the *action* rather than from `args`, a `genre` the model
volunteers on `play_asmr` is dropped before the request is built; it never
reaches the network and never fails.

`category` is one of `bedtime_story` 睡前故事, `meditation` 冥想引导,
`companion` 情感陪伴, `chat` 生活化闲聊, `poetry` 经典诗文, `trigger` 触发音
(no voice: chewing, friction, kneading, tearing, liquid). **Leaving it empty is
the right answer when the user did not name a type** — 「来点助眠的」means any of
them, and the catalogue picks. Looking for a specific object's sound is
`category=trigger` plus the object in `query` (薯片, 纸).

`language` is deliberately absent from the ASMR actions even though the protocol
accepts it for both kinds: every ASMR entry in the catalogue is Chinese, so the
parameter has no reachable effect and its only cost is a slot in the model's
attention.

### `next` stays on the side the user asked for

The station records `kind` along with the other criteria, so 「换一个」after
`play_asmr` fetches another ASMR entry rather than drifting into songs. Crossing
over is an explicit `play` or `play_asmr`.

### ASMR is a playback session like any other

`play_asmr` is in `x-completion.actions` beside `play`. A two-minute bedtime
story needs a pending exactly as a song does — without one, `finish` lands in
the middle of it.

### `now_playing` must not call a bedtime story a song

The protocol shares one track shape between the two kinds, so an ASMR entry
arrives with `genre: 'other'` and an empty artist. Reported verbatim, that is
what makes a robot announce 「为你播放 other 曲风的歌」over a bedtime story — so
`now_playing` blanks `genre`/`artist`/`vocal` for `kind: 'asmr'` and reports
`kind` and `category` instead, which is what the model should speak from.

### Unknown categories, like unknown genres, are not errors

An unrecognised `category` is moved into `query` as free text and reported in
`category_as_query`, exactly as an unknown `genre` becomes `genre_as_query`. So
「想听嚼冰块的声音」gets an answer rather than a vocabulary error. Each reply also
carries the vocabulary that applies to its own kind — `available_categories` for
ASMR, `available_genres` for songs, never both. Handing an ASMR reply a genre
list is precisely how a model learns to put `genre` on the next ASMR call.

`list_genres` returns both vocabularies (`kinds`, `asmr_categories`, plus the
genre/language/vocal lists and `asmr_supported`), from the same 24 h capability
cache and with no extra network path.

## Source visibility of the catalogue connection

**The catalogue address and API key are fixed in this card's source.** Anyone who
can read the repository or pull the image can read them. Hiding them from the
card's UI is not secrecy. Rotating the key means editing the card and
redeploying. They are deliberately absent from `configSchema`, from
`get_tools()` and from every reply, and `config` refuses any attempt to override
them — but that is about not *leaking them further*, not about keeping them.

## 卡片配置：没有

这张卡没有任何设置项，`configSchema` 被整个去掉了。原来有四项，没有一项经得起
「谁比默认值更有资格回答这个问题」的追问：

| 原设置 | 现在 | 为什么不是配置 |
|---|---|---|
| `catalogue_type` | 固定 `motus_music` | 曲库连接内置在卡片源码里，没得选 |
| `timeout_ms` | 固定 3000 ms | 跟曲库说话的实现细节，不是产品问题 |
| `volume` | `set_volume` 动作 | 听的人当场说的，画布上预设的人不知道你想放多大声 |
| `continuous` | `set_autoplay` 动作，**默认关** | 同上。而且「自己一直放下去」是个决定，该被明确要求 |

后两项之所以曾经是设置，是因为它们是**状态**，而卡片上的状态看起来很像配置。

`duck_gain` 在 [`audio_mixer`](audio-mixer.md) 那边 —— 压低音乐是混音的事。

## Actions

| Action | Takes | Does |
|---|---|---|
| `list_genres` | — | The catalogue's own genre / language / vocal / tag vocabularies **and the ASMR categories**. **Call this before guessing a code.** Says whether the values came from the catalogue or from the built-in fallback. |
| `play` | `query`, `genre`, `language`, `artist`, `tags` | Picks one **song** and starts playing. No criteria at all is valid — "放点歌" means anything. Returns as soon as fetching starts, not when the song ends. |
| `play_asmr` | `query`, `category`, `tags` | Picks one **ASMR** entry and starts playing. Where 助眠／哄睡／睡不着／白噪音／放松／睡前故事／冥想／陪我聊聊／读首诗／咀嚼声 go. No criteria at all is valid, and is the right call when no type was named. |
| `next` | — | Next track or entry. Takes no arguments: it reuses the last `play`/`play_asmr` criteria — **including which kind** — and excludes what already played. |
| `previous` | — | Back one track, from the beginning. Costs one catalogue call, because signed audio URLs expire. |
| `pause` / `resume` | — | Pause keeps the decoded buffer, so resuming continues rather than skipping. |
| `stop_music` | — | Stops the music. Does not affect speech. |
| `now_playing` | — | Title, artist, genre, `kind`, `category`, position, remaining, volume, station criteria. Call this before telling a user what is on. For `kind: 'asmr'` there is no artist and no genre — say the category instead. |
| `set_volume` | `volume` | 0–100。只管音乐。 |
| `set_autoplay` | `autoplay` | 一首放完要不要自动接下一首。**默认关**：放完就停。只接受 true/false —— 一个说不清要开还是要关的调用，猜错的后果是歌自己一直放下去。 |
| `search` | `query`, `genre`, `language`, `vocal`, `tags`, `artist`, `top_k` | Answers "is there any **song**…" without playing. Keeps the filters `play` drops. Works with the card idle. |
| `search_asmr` | `query`, `category`, `tags`, `top_k` | Answers "is there any **ASMR**…" without playing, for when a few options should be offered before one is picked. |

`start`, `stop`, `info` and `config` are dispatchable but are **not** in the
declared action enum, so the model never sees them. `present_to_llm` only
filters those four for `processor` cards, and this is an `actuator` — leaving
them visible would offer the model a `stop` that tears the ROS node down, right
next to the `stop_music` it means. That confusion was measured on a rig with
`tts(action="stop")`.

Old spellings still dispatch, undeclared: `interrupt` → `stop_music`, `status` →
`now_playing`, `play_by_id` / `play_by_genre` → `play`.

### Why `play` takes five filters and not nine

`play` is the voice path; `search` is the precision path. A filter earns a place
on `play` only if it is reachable from something a person says:

| | |
|---|---|
| `query` | 放点伤感的 / 放那首下雨天的歌 |
| `genre` | 放点爵士 |
| `artist` | 来首周杰伦 |
| `language` | 放英文歌 |
| `tags` | 放点适合开车的歌 |

`tags` matters most in a vehicle, and is easy to mistake for a duplicate of
`query`. It is not: the catalogue's `scene` vocabulary is
驾驶/通勤/工作/夜晚/旅行/下雨天, filtering on 驾驶 returns tracks that genuinely
carry that tag, and — unlike `query` — **tags are never relaxed**.

Dropped from `play`, measured against the live catalogue rather than assumed:

- **`duration_min_s` / `duration_max_s`** — the index is real and precise
  (`>=240` returned 241–480 s, `<=120` returned 61–115 s). Nobody says it. They
  were here because `SearchRequest` has them. Gone from `search` too.
- **`vocal`** — works for `male`/`female`, but the one value a driver would ask
  for, 纯音乐 → `instrumental`, is relaxed by the catalogue: the reply comes back
  `relaxed: ['vocal']` with vocal tracks in it. Kept on `search`, where a caller
  can see the relaxation and decide.
- **`track_id`** — by voice, nobody searches and then plays a result by id; a
  named song reaches the same place through `query`. The *dispatch* path is
  retained, so the `play_by_id` alias and any stored canvas call still work.

### Unknown filter values are not errors

If `genre` carries a value the catalogue does not list, it is moved into `query`
as free text and reported back in `genre_as_query`. So "city pop" returns music
rather than `invalid_filter`. The catalogue protocol's strict validation is
unchanged; the demotion happens in the card, above it.

Every `search` / `play` reply also carries `available_genres`, so a caller that
used a wrong code can see the right ones without a second call.

## Errors

`error` is always either absent or a dict with a `code`, plus a `hint` where
there is something actionable to do. Common codes: `not_found` (loosen the
filters), `not_playable` (metadata only, pick another), `busy` (a catalogue call
is already in flight — retry), `invalid_filter` (call `list_genres`),
`legacy_wiring` (insert `audio_mixer`), `not_configured`.

## Playback completion

`play` returns an `action_id` and, when the catalogue reports a duration, an
`expected_duration_s`. The session is declared `background`, so nothing waits on
it: a four-minute track does not block `finish`, nor the agent's next tool call.
Its completion is declared `silent_completion`, so the end of a song is a
receipt rather than a message that wakes a turn.

`next` and `previous` continue the *same* session and do not open a second one.

Completion means the **source** drained; a speaker's own buffer may still hold
audio. `now_playing` says so in `completion_scope`.

## Interrupts

Only `on_interrupt_music` is bound, to `stop_music`. `on_interrupt_all` is
deliberately **not** bound: it fires on every new user turn, and barge-in must
stop the robot talking, not the music. The cost, stated plainly: an emergency
stop does not silence the song. Music is not an actuation hazard, so that is the
intended trade — use `stop_music`.

## Deliberately not supported

- **Seek / scrub.** The decoder is a one-shot forward HTTPS → ffmpeg pipe, and a
  seek needs a fresh signed URL that may already have expired. Nobody scrubs by
  voice.
- **Playlists, collections, favourites.** `motus.music/1` has no playlist
  concept, and `usage.scope: personal_playback` forbids persisting audio.
- **An explicit queue.** The station plus `next` covers the intent with no
  state for the caller to hold or re-describe.
- **`repeat` / `shuffle` as actions.** The station already picks fresh,
  de-duplicated tracks. Single-track repeat would replay one signed URL until it
  expires mid-loop.
- **Time-aligned lyrics.** The protocol carries `lyrics_excerpt` only, no
  timings.
- **Picking an ASMR voice or timbre.** The voice-companionship entries are read
  by several different voices, but the catalogue does not index which, so there
  is nothing to filter on. `vocal` is `null` on every ASMR entry and is not a
  gap the card can fill.
- **An ASMR `language` filter.** Every entry is Chinese; see above.

## Limits and failures

50 MiB or 30 minutes per track; a 2 s decoded buffer; HTTPS only, public
addresses only, redirects refused. One catalogue request at a time — a second
returns `busy` rather than queueing, so `now_playing` and `stop_music` stay
responsive while a lookup is slow. No subscriber on the output topic for 5 s, or
a stalled stream for 15 s, is reported as an error rather than hung on.

Audio is used for the current playback only: never written to disk, never
uploaded as a chat attachment.

## Verification

```bash
cd perception
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  tests/test_music.py tests/test_music_catalog.py tests/test_phanthy_music.py \
  tests/test_audio_mixer.py tests/test_bundle_dispatch.py -q
```

The ASMR path is covered offline by `AsmrContractTests` and `AsmrCardTests`
against mock entries shaped like the real ones (no artist, null vocal,
`genre: other`, empty trigger script). The routing itself lives in the action
and parameter descriptions, so
`test_the_declared_asmr_actions_describe_where_sleep_intent_goes` pins the words
that carry it — a model reaches `play_asmr` only if the card says so, and a
silent edit to those strings is a behaviour change.
