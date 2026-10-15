# phanthy_music card

A music player, written for the two callers it has: a person speaking, and the
LLM relaying them. It is an **actuator** that emits music PCM on its own topic
and nothing else — mixing and "duck the music while the robot talks" belong to
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
| `list_genres` | — | The catalogue's own genre / language / vocal / tag vocabularies. **Call this before guessing a code.** Says whether the values came from the catalogue or from the built-in fallback. |
| `play` | `query`, `genre`, `language`, `artist`, `tags` | Picks one track and starts playing. No criteria at all is valid — "放点歌" means anything. Returns as soon as fetching starts, not when the song ends. |
| `next` | — | Next track. Takes no arguments: it reuses the last `play`'s criteria and excludes what already played. |
| `previous` | — | Back one track, from the beginning. Costs one catalogue call, because signed audio URLs expire. |
| `pause` / `resume` | — | Pause keeps the decoded buffer, so resuming continues rather than skipping. |
| `stop_music` | — | Stops the music. Does not affect speech. |
| `now_playing` | — | Title, artist, genre, position, remaining, volume, station criteria. Call this before telling a user what is on. |
| `set_volume` | `volume` | 0–100。只管音乐。 |
| `set_autoplay` | `autoplay` | 一首放完要不要自动接下一首。**默认关**：放完就停。只接受 true/false —— 一个说不清要开还是要关的调用，猜错的后果是歌自己一直放下去。 |
| `search` | `query`, `genre`, `language`, `vocal`, `tags`, `artist`, `top_k` | Answers "is there any…" without playing. Keeps the filters `play` drops. Works with the card idle. |

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
