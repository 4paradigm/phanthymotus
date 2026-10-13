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

## Card configuration

| Key | Default | Meaning |
|---|---|---|
| `catalogue_type` | `motus_music` | `mock` is offline metadata only (no audio); `none` disables lookups. |
| `timeout_ms` | `3000` | Whole-request budget, including the capability query and the one retry. |
| `volume` | `50` | Music volume. Never affects speech. |
| `continuous` | `true` | Station mode: when a track drains, automatically play the next one matching the same criteria. |

`duck_gain` moved to `audio_mixer`, where the ducking happens.

## Actions

| Action | Takes | Does |
|---|---|---|
| `list_genres` | — | The catalogue's own genre / language / vocal / tag vocabularies. **Call this before guessing a code.** Says whether the values came from the catalogue or from the built-in fallback. |
| `play` | `genre`, `query`, `language`, `vocal`, `tags`, `artist`, `duration_min_s`, `duration_max_s`, `track_id` | Picks one track and starts playing. No criteria at all is valid — "放点歌" means anything. Returns as soon as fetching starts, not when the song ends. |
| `next` | — | Next track. Takes no arguments: it reuses the last `play`'s criteria and excludes what already played. |
| `previous` | — | Back one track, from the beginning. Costs one catalogue call, because signed audio URLs expire. |
| `pause` / `resume` | — | Pause keeps the decoded buffer, so resuming continues rather than skipping. |
| `stop_music` | — | Stops the music. Does not affect speech. |
| `now_playing` | — | Title, artist, genre, position, remaining, volume, station criteria. Call this before telling a user what is on. |
| `set_volume` | `volume` | 0–100. Music only. |
| `search` | filters + `top_k` | Answers "is there any…" without playing. Works with the card idle. |

`start`, `stop`, `info` and `config` are dispatchable but are **not** in the
declared action enum, so the model never sees them. `present_to_llm` only
filters those four for `processor` cards, and this is an `actuator` — leaving
them visible would offer the model a `stop` that tears the ROS node down, right
next to the `stop_music` it means. That confusion was measured on a rig with
`tts(action="stop")`.

Old spellings still dispatch, undeclared: `interrupt` → `stop_music`, `status` →
`now_playing`, `play_by_id` / `play_by_genre` → `play`.

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
