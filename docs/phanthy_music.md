# phanthy_music card

One independent Perception card provides catalogue search and background music
playback. Agent Core, ASR, deployment manifests and drivers require no changes.
The remote catalogue implements [motus.music/1](music-catalog.md); the card does
not generate songs, run another model or maintain a local music library.

## Card configuration

Add **phanthy_music** to the canvas and fill in its sidebar settings:

| Field | Meaning |
|---|---|
| `catalogue_type` | `none` (default), `mock`, or `motus_music` |
| `endpoint` | Catalogue's HTTPS base address, supplied by the operator |
| `api_key` | Catalogue credential, supplied through the card's password field |
| `timeout_ms` | Catalogue request budget including capabilities/retries; default 3000 |
| `volume` | Music volume 0–100; default 50 |
| `duck_gain` | Music's relative level during TTS; default 0.15 |

The card receives `endpoint` and `api_key` through the existing `config` action.
There are no music credential environment variables or deployment changes.
Changing either value replaces the provider/cache and cancels pending playback
lookups. Setting the key to an empty string clears it. `none` reports
`not_configured`; `mock` has 24 fictional metadata records and no playable audio.

`api_key` has `format: password`, and `endpoint` has `x-sensitive: true`.
The existing Solution exporter clears both. The input schema shown to the model
contains neither credential nor configuration actions. Status/config replies
never return the key or endpoint; music MCP logs omit payloads. The HTTPS
provider sends the key only in its Authorization header and rejects redirects.
Real endpoints, keys and song assets must not be committed.

**Storage limitation:** the existing card framework saves configuration in the
local SQLite database as plaintext. A password input and Solution redaction do
not encrypt that database. This card uses the existing storage behaviour; it
does not change Agent Core or introduce a credential vault.

## Actions

| Model action | Behaviour |
|---|---|
| `search(...)` | Find songs and return metadata; no playback |
| `play_by_id(track_id)` | Fetch current details/URL and play the selected song |
| `play_by_genre(genre, ...)` | Search with `top_k=1`, then play the returned song |
| `pause` / `resume` | Pause/resume music; TTS continues |
| `interrupt` | Cancel pending playback lookup and stop the current song |
| `set_volume(volume)` | Set music volume; does not change speech volume |
| `status` | Return playback state, position, attribution and error |

`search` accepts query, genre, language, vocal, tags, artist, top_k, exclude_ids,
duration_min_s and duration_max_s. Array-like filters are comma-separated strings.
`play_by_genre` requires genre and accepts the same optional filters except top_k.
It uses the service's first result and returns the selected `track`, `relaxed`
and matching information. If the service relaxes a requested condition, the
caller must say so; it must not claim an exact match. Empty/error results do not
start playback or interrupt an existing song. No global last-user search is stored.

Example card calls (fictional ID):

```json
{"action":"search","query":"下雨天 慵懒","vocal":"female"}
{"action":"play_by_id","track_id":"fictional-track-id"}
{"action":"play_by_genre","genre":"pop","language":"zh","exclude_ids":"fictional-track-id"}
{"action":"interrupt"}
```

The existing `x-action-params` mechanism exposes names such as
`mcp__<perception-id>__phanthy_music__play_by_genre`; MCP still calls the single
`phanthy_music` tool with `arguments.action`. Lifecycle (`start`, `stop`, `info`,
`config`) and internal hooks are handled by the card but omitted from that action
map. Existing Agent Core permissions and subagent policy apply without exceptions.

## Canvas and audio

The bundle advertises the card when `plugins.phanthy_music.enabled` is true
(the shipped default). Connect decision_core's execution output to this card.
Search works while the audio node is idle. To play, start the canvas project.
Starting opens the PCM path without playing anything automatically.

For conversation during playback, connect **TTS → phanthy_music → Speaker**, with
one Speaker input. All audio is mono S16LE at 16 kHz (`audio/pcm-16k`); the existing
output topic is `/perception/music/audio`. Leave the card's input empty for
music-only playback. One music card is supported; feedback or multiple speech
inputs are rejected. Existing AudioChunk/Speaker drivers need no new API.

TTS PCM automatically reduces the music level and restores it when speech ends.
Speech retains its original level, mixed samples saturate safely, and gain
changes ramp. Existing speech-interrupt hooks discard speech without stopping
music. User microphone/VAD end detection is not added by this card; ASR is
unchanged. Say “停歌” using `interrupt`; canvas `stop` closes the entire PCM path.

Playback returns a background `playback_id`, with buffering/playing/paused/
completed/cancelled/error states. It does not hold the mouth resource or wait
for the whole song before allowing further conversation. Position and completion
refer to this card's emitted PCM, not confirmed hardware output. Driver buffers,
stop latency, AEC and microphone self-triggering require robot testing.

## Limits and failures

Catalogue I/O runs outside the player/lifecycle lock. Only one catalogue request
per card is admitted; concurrent calls receive `busy`. Stop, interrupt, replacement
play and catalogue reconfiguration invalidate pending playback, so a late response
cannot restart a cancelled song. Lookup errors preserve current playback.

Track details supply format, expiry and usage. Expired URLs are rejected; retry
`play_by_id` to get fresh details. `personal_playback` audio travels through HTTPS
and ffmpeg pipes to bounded memory and PCM output. No audio files, persistent
caches or chat attachments are created. ffmpeg is already in the Perception image.

Audio URLs/redirects require verified HTTPS and public IPs, with DNS addresses
validated and pinned. Audio downloads carry no catalogue credential and bypass
environment proxies. When all system DNS answers are proxy fake-IP addresses
(`198.18.0.0/15`), bounded Cloudflare/Google HTTPS DNS lookup can resolve a public
IPv4 address. Only the hostname is sent, never signed paths or keys. Private,
literal fake-IP and mixed public/private answers are refused.

Limits include 50 MiB / 30 minutes of audio, a two-second decoded queue and a
20-second speech queue. Cancellation stops ffmpeg; an outstanding socket operation
may finish at its timeout. Missing subscribers and stalled streams become visible
errors. A subscriber or successful local PCM write is not robot acceptance.

## Verification

```bash
cd perception
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_music.py tests/test_music_catalog.py tests/test_phanthy_music.py tests/test_bundle_dispatch.py -q
```

Offline unittest cases use ROS stubs and fictional metadata. They cover search,
ID/genre playback, empty/error/relaxed responses, cancellation during lookups,
credential updates/clearing, log redaction, PCM mixing and transport validation.
Generated WAV/MP3 checks use real ffmpeg and explicitly skip if it is absent.
For deployment acceptance, configure the real service through the card and test
both playback actions, pause/resume, TTS priority and interruption. Robot audible
output, ASR and driver behaviour must be verified separately.

Earlier unmerged `music` / `phanthy-music` prototypes should be replaced with
`phanthy_music`, rewired and configured in the sidebar. No published-card or
production database migration is included.
