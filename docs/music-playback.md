# Music playback card

The Perception **music** card plays a public HTTPS mp3/wav URL and optionally
mixes TTS speech into the same PCM output. It is usable without a catalogue:
call `play` with an authorised audio URL. A catalogue client can supply a track's
URL, format, ID, expiry and usage directly. The card contains no catalogue
endpoint, key, search implementation or song assets.

## Canvas setup

Music is advertised by the Perception bundle when `plugins.music.enabled` is
true (the shipped default). Starting the project creates an idle audio path;
it never starts downloading or playing a song automatically.

For conversation during music, wire **TTS → music → Speaker**, all using
`audio/pcm-16k` (mono S16LE, 16 kHz). Remove the old TTS → Speaker wire so Speaker
has one input. Connect decision_core's execution output to music to expose its
actions to the agent. Leave music's data input empty for music-only playback.
The fixed output is `/perception/music/audio`; only one music card is supported.
Connecting its output back to its input or supplying multiple speech inputs is
rejected. Ordinary TTS still passes through when no song is playing.

No driver change is needed for a Speaker that accepts the existing AudioChunk
format. Device-specific stop latency, buffering and echo cancellation still
need validation on the target robot. This card cannot flush audio that a driver
has already queued. Music may trigger an ASR without working echo cancellation;
microphone placement/volume and the existing AEC path must be checked on-device.

## Actions

| Action | Behaviour |
|---|---|
| `start(input_topic?)` | Open the PCM path; optional TTS input |
| `play(url, audio_format?, track_id?, expires_at?, usage?)` | Replace the current song; return a background session immediately |
| `pause` / `resume` | Pause/resume music; TTS continues |
| `interrupt` | Stop the current song and discard this card's pending music |
| `set_volume(volume)` | Music volume 0–100; never changes TTS volume |
| `status` / `info` | Session, source position, attribution and bounded error |
| `stop` | Close the entire card, including speech passthrough |

Say “停歌” by calling `interrupt`, not `stop`, to keep conversation audio working.
`play` defaults to mp3; explicitly pass `audio_format: wav` for WAV. Forward the
track's `usage` (`scope: personal_playback`, `attribution`) and `expires_at` when
provided. A passed/invalid expiry is rejected before downloading. On expiry or
HTTP 401/403/404, refresh the URL by track ID through the catalogue, then call
`play` again. Credentials for the catalogue never go to the audio host.

Example (replace this fictional URL with an authorised deployment URL):

```json
{"action":"play","url":"https://audio.example/song.mp3","audio_format":"mp3","track_id":"example-track","expires_at":null,"usage":{"scope":"personal_playback","attribution":"Example artist"}}
```

This is a **background session**, with its own `playback_id` and states
`buffering`, `playing`, `paused`, `completed`, `cancelled`, `error`. The MCP action
does not hold the mouth resource or create a pending full-song ACP action, so
`finish()` and subsequent dialogue need not wait minutes for the song to end.
`position_s` is PCM emitted by this card, not a physical speaker clock.
`completed` means the source has drained; a downstream buffer may still contain
audio. It must not be presented as confirmed hardware completion.

## Speech priority

TTS PCM arriving at the input automatically reduces music to `duck_gain`
(default 0.15) of its configured volume; speech keeps its original level.
The mixed samples saturate at S16LE limits. Gain ramps smooth volume changes.
The speech end marker is consumed internally while music continues; only the
combined output's end marker reaches Speaker.

The ASR VAD now emits both speech start and speech end. Its existing hook bridge
fires `on_hearing` / `on_hearing_end`, mapped to music `duck` / `unduck`. The end
hook restores music after a short tail. A 60-second lease prevents a lost end
event from muting music forever. `on_interrupt_all` and `on_interrupt_speak`
discard interrupted speech without cancelling the song; an explicit
`on_interrupt_music` stops music. Hooks do not auto-start idle cards.

## Limits and failures

Audio travels through HTTPS → ffmpeg pipes → bounded memory → ROS. There are no
audio files, persistent caches, uploads or background catalogue crawls. ffmpeg
is already installed in the Perception image. It sees only mp3/wav bytes on stdin
with the `pipe` protocol allowed; it cannot fetch nested media URLs.

URLs and every redirect require HTTPS without embedded credentials. Connections
use TLS verification and validated DNS results pinned to public IPs; private,
loopback and link-local addresses are refused. Environment proxies are bypassed
for audio. Signed URLs and response bodies are not copied into player errors.

Some system proxy/TUN configurations synthesize `198.18.0.0/15` fake-IP answers.
If **every** address for an audio hostname is in that range, the client asks the
fixed [Cloudflare HTTPS DNS endpoint](https://developers.cloudflare.com/1.1.1.1/encryption/dns-over-https/make-api-requests/dns-json/)
for public IPv4 addresses and connects only to a revalidated numeric address.
On a transport failure, HTTP 429 or 5xx, it can try the fixed
[Google HTTPS DNS endpoint](https://developers.google.com/speed/public-dns/docs/doh/json)
once. Invalid/private DNS answers and redirects fail without trying a different
resolver.
Normal public DNS results never trigger this extra lookup. Literal fake IPs,
other private addresses and mixed public/private answers remain rejected; the
client never connects the audio request to the fake IP. Redirect destinations go
through the same checks, with the original hostname retained for TLS/SNI.

The DNS request contains only the hostname (no audio path, signed query or API
key), disables EDNS client-subnet forwarding, requires verified HTTPS, refuses
redirects and bounds response size to 16 KiB. Its network timeout is capped at
3 seconds per resolver within an 8-second DNS budget and the connection budget.
Resolver bootstrap uses system DNS and certificate/hostname verification;
no machine-wide DNS/proxy settings are changed. If the resolver is unreachable
or returns no valid public IPv4 address, playback fails with an actionable DNS
message. Configure real DNS or exclude audio domains from the proxy's fake-IP
mode in that environment; private audio access is never an automatic fallback.

Downloads are limited to 50 MiB / 30 minutes of decoded PCM, with 10-second HTTP
timeouts, a 2-second decoded queue and a 20-second speech queue. Pausing applies
backpressure. Cancelling kills ffmpeg and detaches the old session immediately;
an outstanding DNS/socket operation may finish later. At most two retiring
workers are allowed before another skip is refused. A stalled stream or a missing
PCM subscriber becomes a visible error. An inspector/browser subscription also
counts as a reader; a reader does not prove a physical Speaker is connected.

Disable `plugins.music.enabled` and restore TTS → Speaker to remove this feature.
No database migration or new driver API is involved.

## Validation and five-minute demo

```bash
cd perception
python3 -m unittest discover -s tests -p test_music.py -v
```

Offline tests use the repository's ROS stubs and fake clocks. They cover actual
MCP dispatch/lifecycle, speech passthrough, ducking/restoration, pause/resume,
interrupt/replacement, bounded workers/buffers, expiry and transport failures.
Media checks generate WAV and MP3 in memory and decode using real ffmpeg; they
skip explicitly if ffmpeg is absent. No real songs, network, GPU or hardware are
needed. Stubs do not establish ROS interoperability or physical audio quality.

On a robot: start the wired project, play authorised audio, query status, pause,
speak, resume, speak over the music, then stop the song. Verify audible ducking
and recovery, normal ASR/TTS, no repeated self-triggering, and actual stop delay.
Remove all output subscribers (including inspectors), confirm the no-subscriber
error, then reconnect/retry. Stop/start the
project repeatedly and verify no old audio returns. Record device and image
versions with those results before claiming hardware acceptance.
