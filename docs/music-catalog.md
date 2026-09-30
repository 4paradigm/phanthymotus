# Music catalogue: motus.music/1

PhanthyMotus interprets a user's music request and calls a deterministic external
catalogue. The public repository contains the protocol, a generic HTTPS provider
and fictional metadata; it contains no catalogue, service address or API key.
The machine-readable contract is [motus-music-v1.yaml](openapi/motus-music-v1.yaml).

## Configure and try

Open the decision-core card's configuration. Select **音乐曲库 = mock** to try
search without credentials, network access or audio. Ask “来首中文女声” and inspect
the `MusicSearch` result in Activity: it contains fictional tracks with
`audio: null`, never a fabricated playable link. Ask “换一首”; the agent reuses
the criteria and supplies the prior IDs as `exclude_ids` (at most 50).

Select **motus_music** for a real deployment and enter its HTTPS endpoint, API
key and request budget (default 3000 ms). The client appends `/v1/capabilities`,
`/v1/music/search` or `/v1/music/tracks/{id}` to the endpoint. Credentials go only
in the Authorization header. Redirects are refused, including HTTPS redirects,
so a catalogue cannot forward the key to another origin. Supply the canonical
endpoint directly. TLS verification stays enabled.

Configuration is `desktop_tools.music = {type, endpoint, api_key, timeout_ms}`.
The default is `type: none`; searches then return `not_configured`. API keys are
password fields and the endpoint is `x-sensitive`; both are cleared in shared
Solutions. Store deployment values only in the device configuration, never in
examples, source, logs or a shared Solution. Disable the provider to roll back;
WebSearch and ordinary conversation remain independent.

`MusicSearch` is a built-in read-only tool, not an MCP canvas card. Registered
viewer users and permitted subagents can search; this does not grant playback
permission or widen the untrusted-bot allowlist. A background sensor monitor
still cannot make external searches. Playback requires a separately connected
music card and operator authority.

## Protocol

All requests use HTTPS. JSON responses and search requests carry
`schema: motus.music/1`. The client rejects a different/missing response version.
Optional fields may be added; removing fields or changing their meaning requires
a new protocol version. Every server response includes `request_id`.

* `GET /v1/capabilities`: `provider`, `limits.max_top_k` (1–10),
  `audio.url_ttl_s` (null means non-expiring), `features`, and `filters`.
  Genre/language/vocal options are `{value, label}` arrays. Tags are grouped
  string arrays (`mood`, `scene`, `theme`, `voice`). Actual values come from this
  endpoint; the client does not guess machine codes from human-readable labels.
* `POST /v1/music/search`: optional `query` (up to 200 characters), `filters`,
  `top_k` (default 3, at most the advertised limit) and `exclude_ids` (up to 50).
  Empty criteria select popular songs. There is no pagination or bulk export.
* `GET /v1/music/tracks/{id}`: fetch one track, including a refreshed audio URL.
  `MusicSearch(track_id="...")` invokes this route and ignores search criteria.
  Unknown/withdrawn IDs return 404. The documented response is
  `{schema, request_id, track}`; the client also accepts the versioned flat form
  `{schema, request_id, id, title, ...}`. An unversioned bare track is rejected.
  Confirm the detail envelope with each provider before production integration.

Search filters are `genre: string[]`, `language: string[]`, `vocal: string`,
`tags: string[]`, `artist: string` and `duration_s: {min?, max?}`. The built-in
tool accepts comma-separated lists and converts them into arrays; its
`duration_min_s`/`duration_max_s` use 0 for an omitted bound. Genre, language,
vocal, artist and duration are constraints unless explicitly relaxed. Tags are
ranking hints; unknown tags are left to the server's query handling.

Search responses contain `{schema, request_id, relaxed, tracks}`. Each track
contains ID, title, artist `{id, name, voice?}`, genre, language, vocal, tags,
duration_s, cover_url, audio `{url, format, expires_at}`, lyrics_excerpt, score,
match_reason and usage `{scope, attribution}`. Audio is mp3 or wav over HTTPS.
`expires_at: null` is non-expiring; otherwise refresh by ID before playback when
expired. A refreshed URL is not a new song ID.

The server may relax constraints in order: **query → duration_s → vocal → genre
→ artist**. It lists those fields in `relaxed`. **Language is never relaxed**;
tags were never hard constraints. Explain relaxation to the user, for example
“没有找到该歌手的摇滚歌曲，找到的是她的其他曲风”。Do not promise the original
criteria still hold. The client rejects excluded/duplicate songs, wrong language
and violations of unrelaxed structured criteria. Empty results stay empty.

`track.vocal` may be `null` when voice metadata has not been labelled. The client
preserves it as **unknown**, never inferring a voice from the artist name or
treating it as instrumental. Response compatibility includes this real-service
case; request `filters.vocal` still accepts only the four declared voice codes.
A null voice cannot satisfy an explicit vocal constraint unless the server lists
`vocal` in `relaxed`, which must be explained to the user. A missing vocal field
or an unrecognised non-null code is an invalid response.

## Usage boundary

`usage.scope: personal_playback` permits the current playback only. It does not
permit a persistent audio cache, offline library, file attachment, re-upload or
redistribution. Playback may use bounded transient buffers; do not save the
song for another session. Return attribution when presenting the track.

The integration brief also proposed downloading songs and uploading them as chat
attachments. That contradicts `personal_playback`: deleting the local temporary
file does not remove the chat platform's copy. **This integration does not enable
that flow.** Chat can present song information. A provider must explicitly settle
permitted chat delivery before audio attachment delivery can be an acceptance
criterion; do not invent a permissive scope. No general file tool is newly
granted to viewer users by this feature. This is a music integration boundary,
not a sandbox for separately authorised general-purpose desktop tools.

## Failure handling

Errors are `{schema, request_id, error: {code, message, allowed?}}`:

| HTTP | Code | Client behaviour |
|---|---|---|
| 400 | schema_mismatch | Return error; do not parse as tracks |
| 401 / 403 | unauthorized / forbidden | Return error; no retry |
| 404 | not_found | Return error; no retry |
| 422 | invalid_filter | Return bounded allowed values; no retry |
| 429 | rate_limited | Respect Retry-After; do not retry earlier |
| 5xx | internal | At most one retry within the request budget |

Requests have one total time budget including retries. A Retry-After longer than
that budget is returned to the caller, and the provider refuses further calls
until its cooldown expires; it never stalls an agent turn for a daily quota.
Timeouts, TLS and connection failures return an unavailable error, never songs.
Responses are bounded at 1 MiB. Redirects and malformed/versionless JSON fail.

Capabilities are warmed when the agent starts and cached for 24 hours. A transient
failure uses the documented core filter values and an empty tag vocabulary,
retrying capabilities after 60 seconds. Authentication/protocol failures remain
errors. Startup tags are appended only to the tool description; later refreshed
tags appear in tool results, not changing system-prompt content each turn.
Changing provider configuration invalidates the cache.

## Validation and five-minute demo

```bash
cd agent-core
DB_PATH=$(mktemp -u /tmp/motus-music-test.XXXXXX.db) python3 -m unittest discover -s tests -p test_music_search.py -v
```

Tests use stubbed HTTP and fictional metadata. They exercise the real desktop
tool entry, provider authentication/paths, budgeted retries, cooldown, schema
validation, exclusion, relaxation, capability caching, configuration changes and
mock detail lookup. No network, hardware, GPU, catalogue data or audio is needed.

In Dashboard, select mock, search, change songs and query one returned ID. Select
none and confirm `not_configured` is visible rather than fabricated success.
Export a Solution and confirm the music endpoint/key are blank. For a configured
real provider, repeat the search with valid deployment credentials and verify the
request ID, attribution, constraints and URL refresh against that provider.
Mock results do not establish production service compatibility or physical audio
playback. This PR is independently usable without the music playback plugin.
