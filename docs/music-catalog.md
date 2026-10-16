# Music catalogue: motus.music/1

PhanthyMotus calls a deterministic music catalogue over HTTPS. The public
repository contains this contract, [OpenAPI](openapi/motus-music-v1.yaml), a generic
client and fictional mock metadata. The Phanthy Music card commits a fixed
endpoint and API key in its source for its built-in integration; these are not
editable UI settings. Real catalogue records, audio assets and the remote search
implementation are not included. The REST contract remains provider-independent.

The client belongs to the [phanthy_music card](phanthy_music.md). A model extracts
intent into `search` arguments, then calls `play_by_id(track_id)` on that same card.
`play_by_genre(genre)` combines search and playback within the card.
There is no server-side LLM and no independent MusicSearch built-in. Local MCP
connects Agent Core to the Perception card; the card uses authenticated REST to
the catalogue. No change to MCP authentication or external MCP service is needed.

## Two kinds of content, one endpoint

The catalogue holds **songs** and **ASMR** behind the same address and the same
key; `filters.kind` picks which one a request is about. ASMR is trigger sounds
(chewing, friction, tearing — no voice at all) and softly-read voice
companionship (bedtime stories, meditation guidance, companionship, chat,
classical poetry).

| | Song | ASMR |
|---|---|---|
| How to ask | default; send no `kind` | `filters.kind = "asmr"`, or just `filters.category` |
| Own filters | `genre`, `vocal`, `artist` | `category` |
| Shared filters | `query`, `tags`, `language`, `duration_s`, `top_k`, `exclude_ids` | same |
| Audio | mostly mp3, ~4–5 MB | wav; triggers 20–30 s, voice 1.5–4 min |
| `genre` on the result | a real code | always `other` — **do not show it** |
| `artist` on the result | the singer | none: `name` empty, `id` null |
| `lyrics_excerpt` | lyrics | the script's opening; empty for triggers |

One request returns one kind; they are never mixed. **A caller that only wants
songs changes nothing** — no `kind` means songs, which is why this is still
`motus.music/1` rather than a version bump: every ASMR field is an optional
addition.

`kind` is **never relaxed.** An ASMR request that matches nothing comes back
empty; it does not quietly become a song. `category` *is* relaxable, and the
client accepts it in `relaxed` alongside query/duration_s/vocal/genre/artist.
The categories are `trigger`, `bedtime_story`, `meditation`, `companion`,
`chat`, `poetry`, `other`; each entry has exactly one, and several may be passed
to mean "any of these".

Crossing the fields is a 422 at the server — `genre`/`vocal`/`artist` on an ASMR
request, `category` on a song request. The client raises the same
`invalid_filter` locally, before any network call: the answer is static, so
spending 3 s of a user's patience to be told it is waste.

## HTTP contract

- HTTPS only; JSON UTF-8. Credentials use `Authorization: Bearer <api_key>`, never
  URL parameters. Redirects are refused, including redirects to HTTPS.
- Every envelope has `schema: motus.music/1`. A different version is an error.
- Responses carry `request_id`; clients preserve it in bounded errors.
- `GET /v1/capabilities` returns filters, tag vocabulary, maximum `top_k` and
  audio URL lifetime. `filters.kind` and `filters.category` are `{value,label}`
  lists like `filters.genre`, and `features` contains `asmr` when the provider
  serves it. A provider without ASMR omits all three and keeps the built-in
  fallback vocabulary — the same tolerance `genre` already had. Capabilities are cached in memory for 24 hours. Temporary
  lookup failures fall back to fixed genre/language/vocal codes and no tags for
  60 seconds; authentication and incompatible schemas are never hidden.
- `POST /v1/music/search` accepts `schema`, optional `query`, `filters`, `top_k`
  (default 3, up to 10/capability maximum), and `exclude_ids` (up to 50).
- `GET /v1/music/tracks/{id}` returns refreshed details. The client accepts the
  versioned `{schema, request_id, track}` envelope and a versioned flat track;
  it validates that the returned ID matches the requested ID.

Search filters are `kind`, `category`, `genre` and `language` arrays, `vocal`,
`artist`, `tags`, and `duration_s: {min?, max?}`. Query is at most 200 characters. The card accepts
comma-separated strings for array fields and converts them into REST arrays.
Unknown tags remain free-text hints, not new hard filters. An empty search is
valid. There is no pagination or bulk catalogue download.

Kind/category/genre/language/vocal/artist/duration are hard constraints unless
expressly relaxed. Tags are scoring hints. The server may relax query, duration,
vocal, genre, category and artist, in that order; language and kind are never
relaxed. The response's
`relaxed` array and `match_reason` must be communicated honestly to the user.
An empty result stays empty. Duplicate/excluded IDs and violations of unrelaxed
filters are rejected. “Another song” resubmits the previous explicit criteria
with `exclude_ids`; a process-global last-user query is not inferred.

Every ASMR entry has `vocal: null` and no artist, and that is correct data
rather than missing data. A missing `kind` on a track reads as `song`, which is
what a provider predating the split returns.

`track.vocal` can be null for unlabelled/unknown voice metadata. It does not mean
instrumental and cannot satisfy an unrelaxed vocal filter. Missing voice fields
or unknown non-null codes are invalid. Request vocal values remain
`male`, `female`, `mixed`, `instrumental`.

## Playback and usage

Tracks contain an ID, title, artist, metadata, HTTPS audio URL/format/expiry,
lyrics excerpt, matching information and usage. An `expires_at` or URL TTL of
null means no stated expiry. `play_by_id(track_id)` looks up fresh details and forwards
the returned format, expiry, attribution and usage into the player. A removed
track returns `not_found`; a mock track returns `not_playable`.

`usage.scope: personal_playback` permits current playback, not audio-file caching,
rehosting or redistribution. Audio is streamed through memory to the decoder.
The integration intentionally does not download and upload music as chat
attachments: that conflicts with this usage scope. Chat can return song metadata;
robot/local playback uses the card. No voice cloning or new-song generation is
provided by this catalogue.

## Errors and limits

| HTTP | Client error |
|---|---|
| 400 | schema_mismatch |
| 401 / 403 | unauthorized / forbidden |
| 404 | not_found |
| 422 | invalid_filter, with bounded allowed values |
| 429 | rate_limited, respect Retry-After |
| 5xx | internal |

Only 429 and 5xx receive one retry inside the configured budget. 429 establishes
a provider cooldown, including when Retry-After exceeds the current request's
budget. Other 4xx, redirects and network failures are not retried automatically.
Responses are capped at 1 MiB; errors never expose request URLs, credentials or
arbitrary server error messages. Calls have cancellation checks between network
reads and during retry waits. Playback controls do not wait for catalogue I/O.

## Verification

Measured against the live service on 2026-10-17: capabilities carries both
vocabularies and `features: ['query','lyrics_excerpt','asmr']`; `kind=asmr`
returns wav entries across all categories; `category=trigger` with `query=薯片`
returns 咬/咀嚼·薯片 with an empty `lyrics_excerpt`; a song search is unchanged;
and both crossed-field cases return 422. The mock provider carries ASMR samples
shaped the same way (no artist, null vocal, `genre: other`, empty trigger
script), so the offline suite exercises the path — but **mock success is not
remote compatibility**, and neither is either of them robot acceptance.

Run catalogue contract/provider tests with:

```bash
cd perception
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_music_catalog.py tests/test_phanthy_music.py -q
```

Mocks and tests contain only fictional metadata and placeholder credentials.
For deployment acceptance, exercise real capabilities, filtered search, next
song, condition relaxation and `play_by_id(track_id)` through the registered card;
then check actual PCM output and interruption. Do not count mock success as
remote compatibility or local audio output as robot acceptance.

## Filter vocabularies reach the card

`GET /v1/capabilities` carries `filters.genre`, `filters.language` and
`filters.vocal` as `{value,label}` lists, cached for 24 h (60 s on a transient
failure, with the built-in fallback). `MusicService.vocabulary()` surfaces them
for the card's `list_genres`, and `execute()` adds `available_genres` beside the
`available_tags` it already returned.

This used to be fetched, validated, cached — and never surfaced. A model could
only learn a valid genre code by sending a wrong one and reading `allowed` off
the `invalid_filter` error. `vocabulary()` adds no network path: it reuses the
same `_prepare` and the same cache.

`source` is `catalogue` or `fallback`. It matters: after a transient failure the
values are the built-in list for 60 s, and a caller treating those as
authoritative would confidently offer codes this provider may not have.
