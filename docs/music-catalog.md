# Music catalogue: motus.music/1

PhanthyMotus calls a deterministic music catalogue over HTTPS. The public
repository contains this contract, [OpenAPI](openapi/motus-music-v1.yaml), a generic
client and fictional mock metadata. Deployment endpoints, keys, real catalogue
records and the remote search implementation are not published.

The client belongs to the [phanthy_music card](phanthy_music.md). A model extracts
intent into `search` arguments, then calls `play_by_id(track_id)` on that same card.
`play_by_genre(genre)` combines search and playback within the card.
There is no server-side LLM and no independent MusicSearch built-in. Local MCP
connects Agent Core to the Perception card; the card uses authenticated REST to
the catalogue. No change to MCP authentication or external MCP service is needed.

## HTTP contract

- HTTPS only; JSON UTF-8. Credentials use `Authorization: Bearer <api_key>`, never
  URL parameters. Redirects are refused, including redirects to HTTPS.
- Every envelope has `schema: motus.music/1`. A different version is an error.
- Responses carry `request_id`; clients preserve it in bounded errors.
- `GET /v1/capabilities` returns filters, tag vocabulary, maximum `top_k` and
  audio URL lifetime. Capabilities are cached in memory for 24 hours. Temporary
  lookup failures fall back to fixed genre/language/vocal codes and no tags for
  60 seconds; authentication and incompatible schemas are never hidden.
- `POST /v1/music/search` accepts `schema`, optional `query`, `filters`, `top_k`
  (default 3, up to 10/capability maximum), and `exclude_ids` (up to 50).
- `GET /v1/music/tracks/{id}` returns refreshed details. The client accepts the
  versioned `{schema, request_id, track}` envelope and a versioned flat track;
  it validates that the returned ID matches the requested ID.

Search filters are `genre` and `language` arrays, `vocal`, `artist`, `tags`, and
`duration_s: {min?, max?}`. Query is at most 200 characters. The card accepts
comma-separated strings for array fields and converts them into REST arrays.
Unknown tags remain free-text hints, not new hard filters. An empty search is
valid. There is no pagination or bulk catalogue download.

Genre/language/vocal/artist/duration are hard constraints unless expressly
relaxed. Tags are scoring hints. The server may relax query, duration, vocal,
genre and artist, in that order; language is never relaxed. The response's
`relaxed` array and `match_reason` must be communicated honestly to the user.
An empty result stays empty. Duplicate/excluded IDs and violations of unrelaxed
filters are rejected. “Another song” resubmits the previous explicit criteria
with `exclude_ids`; a process-global last-user query is not inferred.

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
