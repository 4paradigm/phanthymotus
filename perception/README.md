# Perception Stack

Modular ASR/TTS/duplex-audio perception plugins running as an MCP HTTP server. Connects to Agent Core via MCP tool calls and exchanges audio/text over ROS2 DDS topics.

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

---

## Duplex Audio (external microphone + TTS reference + AEC)

`duplexaudio` owns the timing-sensitive TTS reference and AEC path while leaving
ASR as an independent downstream card. It has one external-microphone input and
two PCM outputs:

| Direction | Topic pattern | Format |
|-----------|---------------|--------|
| Input | `start.input_topic` | `audio/pcm-16k` |
| Output port 0 | `{input_topic}/duplexaudio/clean` | `audio/pcm-16k` |
| Output port 1 | `{input_topic}/duplexaudio/tts` | `audio/pcm-16k` |

Connect output port 0 to the existing ASR card and output port 1 to Speaker. Use
MCP `duplexaudio.speak(text)` to synthesize speech. Every paced TTS frame is
also stored as a timestamped AEC reference before it is published to the
downstream speaker actuator. The mic path aligns that reference using
`aec_delay_ms`, performs AEC, and re-chunks clean audio to the ASR minimum.

The plugin is disabled by default. When enabling it, keep standalone ASR enabled
but disable the standalone TTS plugin to avoid loading and playing TTS twice.
Publishing PCM alone does not prove physical speaker playback.

AEC backends:

- `auto`: try LiveKit WebRTC APM, then SpeexDSP.
- `livekit`: parity with the `voice-service` AEC3 backend; requires a compatible
  Python runtime and wheel.
- `speexdsp`: uses the system `libspeexdsp1` library and supports the current
  Python 3.8 Jetson image.

`aec_failure_policy=fail_closed` is the default: a missing backend or processing
failure is reported as an error instead of silently claiming echo cancellation.
`passthrough` exists only for explicit A/B and recovery tests. After several
seconds of robot speech, call `aec_calibrate`; apply its `d_real_ms` suggestion
only when `peak_corr` is credible, then verify `erle_db` and real ASR echo errors.

The full contract, failure semantics, and test gates are in
`plugins/duplexaudio/README.md`.
