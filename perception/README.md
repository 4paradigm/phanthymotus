# Perception Stack

Modular perception plugins running as an MCP HTTP server. They connect to Agent
Core through MCP tool calls and exchange data over ROS2 DDS topics.

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

## Navigation 卡片

PR #99 的 FAST-LIVO2、Nav2 和 VLN 已合并为一张公开 `navigation` 卡片：

- Canvas 只注册一个 `navigation` 工具；
- 正式 Compose 只运行一个 `embodied-perception` 容器；
- FAST-LIVO2 和 Nav2 由卡片在同容器内作为 ROS 子进程组托管，不调用 Docker；
- 统一镜像从锁定 ROS Humble 基础镜像直接编译 FAST-LIVO2 及其全部锁定依赖，
  不需要预构建的 `phanthy-fast-livo2` 镜像；
- lidar、IMU、RGB 是外部必需输入，`goal_pose` 是可选外部入口；
- odom、registered cloud、obstacle map、status 以及 VLN 到 planner 的目标传递
  都是卡片内部边，不需要 Canvas 连接；
- 物理运动仍严格走 `velocity_proposal -> Driver loco`，Perception 不直接控制机器人。

完整 action、topic、配置、构建、许可证和验收边界见
[Navigation 卡片文档](plugins/navigation/README.md)。

统一镜像构建：

```bash
./deploy/build_perception.sh --mirror tuna
```

`navigation` 是仓库默认构建目标，不需要专用构建脚本或预构建 companion
镜像。需要旧 CPU 或 Jetson 镜像时再显式传入 `--variant cpu` 或
`--variant jetson`。

G1 临时测试只创建一个 Perception 容器；将上一步输出的精确镜像名传入：

```bash
export PERCEPTION_IMAGE=local/phanthy-motus/perception:<exact-navigation-tag>
STAGE=preflight bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
STAGE=start bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
```
