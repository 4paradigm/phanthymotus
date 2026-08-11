# Perception Stack

Modular ASR/TTS perception plugins running as an MCP HTTP server. Connects to Agent Core via MCP tool calls and exchanges audio/text over ROS2 DDS topics.

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

## Nav2 卡片

`nav2` 是单实例 `processor` 卡片，提供建图、地图保存/加载、位置标签和点到点导航。
首版接收 G1 Driver 的 `/ubuntu/loco/state` 与专用 PCV2 流
`/ubuntu/navigation/lidar`，输出
`/ubuntu/navigation/nav2/velocity_proposal` 以及仅用于 Canvas 监控的
`/ubuntu/navigation/nav2/map_view`。卡片不会直接调用机器人 SDK；任何物理
执行都必须由 Agent Core 建立受信任务 lease，再由 Driver 独立完成限幅、急停和停车确认。

用户可见约束：

- `speed` 范围 `0.05–0.15 m/s`，首版导航固定允许绕障，不显示无可选值的 `mode`；
- 输入超过 `500 ms`、TF/地图/Nav2 lifecycle 不 ready 时 fail closed；
- Driver v2 输入必须使用 ROS system/Unix 同一时钟域；Perception 会在发布
  odom/TF/cloud 前拒绝陈旧、超前、倒退或 odom/scan 错位的时间戳；
- `namespace=ubuntu`、proposal TTL `250 ms` 和速度上限是首版冻结合同；
- 地图宿主机目录为 `/opt/phanthy-motus/data/nav2/maps`，companion 容器内为 `/maps`；
- `start_mapping` 自动切换 mapping，`stop_mapping` 原子保存后自动切回 localization；
- Canvas 「查看数据流」默认打开 `map_view`，以 1 Hz 显示占用栅格和机器人位姿，
  该数据流不能发起导航或改变机器人状态；
- 当前官方 Agent Core 尚缺 `x-execution-control` / `x-topic-actions` 消费能力，
  因而 `goal_pose` topic 和 Driver 物理 lease 仍是明确的外部依赖。

Nav2 依赖独立的 ROS 2 Humble companion 镜像。它已经作为第二个 service 接入
`perception/deploy/service.yml`，整体执行 `docker compose up -d` 时会一起启动；Canvas
的 `start/stop` 不创建或删除容器。当前 Agent Core Dashboard 部署器仍只启动 fragment
中的第一个 service（`--no-deps`），所以首次自动部署 sidecar 仍需 Core 的 multi-service
部署能力，这是本 Perception PR 的范围外依赖。开发构建入口：

Nav2-only 配置会同时禁用 ASR WebSocket 服务；未启用的 ASR 不会占用 `ws_port`、启动
后台线程或引入 `websockets` 运行时依赖。

```bash
cd perception/plugins/nav2/companion
docker compose --env-file source-lock.env build nav2
```

完整 action、topic/schema、配置、部署限制和验收状态见
[`plugins/nav2/README.md`](plugins/nav2/README.md)。
