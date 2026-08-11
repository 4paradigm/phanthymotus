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

## Semantic Navigation 占位接入

`semantic_navigation` 已显式接入 `PerceptionBundle`，默认使用
`namespace=ubuntu`。当前是与语义导航联调用的最小占位插件：不暴露
MCP tools，不订阅 ROS topic，也不持有运行时资源。后续能力实现应在
`plugins/semantic_navigation/` 内扩展，无需再手工修改容器内的
`main.py` 或 `config.yaml`。

张逸群提供的 `vln` 最小插件也已显式接入 `PerceptionBundle`。默认启用并在
`namespace` 为空时按 hostname 自动生成命名空间；当前同样不暴露 MCP tools、
不订阅 ROS topic，也不持有运行时资源。后续视觉语言导航能力直接在
`plugins/vln/` 内扩展。

---

## Nav2 卡片

`nav2` 是单实例 `processor` 卡片，只提供点到点导航、路径规划、
滚动 costmap 局部避障和速度提案。建图、定位、去畸变、地图存储和可视化
由 FAST-LIVO2 卡片负责。Nav2 不再读取 Driver 原始 state/cloud，不启动
SLAM Toolbox、AMCL 或 Map Server。

用户可见约束：

- 必需输入为 FAST-LIVO2 归一化后的 `/ubuntu/navigation/odom` 和
  `/ubuntu/navigation/cloud_registered`；
- odom 必须是 `map -> base_link`，registered cloud 必须是 `map` frame，
  两者使用 ROS system time；
- 输入超过 `500 ms`、frame 错误、`map -> base_link` TF 或 Nav2 lifecycle
  不 ready 时 fail closed；
- `speed` 范围 `0.30–1.00 m/s`，Rotation Shim 先对齐航向；
- 卡片可配置 X/Y/yaw 三轴非零速度的最小/最大绝对值，默认
  `X=0.30–1.00 m/s`、`Y=0–0 m/s`、`yaw=1.00–2.00 rad/s`；
- `namespace=ubuntu` 和 proposal TTL `250 ms` 仍是首版冻结合同；请求速度
  在每条 `velocity_proposal` 上强制限制正向速度，Nav2 `SpeedLimit` 只作为控制器
  advisory，不再把 DDS transport ack 误判为控制器已应用速度；
- FAST-LIVO2 卡片的 odom 以 `sensor/odometry` 显示实时位姿与速度；
  地图仍以 FAST-LIVO2 `map_view` 为唯一权威源，Agent Core 在显示层把原生
  `/plan` 叠加到该地图，并提供同一三维点云的 `2D/3D` 视图切换，同时保留
  Nav2 卡片的独立路径数据流；Nav2 默认预览为实时二维
  `/global_costmap/costmap`，同屏叠加规划路径、机器人位姿、终点和膨胀障碍；
- Agent Core 会在 Canvas 项目启动后消费工具声明的
  `x-topic-actions`；只有实际连到 `goal_pose` 端口的 JSON topic
  才能调用 `navigate_to_pose`，并严格校验 schema、字段白名单、
  必填坐标和唯一 `goal_id`。`x-execution-control` 的 Driver lease
  编排仍是独立能力，不由 topic action 绕过。

FAST-LIVO2 与 Nav2 都使用独立 ROS 2 companion 镜像，并已作为正式 service
接入 `perception/deploy/service.yml`。整体执行 `docker compose up -d` 时三者一起
启动；Canvas 的 `start/stop` 不创建或删除容器。当前 Agent Core Dashboard
部署器仍只启动 fragment 中的第一个 service（`--no-deps`），所以首次自动部署
sidecar 仍需 Core 的 multi-service 部署能力，这是本 Perception PR 的范围外依赖。
Nav2 companion 开发构建入口：

```bash
cd perception/plugins/nav2/companion
docker compose --env-file source-lock.env build nav2
```

完整 action、topic/schema、配置、部署限制和验收状态见
[`plugins/nav2/README.md`](plugins/nav2/README.md)。

## FAST-LIVO2 卡片

`fast_livo2` 是 Perception Bundle 内的会话级建图/定位 processor。它通过
独立 companion 运行锁定的 FAST-LIVO2 算法，消费既有 Driver
`navigation_sensors` 输出，并向 Nav2 提供权威
`map -> base_link` odom、`map` registered cloud 和 Canvas `map_view`。

当前只支持同一算法进程内边建图边导航。PCD 会持久化，但锁定算法不支持
加载旧图或全局重定位，不能把重启后的新会话称作继续定位。输入输出、坐标
换算、Canvas 连线、构建和部署步骤见
[`plugins/fast_livo2/README.md`](plugins/fast_livo2/README.md)。

正式 compose 现在包含三个服务：主 Perception、FAST-LIVO2 companion 和
planner/controller-only Nav2 companion。开发测试脚本同样一次管理三个带
`com.phanthymotus.test-owner=nav2-card` label 的容器。
