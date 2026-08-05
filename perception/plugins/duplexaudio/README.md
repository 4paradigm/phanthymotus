# Duplex Audio 卡片规格

| 项目 | 必填内容 |
| --- | --- |
| card id | `duplexaudio` |
| card type | `processor` |
| 感知目标 | 接收外接麦克风 16 kHz 单声道 PCM16，用本卡 TTS PCM 和预计播放时间执行 AEC，输出 clean mic 给独立 ASR 卡；同时通过 `speak` 输出 TTS PCM 给 Speaker |
| input topic | `start.input_topic`；`audio_msgs/AudioChunk`；`audio/pcm-16k`，PCM_S16_LE、16000 Hz、mono；`BEST_EFFORT + KEEP_LAST(50) + VOLATILE` |
| output topic | clean mic：`{input_topic}/duplexaudio/clean`；TTS：`{input_topic}/duplexaudio/tts`；两者均为 `audio_msgs/AudioChunk` / `audio/pcm-16k` |
| actions | `info/config/start/stop/speak/aec_stats/aec_calibrate` |
| 配置 | shared：TTS speaker/speed、AEC backend/filter length；instance：`aec_enabled`、`aec_delay_ms`、`aec_failure_policy`。运行中修改会停止受影响实例，下次 `start` 生效 |
| 模型 | 只复用 `plugins.tts` 的 sherpa-onnx Matcha + Vocos adapter；ASR/VAD/KWS 由下游独立 ASR 卡负责；AEC 优先 LiveKit WebRTC APM，不可用时回退 SpeexDSP |
| 部署 | CPU / x86 / Jetson；镜像需 `libspeexdsp1`。Python 3.8 Jetson 路线使用 SpeexDSP，不升级 ROS Python ABI |
| 测试 | 工具包契约校验；AEC 时间对齐、burst、失败语义、TTS observer 和多实例单测；aarch64 SpeexDSP 冒烟；最后用外接麦克风+真实扬声器做 A/B |

## 画布连接

```text
external mic -> duplexaudio input
duplexaudio clean output (port 0) -> standalone ASR input
duplexaudio TTS output   (port 1) -> Speaker input

MCP duplexaudio.speak(text)
  -> TTS adapter
  -> timestamped AEC reference (in-process, before publish)
  -> TTS PCM output
```

ASR 不参与 reference 时间对齐；clean mic 经 ROS2 到 ASR 的延迟只计入识别时延，不影响 AEC 对齐。本卡仍不直接驱动扬声器，所以 `aec_delay_ms` 仍需要真机标定；一体化不能消除 Speaker 队列、DAC、声学传播和麦克风缓冲延迟。

## 失败语义

- `fail_closed`（默认）：AEC backend 不可用时不启动；处理失败后停止 clean mic 输出。
- `passthrough`：只用于显式 A/B/恢复测试；AEC 失败时透传 raw mic，`info/aec_stats` 仍保留错误。
- 非 `audio/pcm-16k` 或奇数字节 PCM 会让 bridge 进入 `error`，不输出伪 clean mic。

## 已知边界

- TTS reference 基于预计发布节奏；Speaker 内部额外排队由 `aec_delay_ms` 和 `aec_calibrate` 吸收。如果多次 `d_real_ms` 明显多峰，需新增 playback receipt，而不是合并 ASR。
- AEC 不保证 double-talk 下人声完全无损；唤醒召回率与时延必须单独验收。
- 同一条语音链路不应同时启用独立 TTS 卡和 `duplexaudio` TTS，否则会重复合成和播放。
