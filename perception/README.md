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
| `vad_pre_roll_ms` | `500` | Audio retained from *before* the VAD tripped. Recovers clipped word onsets — without it the first syllable is often missing, which costs wake-word recall. |

---

## Plugin Concurrency

**`dispatch()` is not single-threaded.** `main.py` serves MCP over
`ThreadingHTTPServer`, so every `tools/call` runs on its own thread. `start`,
`stop`, `config`, and `speak` on the *same* plugin can genuinely run at once —
the canvas does exactly this (config → start, then stop, then config → start).

This has already caused a production incident, so the rules below are not
theoretical.

### The failure mode

Any plugin that keeps per-instance state in a dict is exposed to this shape:

```python
# ❌ WRONG — check-then-act with no lock
node_key = instance_id or input_topic
if node_key not in self._nodes:          # ← two threads both pass here
    node = _ASRNode(...)
    self._executor.add_node(node)
    self._nodes[node_key] = node         # ← only the last one survives
return self._nodes[node_key].start()
```

Both threads build a node with the *same* ROS node name, both add it to the
executor, and the dict keeps only the second. The first is now an **orphan**: its
subscription, its VAD subprocess, and its transcription thread are all still
running and still publishing to the same output topic, but it is not in
`self._nodes`, so `stop` can never reach it. It survives until the process exits.

Observable symptoms: every utterance recognised and published twice, duplicate
files in `/models/vad_segments` with byte-identical content, an extra
`vad_worker` child process that `stop` does not reap, and this from rclpy:

```
Publisher already registered for provided node name. If this is due to multiple
nodes with the same name then all logs for that logger name will go out over the
existing publisher.
```

### The rules

**1. Make the dict access atomic.** One `threading.RLock` per plugin, guarding
every read-modify-write of the state dict:

```python
# ✅ CORRECT — atomic get-or-create
with self._nodes_lock:
    node = self._nodes.get(node_key)
    if node is None:
        node = _ASRNode(...)
        try:
            self._executor.add_node(node)
        except Exception:
            node.destroy_node()          # don't leak a half-registered node
            raise
        self._nodes[node_key] = node
    else:
        self._sync_cfg(node)
```

**2. Never hold that lock across `node.start()`, `node.stop()`, or a model
load.** `_ASRNode.start()` blocks for up to 15 s waiting for the first audio
chunk. If `stop` is queued behind the lock for those 15 s, it cannot set the
cancellation flag in time, `start` sails through to `running`, and you are left
with a pipeline nobody asked for. Register the node inside the lock, then release
it and call `start()` outside.

**3. Register the node *before* starting it.** That is what lets a concurrent
`stop` find it and cancel the in-flight start. Loading a model or otherwise
blocking *before* the node is in the dict means `stop` finds nothing, returns
`{"state": "idle"}`, and silently no-ops — while the start it was meant to cancel
completes anyway.

**4. `stop` signals first, locks second.** Give the node a non-blocking
`request_stop()` that sets its cancellation events, call that before taking any
lock, and only then tear down:

```python
def stop(self) -> dict:
    self.request_stop()                  # non-blocking; unblocks an in-flight start
    with self._lifecycle_lock:
        self._teardown()
        self.state = "idle"
        return {"state": "idle"}
```

**5. Guard the node object too, and treat "starting" as taken.** A per-node
`RLock` plus `if self.state in ("running", "starting")` — otherwise two threads
that resolve to the *same* node object can both enter `_start_inner()` and build
two subscriptions and two subprocesses on one node.

**6. `destroy_node()`, not just `remove_node()`.** `remove_node` detaches the
node from the executor; it does not release the rclpy node, its publishers, or
its node name. Skip it and every start/stop cycle leaks a topic endpoint, and a
later start on the same key collides with the still-registered ghost:

```python
def _dispose_node(self, node, key=""):
    node.stop()
    self._executor.remove_node(node)
    node.destroy_node()                  # ← required
```

**7. Snapshot before iterating.** `info` is a heartbeat probe called constantly.
Iterating the live dict can raise `RuntimeError: dictionary changed size during
iteration` in the middle of a start. Copy under the lock, then iterate the copy.

### Where this applies

Every MCP server in the project uses `ThreadingHTTPServer` — `perception/main.py`
and each robot driver's `main.py`. Any plugin holding a `self._nodes` /
`self._instances` / `self._streams` dict needs the treatment above.

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

## controlled_semantic_spatial 卡片

PR #99 的 FAST-LIVO2、Nav2 和 VLN 已合并为一张公开
`controlled_semantic_spatial` 卡片：

- Canvas 只注册一个 `controlled_semantic_spatial` 工具；
- 正式 Compose 只运行一个 `embodied-perception` 容器；
- FAST-LIVO2 和 Nav2 由卡片在同容器内作为 ROS 子进程组托管，不调用 Docker；
- 统一镜像从锁定 ROS Humble 基础镜像直接编译 FAST-LIVO2 及其全部锁定依赖，
  不需要预构建的 `phanthy-fast-livo2` 镜像；
- lidar、IMU、RGB 是外部必需输入，`goal_pose` 是可选外部入口；
- odom、registered cloud、obstacle map、status 以及 VLN 到 planner 的目标传递
  都是卡片内部边，不需要 Canvas 连接；
- 物理运动仍严格走 `velocity_proposal -> Driver loco`，Perception 不直接控制机器人。

完整 action、topic、配置、构建、许可证和验收边界见
[controlled_semantic_spatial 卡片文档](plugins/navigation/README.md)。

统一镜像构建：

```bash
./deploy/build_perception.sh --mirror tuna
```

`navigation` 仍是仓库内部的默认镜像构建 variant；公开 MCP 工具名为
`controlled_semantic_spatial`。无需专用构建脚本或预构建 companion 镜像。
需要旧 CPU 或 Jetson 镜像时再显式传入 `--variant cpu` 或
`--variant jetson`。

G1 临时测试只创建一个 Perception 容器；将上一步输出的精确镜像名传入：

```bash
export PERCEPTION_IMAGE=local/phanthy-motus/perception:<exact-navigation-tag>
STAGE=preflight bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
STAGE=start bash perception/plugins/navigation/deploy/scripts/owner-start-g1-test-containers.sh
```
