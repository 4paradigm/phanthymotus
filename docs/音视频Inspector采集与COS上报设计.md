# 音视频 Inspector 采集与 COS 上报设计

> 状态：设计提案，尚未实现
> 日期：2026-07-17
> 分支：`feat/audio-video-inspectors`
> 目标设备：上海/北京 G1（Jetson），同时保留 x86 开发验证能力

## 1. 背景

当前 Agent Core 内置的 `api/inspection.py` 负责 ROS2 DDS topic 的实时监看和 WebSocket 转发：注册 topic、按需订阅、缓存最近一帧，并通过 `/ws/bus/{topic}` 推送给 Dashboard。它不负责长期录制、本地滚动留存或云端上报。

本需求希望在画布中增加两张可连线的 Inspector 卡片：

- Audio Inspector：连接麦克风 topic，连续采集音频；
- Video Inspector：连接摄像头 topic，连续采集视频；
- 两张卡片都能配置腾讯云 COS 上报位置；
- 数据先可靠落到设备本地，再异步上传；
- 本地保存时间可配置，并通过滚动清理限制空间；
- 云端保存所有已经成功采集并完成分片的数据。

这不是对现有实时预览 Inspector 的替换。实时预览强调低延迟、允许丢旧帧；录制上报强调持久化、重启恢复、上传确认和可审计，两者应保持独立。

## 2. 当前代码事实与缺口

### 2.1 已有能力

1. Agent Core 已支持 Resource Center 的 `inspection` category，并预留端口 `15671`。
2. 画布会把连线解析出的 `input_topic` 和卡片 `instance_id` 传给 `start` action。
3. MCP tool 已支持 `configSchema`、shared/instance 配置、启动前自动恢复配置和多实例。
4. embedded inspection 已能订阅 `audio_msgs/AudioChunk`、`sensor_msgs/CompressedImage` 等部分 ROS2 消息并做实时转发。
5. 团队现有 `cos-uploader` 已验证“本地落盘与网络上传解耦、断网重试、启动扫描补传”的方向。

### 2.2 必须补齐的能力

1. 当前没有 Inspection Stack 运行时，也没有可以放进画布的音频/视频 Inspector tool。
2. 左侧卡片栏只展示 controller、driver 和 perception MCP，`inspection` MCP 目前不会进入卡片栏。
3. 当前 tool type 只有 `sensor/processor/actuator`。只有输入 topic 的录制卡片会被误判为 actuator。
4. embedded inspection 使用内存队列和最近帧缓存，不具备文件分片、上传账本、TTL 或重启恢复能力。
5. 当前 `cos-uploader` 的 manifest 只记录 `relpath → size`，不能直接证明 COS 对象已经按内容校验，也没有“只有上传确认后才允许本地删除”的硬门禁。
6. 当前通用 tool config 会把配置保存在 Agent Core SQLite，并通过查询接口返回；COS SecretId/SecretKey 不应直接放进普通 tool config。
7. 当前画布只按 format 完全相等连线，不支持 `video/*` 之类的 wildcard。第一版必须明确支持的具体输入 format。

## 3. 关键约束

### 3.1 有限磁盘与“云端全部保存”的边界

有限磁盘、无限时长断网、持续采集和零丢失不能同时保证。

本设计对“云端全部保存”的定义是：

> 每个已经完成本地原子分片的 segment，必须经过 COS 上传与校验，才能进入本地删除流程；未确认上传的数据不得因普通 TTL 被删除。

默认磁盘压力策略为：

1. 删除已经上传确认且超过配置留存时间的 segment；
2. 若仍达到 critical watermark，则暂停对应 Inspector 采集并报警；
3. 不静默删除未上传 segment；
4. 不把“上传请求返回”当成上传确认。

这样可以保证不会为了继续采集而悄悄破坏云端完整性，但极端长时间断网时会出现采集暂停。若未来业务选择“持续采集优先”，必须显式启用 `drop_oldest_unuploaded`，并接受云端不完整；第一版不提供该选项。

### 3.2 不阻塞机器人主链路

- ROS2 callback 只负责复制必要数据并写入有界内存队列；
- 编码、文件写入、COS 上传和清理均在独立 worker 中执行；
- 网络失败不得阻塞 DDS callback、Agent Core 或上游 Driver；
- 队列溢出必须计数和报警，不得只写一条难以发现的日志。

### 3.3 凭证不能进入普通卡片配置

卡片配置可以保存 COS region、bucket、prefix、device ID 和 credential profile 名称，但不能保存明文 SecretId/SecretKey。

第一版凭证由部署层提供：

- Docker secret 或只读文件：`/run/secrets/phanthymotus/cos/<profile>.json`；或
- 容器环境变量：`COS_SECRET_ID/COS_SECRET_KEY/COS_SESSION_TOKEN`，仅用于本地开发和兼容部署。

卡片只保存 `credential_profile=default`。`info`、日志、MCP 返回和 Agent Core SQLite 都不得回显凭证。

## 4. 总体设计

新增一个独立的 Inspection Stack 服务，在同一个 MCP bundle 中暴露两张卡片：

```text
Mic Driver ── audio/pcm-16k ──▶ Audio Inspector ──┐
                                                  │
Camera Driver ── image/jpeg ──▶ Video Inspector ──┤
                                                  ▼
                                      Durable Segment Store
                                      - atomic segment writer
                                      - upload ledger (SQLite)
                                      - retention sweeper
                                                  │
                                                  ▼
                                        COS Upload Workers
                                      - retry / resume / verify
                                                  │
                                                  ▼
                                            Tencent COS

Agent Core Canvas ── config/start/stop/info ──▶ Inspection Stack MCP :15671

Agent Core embedded inspection ──▶ Dashboard live preview
                         与录制上报链路并行，互不依赖
```

### 4.1 为什么是一个 Bundle、两张卡片

- 两张卡片可以独立添加、连线、配置、启动和停止；
- 共用 COS client、上传账本、磁盘统计、secret loader 和清理器，避免重复实现；
- 一个容器只需一套 ROS2、COS SDK 和编解码运行时；
- 音频或视频某一路失败时，另一张卡片仍可工作；
- 后续可继续增加 LiDAR、状态和事件 Inspector，而不修改 Agent Core 的基本协议。

### 4.2 新增 card type：`inspector`

`inspector` 表示消费数据并产生外部持久化副作用的数据终点：

| type | 数据方向 | 主要副作用 |
|---|---|---|
| `sensor` | 设备 → topic | 产生实时数据 |
| `processor` | topic → topic | 转换或推理 |
| `actuator` | command/topic → 机器人 | 改变物理世界 |
| `inspector` | topic → storage | 记录、审计和上报 |

Inspector 不能被 LLM 当成物理执行器。Agent Core 前端需要增加独立的类型样式、分组和画布位置；Flow View 应显示为 topic 的旁路 sink。

## 5. 卡片规格

### 5.1 Audio Inspector

| 项目 | 规格 |
|---|---|
| card id | `audioinspector` |
| class | `AudioInspectorPlugin` |
| type | `inspector` |
| multiInstance | `true` |
| 输入 | 一条由画布连入的麦克风 topic |
| 第一版 ROS type | `audio_msgs/AudioChunk` |
| 第一版 format | `audio/pcm-16k` |
| 音频语义 | PCM_S16_LE、16000 Hz、mono |
| QoS | `BEST_EFFORT + KEEP_LAST(depth=50) + VOLATILE` |
| 本地输出 | 固定时长 WAV segment + JSON metadata |
| 云端输出 | segment 和 metadata 两个不可变 COS object |
| output topic | 无 |

第一版不在卡片内部做 VAD，只保存连续原始音频。后续可以增加 `record_mode=continuous|vad`，但不能让 VAD 成为默认，否则会改变“完整采集”的语义。

### 5.2 Video Inspector

| 项目 | 规格 |
|---|---|
| card id | `videoinspector` |
| class | `VideoInspectorPlugin` |
| type | `inspector` |
| multiInstance | `true` |
| 输入 | 一条由画布连入的摄像头 topic |
| 第一版 ROS type | `sensor_msgs/CompressedImage` |
| 第一版 format | `image/jpeg` |
| QoS | `BEST_EFFORT + KEEP_LAST(depth=2) + VOLATILE` |
| 本地输出 | 固定时长 MP4/H.264 segment + JSON metadata |
| 云端输出 | segment 和 metadata 两个不可变 COS object |
| output topic | 无 |

第一版只承诺 `image/jpeg`。实现前 Gate 0 必须在目标 G1 上核对实际摄像头 topic、ROS type、format、分辨率、帧率和 QoS；若实际 Driver 输出 raw Image 或 H.264，应增加明确的 input port/adapter，不能做隐式格式猜测。

视频编码路线必须显式配置：

- G1 Jetson：优先验证 GStreamer `nvv4l2h264enc`；
- x86 开发机：可显式选择 `libx264`；
- 指定 encoder 不可用时启动失败，不做隐式 CPU 降级；
- 如果输入已经是 H.264，后续版本允许 remux，不重复编码。

## 6. Actions

两张卡片使用同一生命周期协议：

| action | 请求 | 行为 | 关键返回 |
|---|---|---|---|
| `info` | 可选 `instance_id` | 只读查询，不创建订阅或 worker | state、topic、segment、队列、磁盘、backlog、上传统计和最后错误 |
| `config` | shared/instance 配置 | 校验并保存配置；编码和路径类配置仅 idle 可改 | applied、restart_required、storage_estimate |
| `start` | `instance_id`、`input_topic` | 创建 ROS2 订阅、writer 和实例状态 | state、topic_in、session_id |
| `stop` | `instance_id` | 停止订阅，完成当前 segment 并排队上传；不等待全部上传 | state、finalized_segments、upload_backlog |
| `flush` | `instance_id` | 不停止订阅，立即结束当前 segment 并开启下一段 | finalized_segment |
| `testupload` | 无或 `instance_id` | 向配置 prefix 的 `_health/` 写入小对象并 HEAD 校验 | object_key、verified、latency_ms |

重复 `start` 和 `stop` 必须幂等。同一个 `instance_id` 重复 start 且 input topic 不一致时返回冲突，不能悄悄换订阅。

`stop` 只停止继续采集，不取消已经完成 segment 的上传；上传 worker 保持运行直到服务退出。服务重启后由 ledger 恢复补传。

## 7. 配置

### 7.1 Shared 配置

Shared 配置按 tool 保存，因此 Audio Inspector 和 Video Inspector 可以使用不同 bucket/prefix，也可以配置成相同目标。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `storage_backend` | enum | `cos` | 第一版只实现 `cos`；测试可注入 fake backend |
| `credential_profile` | string | `default` | 引用部署层 secret，不是凭证本身 |
| `cos_region` | string | `ap-beijing` | COS region |
| `cos_bucket` | string | 无 | 完整 bucket 名，例如 `embodied-ai-1252788780` |
| `cos_prefix` | string | `inspection-data` | 对象键前缀，不以 `/` 开头或结尾 |
| `device_id` | string | hostname | 例如 `sh-g1`、`bj-g1` |
| `upload_enabled` | boolean | `true` | false 时只落盘，但不得进入 uploaded 状态 |
| `upload_concurrency` | integer | `2` | 范围 1–8 |
| `multipart_threshold_mb` | integer | `64` | 大文件启用 multipart upload |
| `retry_max_seconds` | integer | `300` | 指数退避上限 |

### 7.2 Audio instance 配置

| 字段 | 类型 | 默认值 | 范围/行为 |
|---|---|---|---|
| `segment_seconds` | integer | `60` | 5–600，下一 segment 生效 |
| `local_retention_hours` | number | `24` | 1–720，只删除 uploaded_verified 数据 |
| `local_max_gb` | number | `4` | 该实例 spool 预算 |
| `container` | enum | `wav` | 第一版 `wav`；后续可增加 `flac` |
| `queue_seconds` | number | `5` | 内存队列预算，溢出计数 |
| `record_mode` | enum | `continuous` | 第一版固定为 continuous |

### 7.3 Video instance 配置

| 字段 | 类型 | 默认值 | 范围/行为 |
|---|---|---|---|
| `segment_seconds` | integer | `60` | 5–600 |
| `local_retention_hours` | number | `6` | 1–168，只删除 uploaded_verified 数据 |
| `local_max_gb` | number | `20` | 该实例 spool 预算 |
| `encoder` | enum | `nvv4l2h264enc` | Jetson 默认；x86 显式用 libx264 |
| `target_bitrate_kbps` | integer | `4000` | 256–20000 |
| `max_fps` | number | `30` | 超出时确定性丢帧并计数 |
| `queue_frames` | integer | `8` | 有界队列；不允许无限积压 |

### 7.4 配置预检

卡片启动前计算本地容量估算：

```text
estimated_bytes = bitrate_bytes_per_second × local_retention_hours × 3600
```

- PCM16 16 kHz mono 约 32 KB/s，约 115 MB/h；
- 4 Mbps H.264 约 1.8 GB/h；
- 若估算值超过 `local_max_gb × 0.8`，启动失败并给出建议；
- 实际数据率持续超过估算时进入 warning，并在 critical watermark 前暂停采集。

## 8. 本地落盘格式

### 8.1 目录

宿主机持久卷：

```text
/opt/phanthy-motus/inspection-data/
  audioinspector/<instance_id>/YYYY-MM-DD/HH/
    <utc_start_ns>_<seq>.wav
    <utc_start_ns>_<seq>.json
  videoinspector/<instance_id>/YYYY-MM-DD/HH/
    <utc_start_ns>_<seq>.mp4
    <utc_start_ns>_<seq>.json

/opt/phanthy-motus/inspection-state/
  ledger.sqlite3
  locks/
```

镜像升级、容器重建和卡片重启不能删除这两个宿主目录。

### 8.2 原子分片

每个 segment 经历：

```text
OPEN(.part) → FINALIZING → FINALIZED → UPLOADING
            → UPLOADED_VERIFIED → RETENTION_ELIGIBLE → PURGED_LOCAL
                              └── upload failed → FINALIZED/retry
```

规则：

1. 写入 `.part`；
2. 完成容器尾部、flush 和 fsync；
3. 计算 SHA-256 和文件大小；
4. 原子 rename 为正式文件；
5. 同样原子写入 metadata；
6. 在 SQLite ledger 中提交 FINALIZED；
7. 只有正式文件进入上传队列。

进程重启时：

- 扫描 ledger 与目录并做 reconciliation；
- UPLOADING 回退为 FINALIZED 后重试；
- 可修复的 `.part` 完成修复，否则标记 CORRUPT 并保留诊断；
- 已存在但 ledger 缺失的正式 segment 重新建账，不直接删除。

### 8.3 Metadata

每个 segment 对应一个 JSON：

```json
{
  "schema_version": "1.0",
  "segment_id": "seg-01K...",
  "kind": "audio",
  "device_id": "sh-g1",
  "card_id": "audioinspector",
  "instance_id": "canvas-card-123",
  "input_topic": "/g1/mic/audio",
  "format": "audio/pcm-16k",
  "ros_type": "audio_msgs/msg/AudioChunk",
  "session_id": "session-01K...",
  "sequence": 42,
  "source_stamp_start_ns": 1784246400000000000,
  "source_stamp_end_ns": 1784246459990000000,
  "receive_monotonic_start_ns": 123456789000,
  "receive_monotonic_end_ns": 183456789000,
  "wall_clock_start_utc": "2026-07-17T08:00:00.000Z",
  "wall_clock_end_utc": "2026-07-17T08:01:00.000Z",
  "duration_seconds": 60.0,
  "bytes": 1920044,
  "sha256": "...",
  "samples_or_frames": 960000,
  "dropped_before_writer": 0,
  "timestamp_gaps": []
}
```

音频 header stamp 可能为 0；此时 metadata 必须显式记录 `source_stamp_valid=false`，不能拿接收墙钟伪装为源时间戳。视频同样同时记录源 stamp、单调接收时间和墙钟，不用墙钟做单进程顺序判断。

## 9. COS 上报协议

### 9.1 对象键

```text
<cos_prefix>/<device_id>/<card_id>/<instance_id>/YYYY/MM/DD/HH/
  <utc_start_ns>_<seq>.<ext>
  <utc_start_ns>_<seq>.json
```

示例：

```text
inspection-data/sh-g1/audioinspector/canvas-card-123/2026/07/17/16/
  1784275200000000000_000042.wav
  1784275200000000000_000042.json
```

同一个 segment 的 object key 在重试时保持不变。不同实例、设备和卡片不得共享相同 key 空间。

### 9.2 上传确认

上传成功判据：

1. SDK upload 或 multipart complete 成功；
2. 对对象执行 HEAD；
3. 校验 Content-Length；
4. 上传时写入 `x-cos-meta-sha256`，HEAD 时核对；
5. 数据文件和 metadata 均通过后，ledger 才写 `UPLOADED_VERIFIED`。

不能只依赖 multipart ETag 等于本地 MD5。上传账本至少记录：

```text
segment_id, local_path, object_key, size, sha256, state,
attempts, first_seen_at, uploaded_at, verified_at, last_error
```

### 9.3 重试和幂等

- 指数退避加 jitter；
- 断网和 5xx 无限重试，间隔受 `retry_max_seconds` 限制；
- 401/403 进入 auth_error 并报警，不做高频重试；
- 同一 object key 若 HEAD 已存在且 size/hash 一致，直接恢复为 verified；
- 若 key 已存在但 hash 不同，进入 conflict，禁止覆盖并报警；
- 服务启动时优先上传最旧的 FINALIZED segment；
- 云端对象不由 Inspector 自动删除；COS 生命周期策略必须单独审查。

## 10. 本地滚动与磁盘保护

### 10.1 正常 TTL

segment 同时满足以下条件才允许删除本地数据和 metadata：

```text
state == UPLOADED_VERIFIED
and segment_end < now - local_retention_hours
```

ledger 保留 PURGED_LOCAL tombstone，默认 30 天，用于防止目录扫描时重复上传和审计。

### 10.2 Watermark

每个实例使用自己的 `local_max_gb`，同时检查宿主文件系统剩余空间：

| 水位 | 行为 |
|---|---|
| `< 70%` | 正常采集、上传和 TTL |
| `70–85%` | warning；提高已到期 verified 数据清理频率 |
| `85–95%` | high；拒绝增大留存时间，持续报警 |
| `≥ 95%` | critical；完成当前 segment 后暂停采集，不删除未上传数据 |

恢复到 80% 以下且上传/清理正常后，可以人工或配置允许自动 resume。第一版默认人工 resume，避免空间抖动造成反复启停。

## 11. 并发与故障隔离

每个实例至少包含：

```text
ROS Subscriber → bounded ingest queue → Segment Writer
                                          │
                                          ▼
                                      SQLite Ledger
                                          │
                          shared Upload Scheduler → COS workers
                                          │
                                      Retention Sweeper
```

- 每个卡片实例独立 ingest queue 和 writer；
- Upload Scheduler 可在 bundle 内共享，但不能持有 ROS callback 锁；
- SQLite 使用 WAL，状态转移放在短事务中；
- 编码进程退出只影响对应 Video Inspector 实例；
- 某一路 auth_error 不应停止另一张卡片本地落盘；
- 服务退出时先停止订阅，再 finalize segment，最后提交 ledger；不等待无限上传。

## 12. Agent Core 与 Dashboard 改动

### 12.1 Agent Core

- `drivers.py`：为 inspection image 补充 `mcp_url=http://localhost:15671/mcp`；
- 保留 embedded `api/inspection.py` 作为实时预览模块；
- MCP 注册仍使用 `category=inspection`；
- 不把 Inspector tools 暴露给 LLM 的普通物理执行规划，或默认标记为 non-agent-callable；
- 后续可增加采集状态事件，但第一版通过 `info` 和 Dashboard 轮询即可。

### 12.2 Dashboard

- `sidebar.js` 增加 Inspection 分组；
- `_TYPE_ORDER` 增加 `inspector`；
- Canvas 增加 Inspector badge、颜色和 input-only sink 样式；
- Flow View 按 tool type/category 显示 Inspector，不能继续按 topic 方向推断成 actuator；
- 卡片详情展示：recording state、当前 segment、输入码率、队列、drop、local bytes、backlog、last upload、last error；
- 项目启动时按连线传入 input topic；断线或项目停止时调用 stop 并 finalize；
- COS Secret 不在浏览器回显。

## 13. 代码目录建议

```text
inspection/
  main.py                         # MCP bundle、注册与进程生命周期
  config.yaml
  Dockerfile
  requirements.txt
  deploy/service.yml
  plugins/
    audioinspector/
      __init__.py
      plugin.py
      node.py
      writer.py
    videoinspector/
      __init__.py
      plugin.py
      node.py
      encoder.py
  storage/
    models.py                     # segment 状态与 schema
    ledger.py                     # SQLite WAL 与恢复
    atomic_writer.py
    cos_backend.py
    upload_scheduler.py
    retention.py
    secret_loader.py
  tests/
    test_audio_writer.py
    test_video_writer.py
    test_ledger_recovery.py
    test_cos_retry.py
    test_retention.py
    test_disk_pressure.py
    test_mcp_contract.py

deploy/
  build_inspection.sh

agent-core/
  src/api/drivers.py
  web/js/sidebar.js
  web/js/canvas.js
  web/js/flow-view.js
  web/css/style.css
```

现有 `data-upload/cos-uploader` 作为经过验证的设计参考和测试来源，不作为运行时 git/path 依赖。Inspection Stack 需要更严格的 per-segment ledger、HEAD 校验和 TTL 门禁，直接复制其 size-only manifest 不足以满足本验收标准。

## 14. 分阶段实施

### Gate 0：目标 topic 与运行环境确认

- 在上海 G1 只读确认麦克风、摄像头 topic、ROS type、format、QoS、频率和消息大小；
- 确认 Jetson 可用的视频编码器；
- 确认 `/opt/phanthy-motus` 可用空间和数据卷挂载点；
- 使用测试 prefix 验证 COS profile 的 PutObject 和 HeadObject 权限；
- 确认目标 COS prefix 没有会提前删除对象的生命周期策略。

### Gate 1：Inspector contract 与 Dashboard

- 增加 `inspector` tool type；
- Inspection Bundle 能注册到 Core 并出现在左侧分组；
- 两张卡片可添加、连线、配置、start/stop；
- 此阶段用 fake writer，不触碰真实 COS。

### Gate 2：本地音频录制

- 完成 AudioChunk 订阅、固定时长 WAV、metadata、原子 rename；
- 完成 stop/flush/restart recovery；
- 用 ROS2 回放和真实 G1 mic 验证连续性。

### Gate 3：本地视频录制

- 完成 CompressedImage 订阅、Jetson H.264 编码和 MP4 分片；
- 验证帧率、时长、可播放性、CPU/GPU/内存和 dropped frame；
- 编码器不可用时必须明确失败。

### Gate 4：COS 与滚动留存

- 完成 ledger、COS backend、HEAD/hash 校验和断网补传；
- 完成只删除 uploaded_verified 的 TTL；
- 完成 disk watermark、暂停和告警；
- 使用真实 COS 测试 prefix 验证对象完整性。

### Gate 5：上海 G1 联合验收

- Dashboard 真实连线；
- 音频、视频同时采集；
- 正常网络、断网、重连、服务重启和留存到期；
- 核对本地 ledger、文件数量、COS 对象数量/hash 和资源占用；
- 用户确认后再推送分支和创建 PR。

## 15. 验收标准

### 15.1 功能验收

1. Inspection Bundle 注册后，Dashboard 出现 Audio Inspector 和 Video Inspector。
2. 麦克风 → Audio Inspector、摄像头 → Video Inspector 可以按 format 正确连线。
3. 未配置必填 COS 地址时项目拒绝启动并给出明确错误。
4. 配置测试 prefix 后，`testupload` 可 Put + HEAD 校验。
5. 项目启动后，本地自动生成可播放音频和视频 segment 及 metadata。
6. segment 达到配置时长后自动 finalize，并在后台上传 COS。
7. 停止项目时当前 segment 被正确 finalize，已完成数据继续上传。
8. `info` 能看到 recording、local、backlog、uploaded、drop 和 error 指标。

### 15.2 可靠性验收

1. 断网期间持续落盘，不阻塞上游 Driver 和 Core。
2. 恢复网络后，断网期间所有 finalized segment 自动补传。
3. 强制重启 Inspection 容器后，ledger 恢复且不重复生成冲突对象。
4. COS 已存在且 size/hash 相同的对象被幂等确认，不重复覆盖。
5. 普通 TTL 永远不删除未上传或未校验 segment。
6. 达到 critical watermark 时暂停采集并报警，不静默删除未上传数据。
7. COS 上传失败不会导致音频和视频 writer 线程退出。

### 15.3 数据完整性验收

1. 每个本地正式 segment 都有 metadata 和 ledger 记录。
2. 每个 UPLOADED_VERIFIED segment 在 COS 中都有数据和 metadata 对象。
3. 本地 SHA-256 与 COS metadata 一致。
4. 正常负载下 Audio Inspector 的 callback-to-writer drop 为 0。
5. 正常负载下 Video Inspector 的 drop 为 0；超过 `max_fps` 的确定性限帧单独统计，不伪装成异常丢帧。
6. 音频连续播放无明显断点；视频每段可独立打开，时长和 metadata 误差在约定范围内。

### 15.4 滚动机制验收

自动化测试使用 `segment_seconds=10`、`local_retention_hours=1`，并通过 Retention Sweeper 的可注入 clock 把当前时间向前推进；不为缩短测试而放宽生产配置下限。真机长跑再使用真实墙钟验证。

1. 上传确认前，即使超过 retention，文件仍保留；
2. 上传确认且超过 retention 后，本地文件被清理；
3. COS 对象仍全部保留；
4. ledger tombstone 保留并能防止重传；
5. Inspector 连续运行时本地占用稳定在配置预算内。

### 15.5 性能验收

Gate 0 实测后冻结具体阈值，至少记录：

- Audio Inspector CPU、RSS、写盘带宽、队列深度；
- Video Inspector CPU/GPU、RSS、编码 fps、bitrate、写盘带宽；
- 两卡同时运行时 Agent Core、Perception 和 Driver 的延迟/丢帧变化；
- 网络上传带宽和 backlog 清空速度；
- 1 小时稳定性与 24 小时滚动测试。

## 16. 测试矩阵

| 场景 | 期望 |
|---|---|
| 正常音频 | 连续 WAV 分片、上传和清理 |
| 正常视频 | 可播放 MP4 分片、上传和清理 |
| 两卡并发 | 互不阻塞，统计独立 |
| 输入断流 | 当前 segment 正确结束或等待超时，state=degraded |
| DDS QoS 不匹配 | start 返回诊断，不假装 running |
| COS 断网 | 本地 backlog 增长，持续重试 |
| COS 403 | auth_error，停止高频重试但继续落盘 |
| COS 恢复 | 从最旧 segment 开始补传 |
| 容器重启 | 恢复 ledger、part 和 pending segment |
| 磁盘 high | 报警并加快 eligible 清理 |
| 磁盘 critical | finalize 后暂停，不删 unuploaded |
| TTL 到期但未上传 | 不删除 |
| TTL 到期且已验证 | 删除本地，保留云端 |
| 重复 start/stop | 幂等，无重复订阅和文件冲突 |
| 时间戳为 0/跳变 | metadata 标记异常，不破坏 segment 顺序 |

## 17. 风险与待确认

1. 上海 G1 当前实际麦克风/摄像头 topic、ROS type、format 和 QoS 尚需 Gate 0 真机确认。
2. Jetson 上可用的视频硬编码插件、像素格式转换和稳定码率需要实测。
3. 视频默认 4 Mbps 时 6 小时本地留存约需 10.8 GB，加上 backlog 和安全余量后默认 20 GB 是否合适需按设备磁盘确认。
4. COS 子账号需要 PutObject、HeadObject 和 multipart 相关最小权限；若只给 PutObject，无法完成严格上传确认。
5. 当前 Agent Core 通用 tool config 不是 secret store；若产品要求在 Dashboard 直接录入 SecretKey，需要先增加独立的加密 secret 管理，不能复用现有明文配置接口。
6. 当前 MCP tools 会进入 LLM tool registry。Inspector 生命周期原则上由项目画布管理，是否对 LLM 隐藏需要在实现前确定，推荐默认隐藏。
7. 当前 Inspector Bundle 端口 `15671` 与历史独立 `dds_inspection` 语义有重叠，命名和 Resource Center 镜像元数据需要同步调整，避免把录制服务误认为旧实时代理。

## 18. 推荐结论

第一版采用：

- 一个独立 Inspection Stack 镜像；
- 两张 `type=inspector`、multiInstance 的 input-only 卡片；
- 音频 WAV/视频 H.264 MP4 固定时长不可变分片；
- 本地 SQLite ledger + 原子文件；
- 异步 COS 上传、HEAD/size/SHA-256 校验；
- 只有 uploaded_verified 数据才能按 TTL 滚动删除；
- 未上传数据遇到磁盘 critical 时暂停采集，不静默删除；
- COS 凭证由部署 secret 提供，卡片只配置 region/bucket/prefix/device/profile；
- 先完成上海 G1 topic/编码器/COS 权限 Gate 0，再进入实现。

该方案满足画布连线、可配置上报地址、自动保存、断网补传、本地可控留存和云端完整保存的目标，同时把有限磁盘下不可回避的暂停边界显式化，避免用静默丢数据制造“看起来一直在采”的假可靠性。
