# Inspection Stack

独立的音视频采集卡片宿主。当前分支已完成 Gate 1 卡片与 Dashboard 编排、Gate 2 Audio Inspector 持久化采集、Gate 3 Video Inspector 硬件编码与 MP4 分片，以及 Gate 4–5 COS 校验上传、断点恢复和本地滚动清理。代码闭环已完成，真实 G1 采集与 COS 仍需部署验收。

## 采集生命周期

- Inspection 容器可以常驻，但卡片默认为 `idle`，不订阅数据。
- 点击“开始智能控制”并触发卡片 `start` 后，才创建 ROS2 订阅并开始生成新分片。
- `stop` 先停止新数据进入，再排空有界 writer 队列并 finalize 当前分片；已完成分片的补传不应被取消。
- COS uploader 是独立长驻 worker；点击“停止智能控制”不再产生新数据，但会继续补传 ledger backlog。
- 异常退出后启动时先恢复 `.part` 和 ledger。`auto_resume_after_reboot=false` 时保持 `idle` 并返回 `resume_required=true`，不会偷偷继续采集。
- 重启时当前卡片的 `UPLOADING` 回退到 `FINALIZED`；先通过 HEAD 识别已完整上传的对象，否则从本地正式分片重新上传，超过 `multipart_stale_hours` 的遗留 upload 尽力终止。

## 本地运行

```bash
cd inspection
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash
python3 -m pip install -r requirements.txt
python3 main.py
```

默认配置需要 ROS2、`audio_msgs`、`sensor_msgs` 和 GStreamer Python 绑定。不安装 ROS2 的普通开发机只运行单元测试，不能启动真实采集 runtime。

健康检查：

```bash
curl http://127.0.0.1:15671/health
```

查看两张卡片：

```bash
curl -s http://127.0.0.1:15671/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

## COS 凭证与校验

生产环境默认从只读 secret 文件取凭证，卡片只保存 `credential_profile=default`：

```text
/run/secrets/phanthymotus/cos/default.json
```

```json
{
  "secret_id": "<COS SecretId>",
  "secret_key": "<COS SecretKey>",
  "token": ""
}
```

宿主文件权限应设为 `0600`。本地开发才允许用 `COS_SECRET_ID` / `COS_SECRET_KEY` / `COS_SESSION_TOKEN` 环境变量兼容。日志、MCP 返回和 Agent Core 配置均不回显凭证。

G1 上的持久化宿主路径为 `/opt/phanthy-motus/secrets/phanthymotus/cos/default.json`，通过只读挂载映射到上述容器路径。不要把 secret 放在宿主 `/run`：该目录重启后可能被清空。部署脚本只在镜像构建和多媒体预检通过后才上传临时 secret，安装后无论成功或失败都会删除远端暂存副本。

每个 segment 的媒体文件和 JSON metadata 作为两个不可变对象上传。上传后必须通过 HEAD 同时校验 `Content-Length` 与 `x-cos-meta-sha256`，才记为 `UPLOADED_VERIFIED`。远端同名对象内容不一致时进入 `CONFLICT`，不覆盖。

配置后可通过 MCP 做小对象上传 + HEAD 校验：

```bash
curl -s http://127.0.0.1:15671/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"audioinspector","arguments":{"action":"testupload"}}}'
```

运行测试：

```bash
python3 -m unittest discover -s inspection/tests -v
python3 -m unittest discover -s agent-core/tests -v
node --check agent-core/web/js/sidebar.js
node --check agent-core/web/js/canvas.js
node --check agent-core/web/js/flow-view.js
```

## 落盘与恢复

- 持久化目录：`/opt/phanthy-motus/inspection-data`；
- 账本：`/opt/phanthy-motus/inspection-state/ledger.sqlite3`；
- 音频输入固定为 `audio_msgs/AudioChunk`、`audio/pcm-16k`、PCM_S16_LE 16 kHz mono；
- 音频 QoS：`BEST_EFFORT + KEEP_LAST(50) + VOLATILE`；
- 视频输入固定为 `sensor_msgs/CompressedImage`、`image/jpeg`，QoS 为 `BEST_EFFORT + KEEP_LAST(2) + VOLATILE`；
- Jetson 管线固定为 `nvjpegdec → nvvidconv → nvv4l2h264enc → splitmuxsink`；指定硬件编码器不可用时启动失败，不做隐式 CPU 降级；
- 上海 G1 的 Docker daemon 没有名为 `nvidia` 的 runtime，release 基础镜像中的 Jetson GStreamer 插件又是占位文件；部署契约因此从宿主只读挂载 `tegra` / `tegra-egl` 运行库、三个管线插件和 `libgstnvexifmeta.so`，并设置对应 `LD_LIBRARY_PATH`，不依赖 `runtime: nvidia`；
- 当 release 基础镜像仍使用已无法解析的 `mirrors.tencentyun.com/ubuntu-ports` 时，Dockerfile 会在安装依赖前改用已验证可达的官方 `https://ports.ubuntu.com/ubuntu-ports`；可用 `UBUNTU_PORTS_MIRROR` build arg 显式覆盖；
- PyYAML 约束为 `>=5.3.1,<7`：Jetson release 基础镜像已由 Ubuntu/distutils 提供 5.3.1 时直接复用，避免 pip 尝试卸载系统包；其他镜像缺失时仍会自动安装兼容版本；
- 写入路径：`.wav.part` / `.mp4.part` → fsync → 媒体/JSON 原子 rename → SQLite `FINALIZED`；
- 异常退出留下的 MP4 先由 `gst-discoverer-1.0` 校验，可读才恢复，否则保留为 `CORRUPT` 诊断文件；
- SQLite 使用 WAL 和 `synchronous=FULL`，正式文件存在但账本缺失时可扫描重建。

## 本地滚动策略

- `local_retention_hours` 控制本地保留时间，`local_max_gb` 控制实例最大 spool 预算；
- 只有 `UPLOADED_VERIFIED` 数据可进入删除流程，并在断点可恢复的 `RETENTION_ELIGIBLE → PURGED_LOCAL` 状态间迁移；
- 超预算时先从最旧的已验证数据开始删除；若剩余数据均未上传，暂停该 Inspector 并报 `paused_disk_full`，不静默丢数据；
- 启动前会估算留存窗口所需容量，超过 `local_max_gb` 的 80% 则拒绝启动并给出错误。

## 当前边界

- Audio Inspector 为 `runtime_mode=ros2-durable-audio`、`storage_ready=true`，已真实本地落盘；
- Video Inspector 为 `runtime_mode=ros2-gstreamer`、`storage_ready=true`，已实现本地 MP4 落盘；
- G1 宿主的 NVIDIA GStreamer 元素和管线解析已验证；旧 Perception 容器缺少 Python GStreamer typelib、`gst-discoverer-1.0`，且 ROS setup 路径为 `/opt/ros/humble/install/setup.bash`。本分支 Dockerfile 已补齐依赖并兼容两种 ROS 布局，但新镜像尚未在 G1 实际构建；
- 真实相机录制、播放和长时间稳定性仍需部署后实机验收；
- COS 上传、HEAD 校验、冲突保护、重试与本地滚动的 fake backend 测试已通过，真实 bucket 的 `testupload` 尚未执行；
- COS 凭证只能由部署 secret 提供，不进入卡片配置或日志。
- G1 容器使用 `restart: "no"`，必须在宿主相机、音频和 ROS2 服务就绪后按顺序启动；容器启动后会自动恢复 ledger backlog，但 `auto_resume_after_reboot=false` 时不会自动恢复采集实例。

## G1 部署后验收

1. 确认 `embodied-inspection` 运行，`curl http://127.0.0.1:15671/health` 返回 `ok=true`。
2. 在 Dashboard 加入 Audio Inspector 和 Video Inspector，分别连接 `/phanthymotus_g1_driver/mic/audio` 和 `/phanthymotus_g1_driver/camera/rgb`。
3. 配置 `cos_region`、`cos_bucket`、`cos_prefix`、`device_id`、`credential_profile`，先执行 `testupload`。
4. 点击“开启智能控制”，等待至少一个 `segment_seconds`，再点击停止。
5. 检查本地 WAV/MP4 和 JSON 成对存在，媒体可播放，ledger 状态最终为 `UPLOADED_VERIFIED`。
6. 检查 COS 对象键为 `<prefix>/<device>/<card>/<instance>/YYYY/MM/DD/HH/<file>`，且媒体和 metadata 的 size/SHA-256 与本地一致。
7. 停止智能控制后确认不再产生新分片，但已有 backlog 仍会减少；最后执行一次断网、强制退出和重启恢复验收。

视频验收前必须先确认输入 topic 能在超时窗口内收到至少一帧 `sensor_msgs/msg/CompressedImage`。只有 publisher 数量不为零并不代表相机真实产帧；门禁失败时应先修复上游相机服务，不能用空 MP4 作为通过证据。
