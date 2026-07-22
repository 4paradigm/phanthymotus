# Inspection Stack

独立的音视频采集卡片宿主。当前分支已完成 Dashboard 编排、Audio Inspector 持久化采集、Video Inspector 硬件编码与 MP4 分片、COS 校验上传、断点恢复和本地滚动清理。基础采集、双条件分片、磁盘指标与 COS 已在上海 G1 真机通过；网页交互验收中发现的原生视频管线异常停止路径已增加显式收尾与状态传播，待新镜像复验。

## 画布使用门禁

- 未配置的 Audio / Video Inspector 可以先拖入画布，卡片会显示“待配置”或“待连线”。
- 每张 Inspector 必须恰好连接一条 format 完全匹配的输入：音频为 `audio/pcm-16k`，视频为 `image/jpeg`。
- 点击“开启智能控制”时，Dashboard 在调用任何 `start` 前检查全部 Inspector 的连线和存储配置；任意一张不合格则整个项目不启动。
- Inspector 配置由服务端同步校验成功后才写入 Agent Core；`local_and_cos` 会在保存时写入并 HEAD 校验一个最小健康对象，桶不存在、region 不匹配或无权限都会直接拒绝保存并显示原因。
- 使用影石 MJPEG 中继时，External Camera 实例配置为 `source_mode=mjpeg`、`source_url=http://127.0.0.1:8002/stream`、`reconnect_delay=1`；停止项目后连接 `External Camera → Video Inspector`，再重新开启智能控制。只保存配置不会启动实例。
- Agent Core 重启后不保留“智能控制运行中”的 UI 假状态；页面回到未启动，用户需要显式再次开启，以重建动态卡片实例和 topic。

## 采集生命周期

- Inspection 容器可以常驻，但卡片默认为 `idle`，不订阅数据。
- 点击“开始智能控制”并触发卡片 `start` 后，才创建 ROS2 订阅并开始生成新分片。
- `stop` 先停止新数据进入，再排空有界 writer 队列并 finalize 当前分片；已完成分片的补传不应被取消。
- Jetson 解码/编码管线已进入终止错误时，`stop` 先中止管线以解除阻塞的 `appsrc push-buffer`，丢弃尚未编码的有界队列，并将未完成 MP4 保留为 `CORRUPT` 诊断文件。
- 收尾失败不得继续回报 `recording`：订阅已停止时返回 `stop_error`/`recording=false`，Dashboard 显示“采集已停止，但当前分片收尾异常”并保留错误详情。
- Video Inspector 启动后 10 秒内没有合法 JPEG 首帧时报 `input_start_timeout`；正常收帧后连续 5 秒断流时报 `input_stalled`；缺失 JPEG SOI/EOI 标记时报 `invalid_jpeg`。错误持久化到 ledger，重启后仍在卡片显示，不再只呈现“正在采集”。
- COS uploader 是独立长驻 worker；点击“停止智能控制”不再产生新数据，但会继续补传 ledger backlog。
- 配置通过后发生的断网或权限变化不会删除本地数据，也不会隐式切换为 `local_ring`；卡片会显示“云端上传失败”、错误原因、待上传量和重试间隔，本地采集可在磁盘水位允许时继续。
- 异常退出后启动时先恢复 `.part` 和 ledger。`auto_resume_after_reboot=false` 时保持 `idle` 并返回 `resume_required=true`，不会偷偷继续采集。
- Inspection 容器使用 `on-failure:3` 隔离 Jetson 原生插件导致的非零退出：运行期异常最多自动重启 3 次，但 Docker daemon 或整机重启时不会自动拉起，不改变宿主服务先就绪、再启动容器的顺序。
- 重启时当前卡片的 `UPLOADING` 回退到 `FINALIZED`；先通过 HEAD 识别已完整上传的对象，否则从本地正式分片重新上传，超过 `multipart_stale_hours` 的遗留 upload 尽力终止。
- 卡片每 5 秒刷新本地占用、实例上限、文件系统水位、预计剩余可录时间和上传 backlog；项目停止后仍继续刷新，直到 backlog 清空。

## 存储模式与分片

- `storage_mode=local_and_cos`（默认）：本地原子落盘后异步上传；只有 `UPLOADED_VERIFIED` 的媒体/metadata 才可被本地留存策略删除。
- `storage_mode=local_ring`：不要求 COS，允许按 `local_retention_hours` 和 `local_max_gb` 滚动删除最旧本地分片。这是显式选择的本地有损模式，不会由 COS 失败自动降级得到。
- `upload_enabled` 仅用于兼容旧配置，WebUI 不再显示；与 `storage_mode` 冲突时配置保存失败。
- `device_id` 从 Jetson 硬件序列号自动生成（上海 G1 为 `jetson-<serial>`），WebUI 不显示也不接受用户覆盖；旧配置中的 `device_id` 会被忽略。
- 音频默认按 `segment_seconds=60` 或 `max_segment_mb=4` 先到者切成 WAV。
- 视频默认按 `segment_seconds=60` 或 `max_segment_mb=64` 先到者切成 H.264 MP4。
- `flush` 和 `stop` 都会强制结束当前分片；第一版不按 VAD、画面变化或动作语义切片。

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

COSBrowser / coscli 的 `mode=SecretKey` 配置会将 SecretId / SecretKey 加密存储，不能把 YAML 中的 64 字符密文直接写入上述 JSON。部署工具必须通过官方 coscli 解密通道把凭据直接写入 `0600` 临时文件，且不得把 `coscli config show` 的原始输出发往终端或日志。

每个 segment 的媒体文件和 JSON metadata 作为两个不可变对象上传。上传后必须通过 HEAD 同时校验 `Content-Length` 与 `x-cos-meta-sha256`，才记为 `UPLOADED_VERIFIED`。远端同名对象内容不一致时进入 `CONFLICT`，不覆盖。

保存 `local_and_cos` 配置本身就会执行一次相同的小对象上传 + HEAD 校验；`testupload` 保留为独立诊断动作。配置校验失败时 Agent Core 不持久化新值。配置成功后才发生的网络故障由后台 worker 自动重试，卡片通过 `upload_state=error` 和 `upload_error` 明确暴露，不会把失败状态只显示成“后台上传中”。

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
- 新分片目录为 `<audio-inspector|video-inspector>/<输入源语义名>--<8位实例哈希>/utc-hour=YYYY-MM-DDTHHZ/<UTC时间>--<序号>.<扩展名>`；例如 `video-inspector/ext-camera-rgb--a1b2c3d4/utc-hour=2026-07-22T09Z/20260722T090737.590123456Z--000000.mp4`；
- 画布内部 `instance_id`（如 `card-mrvusdyxxjln`）不再直接进入路径，但完整值和原始 `input_topic` 仍写入 metadata/ledger；短哈希只用于防止同源多实例目录冲突；
- 旧版 `audioinspector/<instance>/YYYY-MM-DD/HH` 与 `videoinspector/<instance>/YYYY-MM-DD/HH` 数据不自动迁移或删除，恢复、补传和滚动清理继续兼容；
- 音频输入固定为 `audio_msgs/AudioChunk`、`audio/pcm-16k`、PCM_S16_LE 16 kHz mono；
- 音频 QoS：`BEST_EFFORT + KEEP_LAST(50) + VOLATILE`；
- 视频输入固定为 `sensor_msgs/CompressedImage`、`image/jpeg`，QoS 为 `BEST_EFFORT + KEEP_LAST(2) + VOLATILE`；
- Jetson 管线固定为 `nvjpegdec → nvvidconv → nvv4l2h264enc → splitmuxsink`；指定硬件编码器不可用时启动失败，不做隐式 CPU 降级；
- 上海 G1 的 Docker daemon 没有名为 `nvidia` 的 runtime，release 基础镜像还包含未使用的 Jetson GStreamer 零字节占位插件；镜像构建时会删除这些无效插件，部署时只读挂载管线必需的 `libgstnvjpeg.so` / `libgstnvvidconv.so` / `libgstnvvideo4linux2.so`、`tegra` / `tegra-egl` 运行库和 `libgstnvexifmeta.so`，避免扫描 Argus/EGL 等无关插件，不依赖 `runtime: nvidia`；
- 当 release 基础镜像仍使用已无法解析的 `mirrors.tencentyun.com/ubuntu-ports` 时，Dockerfile 会在安装依赖前改用上海 G1 实测最快的中科大 `https://mirrors.ustc.edu.cn/ubuntu-ports`；2026-07-22 对 `jammy/main` 索引单次实测约 4.74 MB/s，`jammy-updates/main` 三次约 5.8–6.7 MB/s，四个 pocket 均返回 200；可用 `UBUNTU_PORTS_MIRROR` build arg 显式覆盖；
- 联合验收会同时构建 Agent Core；Core Dockerfile 也使用同一 `UBUNTU_PORTS_MIRROR` build arg 替换失效的腾讯镜像源，部署脚本必须向 Core 和 Inspection 两次 `docker build` 传入同一值；
- Jammy 的 `apt-get` 不接受 `--no-triggers`（该语义属于更底层的 `dpkg` 选项）；Core 在 G1 原生 arm64 构建时按正常 apt 流程处理 libc triggers，不再传入无效参数；
- Python 依赖通过 `pip --target /opt/inspection-python --ignore-installed` 安装到隔离目录，不卸载或覆盖 Jetson release 镜像中由 Ubuntu/distutils 管理的 PyYAML 5.3.1；镜像构建会断言运行时实际加载隔离的 PyYAML 6.0.2 和 COS SDK；
- 写入路径：`.wav.part` / `.mp4.part` → fsync → 媒体/JSON 原子 rename → SQLite `FINALIZED`；
- 异常退出留下的 MP4 先由 `gst-discoverer-1.0` 校验，可读才恢复，否则保留为 `CORRUPT` 诊断文件；
- SQLite 使用 WAL 和 `synchronous=FULL`，正式文件存在但账本缺失时可扫描重建。

## 本地滚动策略

- `local_retention_hours` 控制本地保留时间，`local_max_gb` 控制实例最大 spool 预算；
- Audio / Video Inspector 的 `corrupt_retention_hours` 控制不可恢复分片及诊断文件的保留时间，默认 24 小时；
- `local_and_cos` 只有 `UPLOADED_VERIFIED` 数据可进入删除流程；`local_ring` 可对到期或超预算的本地 `FINALIZED` 数据执行显式环形清理；两种模式最终都经过 `RETENTION_ELIGIBLE → PURGED_LOCAL`；
- 超预算时先从最旧的已验证数据开始删除；若剩余数据均未上传，暂停该 Inspector 并报 `paused_disk_full`，不静默丢数据；
- 删除媒体、metadata 或过期 corrupt 诊断文件后，会从小时目录向上清理空目录；每小时额外清扫历史空目录；
- 启动前会估算留存窗口所需容量，超过 `local_max_gb` 的 80% 则拒绝启动并给出错误。
- 实例预算水位和宿主文件系统水位取较高者：70% warning、85% high、95% critical；critical 时停止接收新输入、完成当前分片后暂停。

## 当前边界

- 2026-07-21 已在上海 G1 实际构建并部署 `phanthymotus/inspection:local-7624db48f262-g1-test`；Jetson GStreamer 硬件管线预检、Core 注册和两张 Inspector 发现均通过；
- Audio Inspector 已从 `/phanthymotus_g1_driver/mic/audio` 采集真实 `AudioChunk`，完成 WAV/JSON 成对落盘、COS 上传及 HEAD 大小/SHA-256 校验；
- Video Inspector 已从 `/phanthymotus_g1_driver/ext_camera/card-mrnbwcls6nji/rgb` 采集真实 `CompressedImage`，使用 Jetson NVENC 完成 MP4/JSON 成对落盘、播放发现、COS 上传及 HEAD 校验；
- 最终真实验收前缀为 `cos://embodied-ai-1252788780/inspection-acceptance/20260721-164837/`，共 12 个对象；验收后 Audio / Video Inspector 和 ext_camera 均为 `idle`，上传 backlog 为 0；
- 2026-07-21 与 2026-07-22 各出现一次 Jetson NVENC native 进程退出，并留下可诊断的零字节 `.part`；同一真实 JPEG 的宿主、容器、Python appsrc、双线程、原样 Runtime 和真实 ROS executor 隔离测试均通过，说明是低频原生编码路径故障而非稳定可复现的 Python 异常。容器现以 `on-failure:3` 限制运行期重启次数，验收只在确认容器发生非零退出并恢复后显式重试一次，不切换软件编码器；长时间连续稳定性仍待 soak test；
- COS 上传、HEAD 校验、冲突保护、重试与本地滚动的 fake backend 测试和真实 bucket `testupload` 均已通过；断网、强制退出后的 backlog 补传及长时间滚动删除仍待专项验收；
- COS 凭证只能由部署 secret 提供，不进入卡片配置或日志。
- G1 容器使用 `restart: "on-failure:3"`：只处理运行期非零退出，最多重试 3 次；Docker daemon/整机重启时不会自动拉起，因此仍必须在宿主相机、音频和 ROS2 服务就绪后按顺序启动。容器恢复后会自动恢复 ledger backlog，但 `auto_resume_after_reboot=false` 时不会自动恢复采集实例。

## G1 部署后验收

1. 确认 `embodied-inspection` 运行，`curl http://127.0.0.1:15671/health` 返回 `ok=true`。
2. 先在未配置、未连线状态把两张 Inspector 拖入 Dashboard，确认卡片可添加且项目启动被明确拒绝。
3. 验证错误 format 无法连线；再分别连接 `/phanthymotus_g1_driver/mic/audio` 和由 External Camera `start` 返回的 `/phanthymotus_g1_driver/ext_camera/<card-id>/rgb`，不得用内置 `camera_rgb` 的 topic 代替影石验收。
4. 配置 `storage_mode=local_and_cos`、`cos_region`、`cos_bucket`、`cos_prefix`、`credential_profile`；确认界面没有 `device_id` 输入框，保存后返回自动设备 ID 和 `upload_validation=verified`。
5. 故意填写一个格式正确但不存在的 bucket，确认配置保存被拒绝且界面明确显示“bucket 不存在或与 region 不匹配”；恢复正确 bucket 后再继续。
6. 点击“开启智能控制”，确认两张卡片均进入“正在采集”，并能看到本地用量、剩余可录时间和磁盘水位。
7. 点击 External Camera 的“查看数据流”，确认 WebSocket 先返回 topic metadata，再连续返回 SOI=`FFD8`、EOI=`FFD9` 的完整 JPEG；未启动时必须明确显示“数据源尚未启动”，不能静默无反应。
8. 使用较小验收值验证按时长分片，再单独使用较小 `max_segment_mb` 验证按大小分片；停止项目时当前段必须被 finalize。
9. 检查本地 WAV/MP4 和 JSON 成对存在，媒体可播放，ledger 状态最终为 `UPLOADED_VERIFIED`。
10. 检查本地与 COS 路径使用自动设备 ID、`audio-inspector`/`video-inspector`、输入源语义名和显式 `utc-hour=...Z`，且媒体和 metadata 的 size/SHA-256 一致。
11. 停止智能控制后确认不再产生新分片，但卡片仍显示 backlog 减少，最终变成“云端已同步”。
12. 另建 `local_ring` 实例验证无需 COS 可启动，过期或超预算后只滚动本地数据，不创建 COS 对象。

视频验收前必须先确认输入 topic 能在超时窗口内收到至少一帧 `sensor_msgs/msg/CompressedImage`。只有 publisher 数量不为零并不代表相机真实产帧；门禁失败时应先修复上游相机服务，不能用空 MP4 作为通过证据。
