# Inspection Stack

独立的音视频采集卡片宿主。当前分支已完成 Gate 1 卡片与 Dashboard 编排、Gate 2 Audio Inspector 持久化采集，以及 Gate 3 Video Inspector 的 ROS2 订阅、Jetson GStreamer 硬件编码、MP4 原子分片和异常退出恢复。COS 上传和本地滚动清理尚未接入。

## 采集生命周期

- Inspection 容器可以常驻，但卡片默认为 `idle`，不订阅数据。
- 点击“开始智能控制”并触发卡片 `start` 后，才创建 ROS2 订阅并开始生成新分片。
- `stop` 先停止新数据进入，再排空有界 writer 队列并 finalize 当前分片；已完成分片的补传不应被取消。
- 异常退出后启动时先恢复 `.part` 和 ledger。`auto_resume_after_reboot=false` 时保持 `idle` 并返回 `resume_required=true`，不会偷偷继续采集。

## 本地运行

```bash
cd inspection
source /opt/ros/humble/setup.bash
source /ros_ws/install/setup.bash
python3 -m pip install -r requirements.txt
python3 main.py
```

默认配置需要 ROS2、`audio_msgs`、`sensor_msgs`和 GStreamer Python 绑定。不安装 ROS2 的普通开发机只运行单元测试，不能启动真实采集 runtime。

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
- 写入路径：`.wav.part` / `.mp4.part` → fsync → 媒体/JSON 原子 rename → SQLite `FINALIZED`；
- 异常退出留下的 MP4 先由 `gst-discoverer-1.0` 校验，可读才恢复，否则保留为 `CORRUPT` 诊断文件；
- SQLite 使用 WAL 和 `synchronous=FULL`，正式文件存在但账本缺失时可扫描重建。

## 当前边界

- Audio Inspector 为 `runtime_mode=ros2-durable-audio`、`storage_ready=true`，已真实本地落盘；
- Video Inspector 为 `runtime_mode=ros2-gstreamer`、`storage_ready=true`，已实现本地 MP4 落盘；
- 代码与 G1 上的 GStreamer 元素/管线解析已验证，真实相机录制、播放和长时间稳定性仍需部署后实机验收；
- `testupload` 返回 `unsupported`，直到 Gate 4 接入真实 COS backend；
- COS 凭证只能由部署 secret 提供，不进入卡片配置或日志。
