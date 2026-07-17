# Inspection Stack

独立的音视频采集卡片宿主。当前分支已完成 Gate 1 卡片与 Dashboard 编排，以及 Gate 2 Audio Inspector 的 ROS2 订阅、WAV 原子分片、SQLite 账本和异常退出恢复。Video Inspector 目前仍为合同模式，COS 上传和本地滚动清理尚未接入。

## 采集生命周期

- Inspection 容器可以常驻，但卡片默认为 `idle`，不订阅数据。
- 点击“开始智能控制”并触发卡片 `start` 后，才创建 ROS2 订阅并开始生成新分片。
- `stop` 先停止新数据进入，再排空有界 writer 队列并 finalize 当前分片；已完成分片的补传不应被取消。
- 异常退出后启动时先恢复 `.part` 和 ledger。`auto_resume_after_reboot=false` 时保持 `idle` 并返回 `resume_required=true`，不会偷偷继续采集。

## 本地运行

```bash
cd inspection
python3 -m pip install -r requirements.txt
python3 main.py
```

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
- QoS：`BEST_EFFORT + KEEP_LAST(50) + VOLATILE`；
- 写入路径：`.wav.part` → fsync → WAV/JSON 原子 rename → SQLite `FINALIZED`；
- SQLite 使用 WAL 和 `synchronous=FULL`，正式文件存在但账本缺失时可扫描重建。

## 当前边界

- Audio Inspector 为 `runtime_mode=ros2-durable-audio`、`storage_ready=true`，已真实本地落盘；
- Video Inspector 仍为 `runtime_mode=gate1-contract-only`、`storage_ready=false`；
- `testupload` 返回 `unsupported`，直到 Gate 4 接入真实 COS backend；
- COS 凭证只能由部署 secret 提供，不进入卡片配置或日志。
