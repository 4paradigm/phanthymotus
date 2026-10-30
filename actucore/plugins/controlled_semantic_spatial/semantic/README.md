# controlled_semantic_spatial 内部视觉语言导航模块

该模块属于统一 `controlled_semantic_spatial` 卡片，不再注册独立 `vln` 卡片。

- `capture`：等待新的 RGB/odom 对，调用配置的 VLM 生成描述，并记录当前
  `map -> base_link` 位姿。
- `navigate(query)`：只在当前 FAST-LIVO2 map session 内匹配已记录地点。
  命中后直接调用同卡片 Nav2 planner，由 planner 为该任务
  生成新 `nav_id`；未命中时不产生目标。

## VLM 配置

统一卡片配置字段为 `vlm_base_url`、`vlm_api_key`、`vlm_model` 和
`vlm_timeout_sec`。这些字段对建图和坐标导航均为可选；未配置 API key
时卡片仍可启动，但 `capture` 和语义 `navigate` 会明确返回
`vlm_not_configured`。API key 只保存在进程内，不出现在 action 回执、状态
或日志中。配置支持单字段增量更新；未提交或以 `****` 占位的 API key 会保留
当前进程内密钥。`capture` 会把相机 JPEG 发送到所配置的外部 VLM 服务。

## 内部数据边

| 数据 | topic | 来源 |
| --- | --- | --- |
| RGB | `/ubuntu/camera/rgb` | Canvas 唯一 RGB 外部输入；标准 `sensor_msgs/msg/CompressedImage` JPEG |
| odom | `/ubuntu/navigation/odom` | 同卡片 FAST-LIVO2 模块 |
| map status | `/ubuntu/navigation/fast_livo2/status` | 同卡片 FAST-LIVO2 模块 |

语义 `navigate` 使用进程内 planner handler；不再提供 `goal_pose` 外部 topic
入口，也不再向其发布。未接入 planner 时明确返回 `planner_unavailable`。

## 限制

- `capture` 只接受正在建图，或已加载地图且对齐确认的 `relocalized`
  状态。FAST-LIVO2 每次启动建图或定位运行时都会发布稳定的
  `map_session_id`；短暂心跳 stale、传感器未就绪只会暂停使用，恢复后不会
  误清空同一会话的坐标。运行时重启、换图或 bridge 重启仍会阻止复用旧坐标。
- 新鲜度和图像/位姿近似配对仅使用本地单调接收时间，不说明采集时间同步；
  不依赖相机 header、PSE1 时间、标定或外参。旧 `sensor_sync_mode=source_timestamp`
  配置不再受支持，须移除或改为 `receive_time`。JPEG 格式和标记无效时拒绝该帧；
  标记检查并不保证 JPEG 可完整解码。
- VLM 匹配成功不等于物理执行；标准速度输出的接收、TTL、急停、障碍和
  物理停车等安全行为须由下游独立验证。标准速度消息不携带导航任务 ID。
