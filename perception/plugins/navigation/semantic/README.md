# Navigation 内部视觉语言导航模块

该模块属于统一 `controlled_semantic_spatial` 卡片，不再注册独立 `vln` 卡片。

- `capture`：等待新的 RGB/odom 对，调用配置的 VLM 生成描述，并记录当前
  `map -> base_link` 位姿。
- `navigate(query)`：只在当前 FAST-LIVO2 map session 内匹配已记录地点。
  命中后直接调用同卡片 Nav2 planner，并透传 Agent Core 的
  `_control_nav_id`；未命中时不产生目标。

## VLM 配置

统一卡片配置字段为 `vlm_base_url`、`vlm_api_key`、`vlm_model` 和
`vlm_timeout_sec`。API key 无源码或 YAML 默认值，不出现在 action 回执、状态
或日志中。`capture` 会把相机 JPEG 发送到所配置的外部 VLM 服务。

## 内部数据边

| 数据 | topic | 来源 |
| --- | --- | --- |
| RGB | `/ubuntu/camera/rgb` | Canvas 外部输入 |
| odom | `/ubuntu/navigation/odom` | 同卡片 FAST-LIVO2 模块 |
| map status | `/ubuntu/navigation/fast_livo2/status` | 同卡片 FAST-LIVO2 模块 |

`goal_pose` topic 仅保留为兼容的可选外部控制入口。语义 `navigate` 的正常路径
不再 publish 后等待另一个卡片订阅，而是使用进程内 planner handler，以免
Canvas 自连线缺失导致目标丢失。

## 限制

- FAST-LIVO2 坐标是 session-local；bridge 重启、心跳 stale、非 mapping 状态
  或 session 变化都会阻止复用旧坐标。
- 默认按 ROS 接收时间配对传感器；只有确认上游共享 header 时钟域后才使用
  `source_timestamp`。
- VLM 匹配成功不等于物理执行授权；proposal 仍必须经过 Agent Core lease 和
  Driver 安全检查。
