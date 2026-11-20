# autonomous_task

这张 ActuCore 卡片接收自然语言桌面操作任务。每轮从已连接的相机取一组新图像，读取末端状态，将这些观测和历史动作回执交给多模态模型。模型可以选择一只手执行一个动作，要求再看一眼，或给出任务结果。动作执行后，卡片根据最新实测状态和图像开始下一轮推理。

## 接线与配置

| 方向 | 端口 | 格式 | 接线 |
|---|---|---|---|
| 输入 1 | 头部相机 | `image/jpeg` | 可选，用于观察桌面和全局位置 |
| 输入 2 | 左腕相机 | `image/jpeg` | 可选，用于观察左爪附近 |
| 输入 3 | 右腕相机 | `image/jpeg` | 可选，用于观察右爪附近 |
| 输入 4 | 末端状态 | `state/eef` | 连接上游 `eef_state` 的实测状态输出 |
| 输出 1 | 末端控制 | `control/eef` | 连接下游 `servo_eef` 的绝对目标输入 |

执行视觉任务至少接一路相机。相机数与启用哪只手可以独立选择。卡片只在决策和复核时取图，每次等已接入的各路相机提供新 JPEG。模型需要局部细节时可以请求原图裁剪。下一轮会同时看到注明来源的历史裁剪和新拍的全景。

TRON2 左臂实验接入 `camera`、`camera_left` 和 `/limx_tron2/state/eef`，输出连到 `/limx_tron2/control/eef`。驱动的 `control_interface` 给出消息版本、参考系、左右 TCP 名称、各组起点和长度，以及开口范围。如需显式指定接线，可使用 `head_topic`、`left_wrist_topic`、`right_wrist_topic`、`state_topic` 和 `control_topic`。

在卡片配置中填写 `base_url`、`key` 和 `model_name`。模型服务须提供支持图像和函数调用的 Responses API。`arm_mode` 默认为 `left`，也可选 `right` 或 `both`。可用 `frame_description` 说明当前安装中已知的坐标轴方向、夹爪和视角约定，模型每一轮都能读到。需要留存实验时，可在部署 YAML 中设置 `record_dir`。卡片按任务保存模型看到的 JPEG、动作与观测记录、最终状态。

## MCP 调用

启动画布后，通过 MCP 工具 `autonomous_task` 的 `submit` 提交任务。

```json
{"action":"submit","instruction":"把胶带抓起来，向左移动约 5 厘米，再放下","arm_mode":"left","request_id":"tape-transfer-01"}
```

提交立即返回 `action_id`。用 `{"action":"status","action_id":"…","include_history":true}` 获取执行阶段、模型动作、实测回执和终态原因。用 `{"action":"cancel","action_id":"…"}` 取消后续决策。已发布目标的执行由下游控制。卡片一次接受一个任务，保留最近 32 个任务。重复提交相同 `request_id`、指令与模式时返回已有任务。启用归档后，重启可按原 `action_id` 查询已保存的终态和历史。上层也可以按 ActuCore 的 ACP 契约接收完成事件。

## 末端工具与实测反馈

每只启用的手向模型提供以下工具，工具名带 `left_` 或 `right_` 前缀。双臂模式每轮仍只选一个工具。

| 工具 | 参数 | 目标 |
|---|---|---|
| `move_to` | `position_m`、`quaternion_xyzw` | TCP 在声明的参考系中的绝对位置（米）和姿态（xyzw 单位四元数） |
| `move_by` | `translation_m`、`rotation_vector_rad` | 从本轮实测 TCP 出发，沿参考系轴平移，以 TCP 自身为中心按参考系旋转向量转动 |
| `set_gripper` | `opening_mm` | 夹爪的名义开口（毫米），工具可用范围由下游声明 |

相对目标按 `p' = p + Δp`、`R' = Exp(Δr) · R` 合成。模型推理期间手的位置发生明显变化时，卡片重新观测。卡片将一次相对决策限制在 10 厘米和 30 度内，实际限速、逆解和硬件下发由执行卡处理。工具使用观测中的 `eef_frame` 和 `reference_label`。例如 TRON2 的左右 TCP 分别是 `L_ee_link` 和 `R_ee_link`，两者都在 `base_Link` 中表达。当前反馈可以直接作为绝对目标。迁移旧版 TRON2 记录的 grasper 绝对位姿时，需要做一次参考点换算，`p_tcp = p_grasper + R_grasper × [-0.0021, 0, -0.1313] m`，四元数相同。

对于 TRON2，`state/eef` 和 `control/eef` 两端都是 ROS 2 的 `std_msgs/String`，内容为带 `schema`、`frame`、`stamp_ms`、`values` 的 JSON。`tron2.eef/1` 的 16 项依次为左 TCP 的 xyz 和 xyzw、左开口、右 TCP 的 xyz 和 xyzw、右开口。线上位置和开口都使用米。每条指令生成递增的 Unix 毫秒时间戳。单臂操作也发送完整目标，保持静止的分组取自最新实测值。

卡片发令后等待时间戳推进的连续状态帧。历史记录中可查看实际 TCP 或开口、是否观察到移动以及是否达到名义目标。夹爪接触物体后可能在目标开口之外稳定，模型会结合下一轮视觉决定是否试提。指令已发布、持续有反馈而手没有移动时，回执会记录 `no_motion_observed`。缺测超时会返回具体原因。模型请求临时超时会重新观察再推理，连续三次推理无响应则结束任务。模型仅返回无法解析的文字时，卡片保留可见回复、重新取图，并在三次无有效决策后结束任务。

模型提出完成后，卡片等待 3 秒再取图，同一模型在无动作工具的情况下复核。复核可查看初始和最新图像，以及夹爪收紧后首次平移、搬运最远处和张开时的历史帧。对于搬运任务，模型须从图像确认物体曾随夹爪离开支撑面，并在目标位置独立放稳。证据不足时，模型可以继续观察、纠正或给出具体失败原因。
