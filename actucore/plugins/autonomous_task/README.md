# autonomous_task

这张 ActuCore 卡片接收自然语言桌面操作任务。每轮从已连接的相机获取一组新 RGB，读取位姿、夹爪状态与历史回执，请求多模态模型选择一个动作。模型宣告完成后，卡片等待 3 秒，重新采集图像，并让同一模型在没有动作工具的情况下复核。复核可以确认完成、说明失败原因，或要求继续观察与纠正。

## 接线与配置

三个 `image/jpeg` 输入口分别为头部、左腕、右腕。任一路都可省略，执行视觉任务至少要接一路。卡片只在决策和复核时取图。已接入的每一路都必须在本轮提供新的 JPEG。TRON2 的 `camera`、`camera_left`、`camera_right` 可直接连接。其他安装方式可用上游声明的相机 ID 或配置项 `head_topic`、`left_wrist_topic`、`right_wrist_topic` 将已连接的话题映射到物理位置。

在卡片配置中填写 `base_url`、`key`、`model_name`。服务需提供支持图像与函数调用的 Responses API。`arm_mode` 默认为 `left`，也可选 `right` 或 `both`。设置通过画布保存，或在部署环境的 ActuCore YAML 中提供。仅在需要留存实验图像时设置卡片 YAML 的 `record_dir`，卡片按任务分别写入模型实际看到的 JPEG、逐步反馈和终态。

## 调用

启动画布后，上层通过 MCP 工具 `autonomous_task` 提交任务。下例是该工具的 `arguments`。

```json
{"action":"submit","instruction":"把胶带抓起来，向左移动约 5 厘米，再放下","arm_mode":"left","request_id":"tape-transfer-01"}
```

提交返回 `action_id`。用 `{"action":"status","action_id":"…","include_history":true}` 获取执行阶段、模型动作、驱动回执和最终原因。用 `{"action":"cancel","action_id":"…"}` 停止后续决策。已经下发的动作仍由驱动跟踪至终态。一次只接受一个任务。在同一进程保留的最近 32 个任务内，相同 `request_id`、指令和模式可重复查询。上层也可以按 ActuCore 的 ACP 契约接收完成事件。

## 实机执行

每只启用的手臂暴露 `move_to`、`move_by` 和 `set_gripper`。动作串行执行，每步之后重新观察，包括双臂模式。TRON2 使用基座系，位置单位为米，绝对姿态用 xyzw 单位四元数，相对旋转用基座系旋转向量（弧度），夹爪用名义开口毫米。沿用 v1 的 `grasper_L_Link` 或 `grasper_R_Link` 原点作为末端参考。厂商 TCP 和指尖的坐标需要各自的几何变换。

当前执行适配器位于 `tron2.py`。适配器从 Motus 驱动获取 URDF，并用反馈关节位置求解末端目标的 IK，再通过驱动的 MoveJ 和夹爪接口执行。模型只能指定末端位姿与开口。单步平移最多 10 厘米、旋转最多 30 度，目标经关节限位和单步幅度检查。MoveJ 执行关节插值，末端路径可能弯曲。驱动未确认或失去动作归属时，卡片结束任务并返回原因。

两个 `data/json` 输出话题为 `/actucore/autonomous_task/arm` 和 `/actucore/autonomous_task/gripper`，使用 `std_msgs/String`。每个已确认下发的动作发布一次，包含 `action_id`、`step`、`tool`、`arguments` 和实测 `feedback`。当前由内部适配器负责实际执行；未来接入下游执行卡片时，可在该边界替换适配器。
