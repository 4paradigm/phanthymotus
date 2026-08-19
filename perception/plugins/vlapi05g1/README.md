# `vlapi05g1` Perception 卡片规格

> 状态：阶段四 Gate A/B/C、阶段五 PerceptionBundle/MCP/Core 接入和阶段六 Gate D/E/F 的 ROS2/Core/故障现场验证已于 2026-07-14 通过；用户于 2026-07-15 修复代理并实际进入 Dashboard，阶段六浏览器访问 blocker 已关闭。详见[阶段四验证记录](阶段四验证记录.md)、[阶段五验证记录](阶段五验证记录.md)和[阶段六验证记录](阶段六验证记录.md)。
>
> 安全边界：本卡片只生成 action proposal，不持有 execution token，不调用 G1 executor，也不向 `rt/arm_sdk` 发布命令。proposal 生成后是否仍可执行，由下游 actuator 根据机器人实时状态和控制权重新判断。

## 规格摘要

| 项目           | 必填内容                                                                                                                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| card id      | `vlapi05g1`                                                                                                                                                                                                       |
| card type    | `processor`；消费图像与机器人状态，输出动作提案                                                                                                                                                                                     |
| 感知目标         | 缓存最新 RGB 图像和 29 维 G1 关节状态，结合任务文本调用外部 π0.5 policy server，输出物理空间的 `[1,50,18]` action chunk                                                                                                                          |
| input topic  | `image_topic`：`sensor_msgs/msg/CompressedImage`、`image/jpeg`、`BEST_EFFORT + KEEP_LAST(1) + VOLATILE`；`state_topic`：本卡片约定的 `std_msgs/msg/String` JSON 适配 topic、`data/json`、`BEST_EFFORT + KEEP_LAST(1) + VOLATILE` |
| output topic | 默认 `{image_topic}/vlapi05g1`；`std_msgs/msg/String`；`data/json`，schema=`pi05.g1.action_chunk.v1`；`RELIABLE + KEEP_LAST(1) + VOLATILE`                                                                              |
| actions      | `info`、`config`、`start`、`stop`、`predict`；具体请求、返回与状态约束见下文                                                                                                                                                          |
| 配置           | shared：policy endpoint、超时和输入大小上限；instance：任务文本、seed、图像/状态新鲜度；非法值必须失败，禁止零 observation 和隐式运行路线降级                                                                                                                    |
| 模型           | `xiaopeng-wu/pi05_unitree_g1`，revision `580238372153c51ad564b894d8fea736ab4cdeb8`，Apache-2.0；LeRobot π0.5，4,143,404,816 参数                                                                                        |
| 部署           | 卡片运行在可同时访问 ROS2 数据面和 policy endpoint 的 Perception 宿主；policy 固定在 wlcb-23 外部 GPU runtime，第一阶段不在 G1 Jetson 上加载模型；北京 G1 通过[腾讯云 TLS Relay](../../../../../team-utils/北京G1到wlcb23推理网络.md)访问 policy                                                         |
| 测试           | Gate A/B/C/D/E/F 的自动化与现场 API/ROS 验证通过；已覆盖真实 ROS2 双输入单输出、多实例、Core 重启配置恢复、故障隔离、10 次资源回收和日志脱敏；用户已实际进入 Dashboard，未单独保留四项视觉检查截图                                                                                                          |

## 职责与非职责

本卡片负责：

1. 从两个正式 input topic 缓存每个实例最新的 JPEG 图像和 G1 joints 状态。
2. 校验 observation 完整性、新鲜度和维度。
3. 接受 `predict` 触发，将图像、29D state、task、seed 发送给外部 policy server。
4. 校验 policy 响应并发布 action proposal。
5. 暴露实例状态、最近一次推理耗时和错误原因。

本卡片不负责：

1. 不解释或执行 action，不直接连接 `rt/arm_sdk`。
2. 不持有 G1 executor token、`confirm` 字符串或执行步数权限。
3. 不屏蔽关节或 remote 维度；这些执行安全策略由独立 actuator 卡负责。
4. 不在输入缺失时生成零图像、零 state 或零 token 继续推理。
5. 不自动把每一帧图像送入 policy；只有显式 `predict` 才触发推理。
6. 不判断 action proposal 生成后机器人是否因其他运控、人工操作或外力发生状态变化，也不据此授权或拒绝执行；该门禁属于下游 actuator。

## 输入契约

### 图像输入 `image_topic`

- topic：由 `start.image_topic` 指定。
- ROS type：`sensor_msgs/msg/CompressedImage`。
- format：`image/jpeg`；`msg.data` 必须是可解码的完整 JPEG。
- 模型输入：RGB，policy server 解码后统一 resize 为 `640 x 480`。
- 大小上限：默认 8 MiB，由 shared 配置 `max_image_bytes` 控制。
- 溯源语义：保留 `header.frame_id`；时间优先使用非零 `header.stamp`，否则使用卡片接收消息时的墙钟时间，并标记时间来源；新鲜度始终按本机单调时钟计算。
- QoS：`BEST_EFFORT + KEEP_LAST(1) + VOLATILE`；只缓存最新帧，不反压相机、不积压旧帧。

### 状态输入 `state_topic`

- topic：由 `start.state_topic` 指定，是与 `image_topic` 并列的正式 input topic；上海 G1 当前已验证的原始数据源是 Agent Core WebSocket bus `/embodied_unitree_g1/state/joints`。
- ROS type：`std_msgs/msg/String`。上游 state provider 负责把机器人状态发布成该 ROS2 topic；卡片不得绕过 topic 在内部直接轮询 WebSocket、HTTP 或机器人 SDK。
- format：`data/json`，最小结构：

```json
{
  "joints": [
    {"idx": 0, "q": -0.0336},
    {"idx": 1, "q": -0.046}
  ],
  "imu_quat": [0.9989, 0.0076, 0.0030, 0.0456]
}
```

- 映射：必须包含 `idx=0..28` 的有限数值 `q`；按 index 升序构造 29 维 `observation.state`。额外 joints 和 `imu_quat` 不进入当前模型输入。
- 当前部署前置：需要把已验证的 WebSocket joints JSON 原样桥接为上述 ROS2 `String` topic；桥接实现不属于本卡片阶段二。
- 时间语义：当前 WebSocket 样本没有可用的源时间戳，按卡片收到状态消息时的单调时钟判断新鲜度。
- QoS：`BEST_EFFORT + KEEP_LAST(1) + VOLATILE`。

### Observation 组合规则

1. `start` 后只缓存输入，不自动推理。
2. `predict` 使用调用瞬间最新的图像与 state，并复制为本次请求的不可变快照。
3. 图像或 state 缺失、格式错误、包含 NaN/Inf，或年龄超过实例配置上限时，`predict` 失败且不得请求 policy。
4. 不做近似时间同步；输出必须带两路 input topic、接收时间、请求时年龄、图像 `header.frame_id` / `header.stamp` 和推理使用的 29D state 快照，供下游判断关联质量和 proposal 是否仍有效。
5. 同一实例最多一个在途推理；并发 `predict` 返回 `busy`，不排无界队列。
6. 新鲜度只在冻结推理输入快照时检查；推理期间或结果发布后机器人状态发生变化，不由本卡片判定 proposal 失效。

## Policy 请求契约

默认请求 `POST {policy_url}`，body：

```json
{
  "image_base64": "<JPEG bytes as base64>",
  "state": ["29 finite float values"],
  "task": "move blue box back and forth between tables",
  "seed": 0
}
```

约束：

- `image_base64`、`state` 和 `task` 在 production 都是必填项。
- `state` 固定 29 维。
- `task` 必须是去除首尾空白后非空的字符串；当前只验证过训练任务原文，其他任务不得标记为已验证能力。
- `seed` 范围 `0..2147483647`，默认 0。
- 不从卡片传 `task_tokens` 或 `attention_mask`；tokenizer 由 production policy runtime 管理。
- HTTP 请求显式绕过环境代理，避免 loopback/tunnel endpoint 被代理转发。

## 输出契约

默认输出 topic 为 `{image_topic}/vlapi05g1`。每次成功 `predict` 只发布一条 `std_msgs/msg/String` JSON：

```json
{
  "schema": "pi05.g1.action_chunk.v1",
  "request_id": "vlapi05g1-main-000001",
  "created_at": 0.0,
  "observation": {
    "image_topic": "/robot/camera/image/compressed",
    "image_frame_id": "g1_front_camera",
    "image_stamp": 0.0,
    "image_stamp_source": "header_or_receive_time",
    "image_received_at": 0.0,
    "image_age_at_request_s": 0.0,
    "state_topic": "/robot/state/joints",
    "state_received_at": 0.0,
    "state_age_at_request_s": 0.0,
    "state": ["29 finite float values"]
  },
  "task": "move blue box back and forth between tables",
  "action_space": "physical_quantile_unnormalized",
  "frequency_hz": 30,
  "action_shape": [1, 50, 18],
  "action_chunk": [["18 finite float values"]],
  "fresh_inference_per_request": true,
  "num_inference_steps": 10,
  "seed": 0,
  "policy_infer_seconds": 0.0,
  "card_elapsed_seconds": 0.0,
  "execution_authorized": false
}
```

发布前必须校验：

- `ok=true`；
- `action_shape == [1,50,18]`；
- `action_chunk` 为 50 行、每行 18 个有限数值；
- `action_space == "physical_quantile_unnormalized"`；
- `fresh_inference_per_request == true`；
- `num_inference_steps == 10`；
- 返回 seed 与请求 seed 一致。

任一项不满足都视为 policy contract error，不发布 action proposal。卡片不在此阶段做 URDF 限位、18D→10 关节映射或执行授权。

action proposal 表达的是“基于 `observation.state` 快照得到的候选动作”，不是可直接执行命令。下游 actuator 必须在执行前重新读取机器人实时状态，并自行检查 proposal 年龄、控制权、状态漂移和动作安全；即使推理期间机器人被其他运控改变，本卡片仍只发布带原始快照的 proposal，不代替下游作有效性判断。

## Actions

### `info`

- 请求：`action`；可选 `instance_id`。不传 `instance_id` 时返回全部实例摘要。
- 允许状态：任何状态。
- 行为：每次调用同步执行一次只读 `GET health_url`，超时取 `min(request_timeout_s, 5.0)`；不得创建 ROS node、启动线程或触发 policy 推理。
- 返回：卡片版本、实例状态、实际 topics、当前配置、缓存是否就绪、两路输入年龄、是否有在途请求、最近一次 request id/耗时/错误，以及本次只读 `/health` 结果。
- `/health` 失败不使 `info` 整体失败；`last_health.status=error` 并包含 `error_code`、`message`、检查时间和耗时。

### `config`

- 请求：`action` 加待更新字段；instance 字段要求 `instance_id`。
- shared 配置在已有实例运行时拒绝真实修改并返回 `restart_required=true`；Core 重放与当前值完全相同的已保存配置时作为幂等 no-op 接受，不静默重启。
- instance 配置可立即更新，只影响下一次 `predict`，不修改已在途请求快照。
- 非法字段、范围错误和空 task 必须失败，不能截断或回退默认值。
- 返回：`status=configured`、实际生效字段、`applied` 和 `restart_required`。

### `start`

- 必填：`instance_id`，以及显式的 `image_topic` + `state_topic` 或 Dashboard 画布形态的 `input_topics=[image_topic, state_topic]`；两种形态不得混用。可选 `output_topic`。
- `input_topics` 的顺序严格等于卡片 `topic_in` 端口顺序：第一路图像，第二路机器人 state；这是 Agent Core Dashboard 对多输入 processor 的通用启动参数。
- 行为：校验 shared 配置，创建两个订阅、一个发布器和单工作线程；不在 `start` 时触发推理。
- 相同参数重复调用保持幂等。
- 同一 `instance_id` 使用不同 topic 重复调用时拒绝，要求先 `stop`。
- 成功返回：`state=running`、三个实际 topic、缓存就绪状态。

### `stop`

- 必填：`instance_id`；不支持用空 `instance_id` 一次停止全部实例。
- 行为：停止接受新请求，等待在途 HTTP 请求至超时上限，释放订阅、发布器、worker 和缓存。
- 重复调用或实例不存在时返回稳定的 `state=idle`。
- 不发送任何释放机器人控制权的命令，因为本卡片从未取得控制权。

### `predict`

- 必填：`instance_id`；可选 `task` 和 `seed` 仅覆盖本次请求。
- 前置状态：实例必须为 `running`，图像和 state 缓存必须完整且新鲜，没有其他在途推理。
- 行为：冻结两个 input topic 的 observation 快照，调用一次外部 policy，校验响应并发布包含该快照来源的 action proposal；不在响应返回后用最新 state 作二次执行有效性判断。
- 成功返回：`status=published`、`request_id`、`output_topic`、`policy_infer_seconds`、`card_elapsed_seconds`。
- 失败返回：结构化 `status=error`、`error_code`、`message`；失败时不发布 action proposal。
- 超时后可由调用方安全重试；卡片自身不自动重试，避免迟到响应形成陈旧动作。

## 配置

### Shared 配置

| 字段 | 类型 | 默认值 | 范围与更新行为 |
| --- | --- | --- | --- |
| `policy_url` | string | `http://127.0.0.1:18080/predict` | 仅允许 `http://` 或 `https://`；运行中修改需先 stop 全部实例 |
| `health_url` | string | 从 `policy_url` 同源推导 `/health` | 可显式覆盖；运行中修改需重启实例 |
| `request_timeout_s` | number | `120.0` | `0.1..300.0`；运行中修改需重启实例 |
| `max_image_bytes` | integer | `8388608` | `1024..16777216`；运行中修改需重启实例 |

### Instance 配置

| 字段 | 类型 | 默认值 | 范围与更新行为 |
| --- | --- | --- | --- |
| `task` | string | `move blue box back and forth between tables` | 非空；立即生效于下一次 `predict`；其他任务尚未验证 |
| `seed` | integer | `0` | `0..2147483647`；立即生效于下一次 `predict` |
| `max_image_age_s` | number | `1.0` | `0.05..10.0`；按卡片接收时的单调时钟判断 |
| `max_state_age_s` | number | `0.5` | `0.05..10.0`；按状态接收时间判断 |

禁止配置：`allow_zero_observation`、execution token、执行确认字符串、执行步数、关节限位和 remote 映射。这些字段不属于本卡片权限面。

## 模型与运行路线

| 项 | 当前 production 契约 |
| --- | --- |
| model | `xiaopeng-wu/pi05_unitree_g1` |
| revision | `580238372153c51ad564b894d8fea736ab4cdeb8` |
| license | Apache-2.0 |
| dataset | `nepyope/unitree_box_move_blue_full`，550 episodes、475,206 frames、30 Hz |
| parameters | 4,143,404,816 |
| main weight | `model.safetensors`，9,354,050,752 B |
| input | RGB `3 x 480 x 640`、29D G1 state、task text |
| output | 50 steps x 18D，物理反归一化绝对关节目标；每步 30 Hz |
| tokenizer | openpi `paligemma_tokenizer.model`，由 policy runtime 持有 |
| runtime image | `pi05-g1:runtime-v3`，image id `sha256:cf89490a504d49649e663fbd1e7c3d400121cfd7fd0999add6843d80a3dc13dd` |
| policy location | wlcb-23，A100-SXM4-80GB，当前 GPU 6 / port 18080 |
| dtype / denoising | float32 / 10 steps |
| local fallback | 无；policy 不可达时明确失败，不降级到 Jetson 或 mock |

已验证单次 production shadow 的 policy 推理约 `0.391..0.675 s`，adapter 总耗时约 `0.738..1.195 s`；CUDA 峰值显存记录为 `16,736,264,192 B`。阶段四 Gate C 已用三次正式样本重测 policy P95 `0.3875 s`、SSH tunnel 端到端 P95 `2.9224 s`；样本数仍不足以上升为 production SLO。

## 故障与状态

实例状态：`idle -> running`；`predict` 期间通过 `in_flight=true` 表示忙，不额外引入不可恢复的 `loading` 状态。最近错误记录在实例状态中，但单次推理失败不退出 MCP server、不销毁实例。

| error_code | 条件 | 行为 |
| --- | --- | --- |
| `not_running` | 实例未启动 | 拒绝 predict |
| `invalid_request` | 单次 task 为空或 seed 不是范围内整数 | 拒绝 predict，不请求 policy |
| `observation_missing` | 缺图像或 state | 拒绝 predict，不请求 policy |
| `observation_stale` | 冻结请求快照时任一路输入超过新鲜度上限 | 拒绝 predict，不请求 policy |
| `invalid_image` | 非 JPEG、截断或超过大小限制 | 丢弃该帧，保留服务 |
| `invalid_state` | JSON 错误、缺少 0..28、非有限数值 | 丢弃该状态帧，保留服务 |
| `busy` | 同实例已有在途推理 | 立即失败，不排队 |
| `policy_unreachable` | 连接失败或超时 | 不自动重试，不发布 proposal |
| `policy_rejected` | HTTP 非 2xx 或 `ok != true` | 记录远端错误，不发布 proposal |
| `policy_contract_error` | shape、action space、steps、seed 或数值不合约 | 拒绝结果，不发布 proposal |

日志不得打印完整 base64 图像、完整 action chunk、token 或整帧隐私数据；只记录 request id、长度、shape、耗时和错误摘要。

## 测试验收矩阵

### 纯函数与契约

- 29D joints 映射：正常、缺 index、重复 index、NaN/Inf、额外 joints。
- JPEG：有效样本、非 JPEG、截断、超过大小上限。
- policy payload：字段、维度、task、seed 和 base64 解码一致。
- policy response：正确 `[1,50,18]`、错误 shape、空 chunk、NaN/Inf、错误 action space、错误 seed。
- action proposal：schema 完整、JSON 可序列化、`execution_authorized` 恒为 false。
- proposal provenance：完整保留 image/state topic、时间元数据和推理所用 29D state 快照。

### 生命周期与并发

- `info` 在 idle/running 都不创建 ROS node 或触发推理，只执行一次有界的只读 policy health 检查。
- 非法/合法 `config`，shared 运行中修改门禁，instance 热更新。
- 相同参数连续 `start` 两次不创建重复 node；不同 topic 明确失败。
- 连续 `stop` 两次均为 idle；start/stop 十轮无线程、订阅和缓存残留。
- 两实例 topic、缓存、request id、配置与输出互不串扰。
- 同实例并发 predict：一个执行，其他稳定返回 busy。

### 录制数据与性能

- 固定使用 `vla-g1-deployment/artifacts/shadow/g1-fixed-frame.jpg`、29D state artifact 和训练任务原文。
- 同一 revision、同一 observation、seed 0 连续至少三次，action chunk 在严格相等或预先定义的数值容差内一致。
- 记录端到端 P50/P95、policy P50/P95、吞吐、CPU、RSS 和 policy CUDA 峰值显存。
- 阶段四实测基线：policy P95 `0.3875 s`、SSH tunnel 端到端 P95 `2.9224 s`；仅有三次正式计量样本，不把该数值上升为 production SLO。硬门禁仍是单实例一个在途请求、不积压、严格确定性和完整响应契约。

### ROS2 与外部故障

- 向两个真实 input topic 发布录制图像和 state，`predict` 后只出现一条匹配 request id 的 output。
- 缺一路输入、输入过期、输入断流时不得发布陈旧 proposal。
- 推理期间发布更新 state 时，proposal 仍保留请求开始时的 state 快照；卡片不自行判断其执行有效性。
- policy 连接拒绝、超时、HTTP 400、错误 JSON 和 contract mismatch 均不拖死 ROS executor 或 MCP server。
- policy 恢复后下一次显式 predict 可成功，不依赖重启卡片。

### PerceptionBundle / MCP / Agent Core

- Bundle 显式 loader 与运行配置能加载 `vlapi05g1`，进程日志出现 `VLAPi05G1Plugin loaded`。
- Core display name 和 MCP `serverInfo.name` 由宿主配置显式提供，缺失、同名或继续使用默认名时拒绝启动。
- `initialize`、`tools/list`、`info`、非法/合法 `config`、重复 `start/stop` 均通过 MCP HTTP 验证。
- 真实 Agent Core 在未保存 tool config 时拒绝业务 action；通过 tool-config API 保存 shared 配置后，Core 代理 `start/info/predict/stop` 成功。
- 阶段五的 Core `predict` 预期返回 `observation_missing`：这证明业务 action 已路由到卡片且缺输入时 fail closed，不代表 ROS2 数据闭环已完成。

### 当前未验证边界

- WebSocket joints JSON 到正式 ROS2 state input topic 的上游 state provider 尚未实现和验证。
- 录制数据已通过真实 ROS2 双输入/单输出闭环、断流/陈旧输入和多实例隔离；本结论不代表上述 live state provider 已存在。
- Agent Core 唯一注册、tool-config 保存/懒恢复、Dashboard `input_topics` 启动参数、输出 topic 推导和多实例已通过 API/ROS 现场验证；用户已实际进入 Dashboard，但未单独保留侧栏、配置表单和多实例拖拽四项截图。
- wlcb-23 Gate F 当时的画布没有 `vlapi05g1` 卡片，因此未触发可选的 AgentLoop 执行线绑定项；现有录制 observation 的真实 policy `predict` 已经 Core 代理调用并通过 Gate D。
- 北京 G1 已完成独立 overlay 构建、policy health 和 Core 注册，但尚未用北京 G1 的 live RGB + joints state 完成一次 `predict`。
- 本卡片不会证明 G1 实机动作安全；物理执行必须由独立 actuator 卡和现场 owner 验证。

## 资料依据

- [pi0.5 上海 G1 实机连续控制部署](../../../../vla-g1-deployment/details/pi05实机连续控制部署.md)
- [VLA 动作到 G1 实机指令](../../../../vla-g1-deployment/details/VLA动作到G1实机指令.md)
- [pi0.5 模型选择与 HF 验证](../../../../vla-g1-deployment/details/pi05模型选择与HF验证.md)
- [训练数据 Action 契约](../../../../vla-g1-deployment/artifacts/dataset-contract/action-contract.md)
- [production policy server](../../../../vla-g1-deployment/scripts/pi05_policy_server.py)
- [真实 10-step shadow artifact](../../../../vla-g1-deployment/artifacts/shadow/g1-real-runtime-v2-step10.json)
