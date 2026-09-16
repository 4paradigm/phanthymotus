# VLA 接入方案

状态：设计草案，未实现。撰写日期 2026-09-16。

本文给出把 VLA（Vision-Language-Action）策略接入 PhanthyMotus 的设计：**通用控制接口**、卡片底座、本地/云双 provider、云端接口契约、以及本地模型选型。

其中通用控制接口（§2）是前提。驱动已经吸收了各家 SDK 的差异，但这份适配目前只有 agent-core 用得上——因为它的输入天然统一（文本 + MCP）。要让 actucore 里的 VLA 和其它执行卡片也"接上"同一份适配，需要在数据面补一层格式与协商约定。没有它，后面所有内容都无处落地。读之前请先读 `actucore/README.md`（卡片契约）和 `perception/README.md` 的「Plugin Concurrency」（并发规范）。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| VLA 放哪 | `actucore/`，这一层就是为它建的，目前 `plugins/` 为空 |
| VLM 放哪 | 产出语义的放 `perception/`（同 `vop.py`）；产出动作的算 VLA |
| 一张卡片能否通吃各家模型 | 能，但必须拆成 **卡片（不变） + provider（每家一个 adapter）** |
| 缺的是什么 | 一层**通用控制接口 `motus.control/1`**：驱动声明能力、上游按声明发指令、驱动侧仲裁。见 §2 |
| VLA 驱动谁 | **画布上连上了什么就驱动什么**，和 agent-core 只用连上的 MCP 同构。不是配置里的字符串 |
| 端上能不能跑 π0 / UnifoLM | **不能**。实测 Orin NX 8GB 总内存 7 GB、跑着 perception 时仅余 ~3 GB；π0 类 3.3B 模型 bf16 光权重就 6.6 GB |
| 本地能跑什么 | SmolVLA-450M 级别。需实测，公开资料没有 Orin NX 的数字 |
| 云/边缘怎么放 | **局域网边缘服务器**是主路径；公网云只用于开发、评测、数据回流 |
| 最大的架构缺口 | ActuCore → Driver 的**动作通路目前是断的**，且没有指令仲裁和 watchdog |

---

## 1. 现状与缺口

### 1.1 已经具备的

- `actucore/` 骨架：MCP HTTP 15730、容器 `embodied-actucore`、`/opt/embodied/models` 挂载、注册与探活全通
- 卡片契约（duck typing，无基类）：`PREFIX` / `get_tools()` / `dispatch()`，`action.enum` 必须含 `info`
- ACP：`x-completion` 完成回调、`x-resource` 物理通道互斥（`mcp_client.py` 的 `parse_resources` / `resources_conflict`）
- 打断：`x-hooks` 的 `on_interrupt_*`（`hooks.py`，`event/llm.py` 里 `hooks.fire('on_interrupt_all')`）
- 授权面：画布连线即授权（`canvas_binding.py`）
- MCP 工具可返回 image content，`mcp_client.py` 会转成 `image_url` 交给主 LLM（`_trim` 只保留最近 5 张）

### 1.2 必须补的（按依赖顺序）

| # | 缺口 | 说明 |
|---|---|---|
| 1 | **ActuCore → Driver 的动作通路** | `control/velocity` / `control/joint` 只出现在 README 和 `mcp_manage.py` 的格式推断表里，**没有任何驱动订阅它们**。驱动里唯一以 topic 为输入的是 speaker（`topic_in: audio/pcm-16k`） |
| 2 | **通用控制接口 `motus.control/1`** | 目前只有 format 字符串，没有消息字段、单位、关节顺序，也没有"驱动声明自己接受什么"的描述面。这是核心缺口，§2 |
| 3 | **指令仲裁** | VLA 闭环在 ROS 上持续发布时，LLM 仍可直接调 `loco` / `arm`。`x-resource` 只在 agent-core 的 dispatch 路径生效，自跑的 ROS 节点不经过 barrier |
| 4 | **Watchdog / e-stop** | 远端推理断连、卡片崩溃、进程被 OOM kill 时谁停机器人 |
| 4b | **descriptor 反向传递** | `_resolve_input_topics` 目前只把上游 topic 交给下游，没有把下游能力带回上游 |
| 5 | **ACP 进度通道** | 只有 `/api/acp/complete`，没有中途进度。长任务期间 LLM 完全不知道状态 |
| 6 | **观测时间同步** | 现有 QoS 是 BEST_EFFORT / depth 2，没有多路相机 + 本体状态的对齐机制 |
| 6b | **碰撞复检** | 没有 planning scene，chunk 无法对照场景校验。第一版补不上，见 §2.3 第 6 条 |
| 7 | **`control/*` 渲染器** | `web/js/renderers/` 里没有，动作会落到默认 KV 面板 |
| 8 | **数据录制** | 观测-动作对无处存储，数据闭环无从谈起 |

**缺口 1–4 是安全相关的，必须在第一个 VLA 卡片上线前完成。** 5–8 可以后补。

---

## 2. 通用控制接口 `motus.control/1`

这是当前架构里真正缺的一层，比 VLA 卡片本身更基础。

### 2.0 为什么需要它

驱动已经完成了"本地控制适配"——每家机器人的 SDK 差异被 `phanthymotus-driver/<vendor>/` 吸收掉了。但这份适配目前**只对 agent-core 可用**，因为 agent-core 的输入是统一的（文本 + MCP `tools/call`）。

对 actucore 的卡片不成立：VLA 每秒产出几十帧连续指令，走 `tools/call` 既不现实也不该走。它需要的是一条**数据面**通路，而这条通路上流的东西目前没有任何约定——`control/velocity` / `control/joint` 只是 `mcp_manage.py` 格式推断表里的两个字符串，没有字段、没有单位、没有关节顺序，也没有任何驱动订阅它们。

所以补的不是"第三种传输"，而是**已有传输上的一层格式与协商约定**：

| 面 | 传输 | 用途 | 频率 |
|---|---|---|---|
| 控制面 | MCP `tools/call` | 启停、配置、查询、能力协商 | 低频，要应答，要授权 |
| **数据面** | **DDS topic（`control/*`）** | **连续指令流** | **高频，一对多，Dashboard 可观测** |

这和 speaker 的现有模式完全一致：播放数据走 topic（`topic_in: audio/pcm-16k`），启停走 tool。

注意 DDS 已被锁在 loopback（`dds-local.xml`），所以控制流**不会离开本机**——这对安全是好事，同时意味着边缘服务器上的 VLA 必须经由 actucore 卡片进来（HTTPS），不能从机器外直接往 DDS 上写。

### 2.1 描述面：驱动声明自己接受什么

驱动的命令卡片在 `info()` 里返回一份 **action space descriptor**。这是动作接口的**唯一权威来源**：

```jsonc
{
  "control_interface": "motus.control/1",
  "mode": "joint_position",            // joint_position | joint_velocity | joint_torque
                                       // | eef_pose | twist
  "dof": 14,
  "joint_names": ["left_shoulder_pitch", "..."],   // 顺序即数组顺序，权威
  "units": {"angle": "rad", "linear": "m", "time": "s"},
  "limits": {
    "lower": [...], "upper": [...],
    "max_velocity": [...],
    "max_delta_per_step": [...]        // 单步限幅，驱动侧强制执行
  },
  "frame": "base_link",                // eef_pose / twist 的参考系
  "end_effector": {"type": "gripper_2f", "range": [0.0, 0.09], "units": "m"},
  "rate": {"max_hz": 100, "expected_hz": 30, "watchdog_ms": 200},
  "urdf_ref": "mcp__<id>__model"       // 可选补充，见下
}
```

**URDF 不是这份 descriptor 的来源，只是可选补充。** 它对碰撞检查和 FK 有用，但它没有单位约定、没有控制频率、不说明驱动接受绝对角还是增量、也没有归一化范围。把它当动作接口的权威来源是错的。

### 2.2 数据面：流在 topic 上的消息

```jsonc
{
  "schema": "motus.control/1",
  "seq": 1024,
  "stamp_ms": 1789234567123,           // 指令生成时刻
  "obs_stamp_ms": 1789234566900,       // 这条指令基于哪一帧观测
  "ttl_ms": 100,                       // 超期即作废，驱动丢弃
  "source": "mcp__actucore__vla",      // 谁发的 —— 仲裁与审计的依据
  "priority": 50,
  "mode": "joint_position",            // 与 descriptor 对账，不符即拒绝
  "dof": 14,
  "values": [...],                     // 工程单位，顺序同 descriptor.joint_names
  "gripper": 0.04,
  "chunk": {"index": 3, "size": 50}    // 可选，便于调试与 RTC 归因
}
```

几个字段的理由，都不是可选的：

- **`obs_stamp_ms` + `ttl_ms`** —— 远端推理天然会产出陈旧指令，这是"陈旧动作不许执行"的唯一可靠手段。只看 `stamp_ms` 不够：指令可以是刚生成的，但基于 800 ms 前的观测。
- **`source` + `priority`** —— 仲裁与事后审计的基础。
- **`seq`** —— 网络抖动时旧 chunk 后到，直接丢，不要覆盖新的。
- **`mode` + `dof` 冗余携带** —— 驱动执行前和自己的 descriptor 对账。这是防"动作维度变了而上游不知道"的最后一道；不匹配要拒绝，而不是照着做。

载荷先用 `std_msgs/String` 装 JSON，和现有 perception 卡片一致、Dashboard 可直接渲染；30 Hz × 14 DOF 约 30 KB/s，完全够用。将来出现 200 Hz 或 50+ DOF 的场景再加一个紧凑二进制 profile，`schema` 字段留了升级位。

### 2.3 仲裁与安全检查链

指令源不止一个（VLA 卡片、LLM 直接调的 `loco`、遥控器、未来的导航卡片）。仲裁和检查都必须在驱动侧，因为只有它无条件在线。

这条链的设计参考了 MoveIt Pro 的 `ExecutePolicy`——它为同样的问题（把学习策略的 chunk 安全地送到控制器）给出了一组经过实战的防护，我们此前只有其中两条。按执行顺序：

| # | 检查 | 失败时 | 备注 |
|---|---|---|---|
| 1 | `schema` / `mode` / `dof` 与 descriptor 对账 | **拒绝并上报 error** | 这是配置错误，不是运行时抖动，必须让人看见 |
| 2 | 新鲜度：`ttl_ms` 过期 / `obs_stamp_ms` 过旧 / `seq` 回退 | **静默丢弃**（计数） | 正常的网络抖动，不是错误 |
| 3 | 仲裁：`priority` 最高优先，同级取 `seq` 最新 | — | 多源合法化，见下 |
| 4 | 单步增量超 `max_delta_per_step` | **限幅 + 告警** | 见下"4 与 5 的区别" |
| 5 | 硬限位：position `lower`/`upper`、`max_velocity`、加速度 | **拒绝整个 chunk** | 不 clamp，见下 |
| 6 | 碰撞复检 | 拒绝整个 chunk | **第一版做不到**，见下 |
| 7 | 力矩阈值（per-axis） | **立即中止运行** | 新增，见下 |
| 8 | 连续性：平滑、加密、blend；committed window | — | 新增，见下 |
| 9 | `watchdog_ms` 内无有效指令 | hold / 减速 / 停 | 已有 |
| 10 | 连续 N 个周期无有效指令 | **从 hold 升级为 abort**，ACP 报失败 | 新增，见下 |

e-stop 不在这条链里——它优先级最高、完全本地、绕过一切。

#### 4 与 5 的区别：一个限幅，一个拒绝

这两条看起来都是"超限"，处理方式却必须相反：

- **单步增量超限**（上游给的跳变太大）→ **限幅**。clamp 之后的点仍在策略意图的方向上，只是走慢一点；拒绝反而会让动作断断续续。
- **硬限位超限**（超出关节的物理极限或速度上限）→ **拒绝整个 chunk**。clamp 到边界会产生一条**既不是策略意图、也没有任何人验证过**的轨迹——策略以为机器人到了 A，实际停在边界 B，后续 chunk 全部建立在错误的前提上。MoveIt Pro 在这里同样是整块拒绝而不是逐点 clamp。

拒绝之后走 watchdog 路径（hold），不要试图"修复"这个 chunk。

#### 6. 碰撞复检：第一版做不到，要写明

MoveIt Pro 会把 chunk 的每个点对照 planning scene 做自碰撞与环境碰撞检查（带 `link_padding` 余量），并在整个运行期间持续监控场景，**中途出现的物体也能停机**。

**我们没有 planning scene。** 现有的只有驱动 `resource` 工具返回的 URDF，既没有场景表示，也没有 FK + 碰撞几何的运行时。诚实的结论是这条第一版补不上，不要假装能做。第一版的替代是三条弱得多的措施，必须知道它们弱在哪：

- descriptor 增加可选的 `workspace`（笛卡尔包围盒），越界拒绝——只挡得住大范围跑飞，挡不住桌面上的碰撞
- 依赖厂商 SDK 自带的保护（各家都有一些，但覆盖范围不明、不可依赖）
- **靠流程**：在仿真里跑到信任为止再上真机；真机首次运行降速、留人、留物理急停

碰撞检查列为后续项，需要先引入一个场景表示（最省事的路径是接一个 MoveIt/planning scene，而不是自己写）。

#### 7. 力矩阈值中止

MoveIt Pro 用 per-axis 的绝对力矩阈值捕捉"策略没预期到的接触"，越过即停。这条我们**可以做**：`mcp_manage.py` 的格式表里已经有 `sensor/force-torque`，`ControlSink` 可选订阅一路力矩 topic，越阈直接走中止路径。

有力矩传感器的机器人应当配这条。没有的，descriptor 里明确声明 `force_torque: null`，让人知道这台机器缺这道防护，而不是默认它有。

#### 8. 连续性与 committed window

chunk 之间要平滑、加密中间点、互相 blend，否则每次换 chunk 都是一次速度突变。更重要的是 **committed window**——已经提交给控制器、无法再撤销的那一段时间：

- `committed_window` **必须 ≥ 推理延迟的 p99**，否则当前 chunk 执行完而下一个还没到，机器人就会在两个 chunk 之间停顿
- 但 `committed_window` **越大，e-stop 到实际停止的延迟越大**

这是一个显式的取舍，要写进 descriptor 或卡片配置并在 `info()` 里回显，**不能给一个默认值就算了**。它同时也是 §5.2 里 RTC `inference_delay` 该填多少的依据——两个数来自同一次延迟实测。

#### 9→10. "暂停不是安全状态"

MoveIt Pro 文档里有一条值得整段抄过来的警告：chunk 迟到时机器人停在原地，**下一个 chunk 一到就毫无预警地恢复**，而那个 chunk 是基于最多 `policy_call_timeout` 之前的观测算出来的——图像可能已经是几秒前的。所以**永远不要把暂停当成可以靠近机器人的安全状态**。

我们的设计在这一点上比它保守：过期的指令被 §2.2 的 `ttl_ms` 直接丢弃，不会执行。**但这个保守性完全依赖 `ttl_ms` 配对**——配大了就退化成 MoveIt 的行为。所以：

- `ttl_ms` 要按实测延迟设定，且在 `info()` 里回显
- **恢复必须有显式提示**：活动流事件（参照 `peer_tool_call` 的做法），有 LED/声音的机器人应当同时用上。静默恢复是这条链里唯一一个"设计正确但操作上仍会伤人"的环节
- 另外要区分**迟到**与**调用失败**：迟到只是丢弃 + hold，不算失败；而 provider 调用失败或超时要计入失败计数，连续 N 次后从 hold 升级为 **abort**，通过 ACP 向 LLM 报错。否则机器人会无限期保持一个举着手臂的姿势，而 LLM 以为动作还在进行。

#### 多源合法化

这条链顺带解决了 agent-core 里 `_topic_clash` 的情形——"两张卡片发布到同一个 topic"现在是报错阻止启动，有了 `priority` 与仲裁之后它可以是合法的多源。

### 2.4 控制 SDK：共享实现，不要 14 个驱动各写一遍

放在 `phanthymotus-driver/common/control/`。驱动只提供 descriptor、一个 `apply(values)` 回调，以及可选的力矩源：

```python
class ControlSink:
    """订阅一个 control/* topic，执行 §2.3 的整条检查链，
    把通过检查的指令交给驱动自己的 apply()。"""
    def __init__(
        self,
        descriptor: dict,
        apply,                    # (values, gripper) -> None
        *,
        on_watchdog=None,         # 默认 hold；可覆盖为减速到停 / 回安全位姿
        on_abort=None,            # 连续失败后的终止路径，默认同 on_watchdog + 上报
        force_torque_source=None, # 可选，第 7 条
        clock=None,               # 注入时钟，便于测试
    ): ...
```

**这个类是整套方案的安全核心。** 它的测试不是可选项，且必须是假时钟 + 假 `apply` 的纯单元测试（不需要机器人、不需要 ROS）：

| 用例 | 断言 |
|---|---|
| descriptor 对账失败 | 拒绝且上报 error，`apply` 未被调用 |
| `ttl_ms` 过期 / `seq` 回退 | 静默丢弃，计数递增，`apply` 未被调用 |
| 单步增量超限 | `apply` 收到的是**限幅后**的值，且有告警 |
| 硬限位超限 | 整个 chunk 拒绝，`apply` 一次都没被调用 |
| 力矩越阈 | 中止路径被触发，且**优先于**当前 chunk 的剩余点 |
| watchdog 时序 | 恰好在 `watchdog_ms` 后触发，不早不晚 |
| 连续失败 | 在第 N 次后从 `on_watchdog` 升级为 `on_abort` |
| 优先级仲裁 | 低优先级源在高优先级源活跃期间完全不影响输出 |

这些用例在 Phase 0 就要全部通过——它们是"在接任何模型之前先把安全做对"的具体含义。

---

## 3. 底座设计

### 3.1 两条正交的轴

换模型和换机器人是两件独立的事。混在一个配置里，每个「模型 × 机器人」组合都要改一次代码。

```
                  ┌──────────────────────────────┐
                  │   VLAPlugin（卡片，不变）      │
                  │  MCP 契约 / ROS 收发 / 生命周期 │
                  │  限幅 / 资源声明 / e-stop 响应  │
                  └───────┬──────────────┬────────┘
                          │              │
              backend 轴  │              │  embodiment 轴
              （换模型）   │              │ （换连上的驱动）
                          ▼              ▼
              ┌───────────────────┐  ┌──────────────────────────┐
              │ VLAProvider       │  │ 画布连线 → 下游 descriptor │
              │  local / remote   │  │  §2.1，start 时协商        │
              │  openpi / lerobot │  │  对不上就拒绝启动          │
              │  unifolm / openvla│  │                          │
              └───────────────────┘  └──────────────────────────┘
```

### 3.2 embodiment 来自连线，不来自配置

和 agent-core 只用画布上连上的 MCP（`canvas_binding.py`）完全同构：**VLA 卡片只驱动画布上连到它输出口的那张驱动命令卡片。**

机制已经存在，不用新造：

- `canvas.js` 的 `_connections` 持有 `{fromCardId, fromPortIdx, toCardId, toPortIdx, format, fromTopic}`
- `api/config.py` 的 `_resolve_input_topics` / `_try_resolve` 在项目启动时把上游 `topic_out` 解析出来，作为 `input_topic` 传给下游卡片的 `start`
- 端口按 `format` 匹配，`control/joint` 就是这条连线的接口类型

**需要新增的只有一步：把下游的 descriptor 反向带给上游。** 现在 resolve 是单向的（上游 topic → 下游 `input_topic`）；`_try_resolve` 已经会调下游的 `info()` 拿结果，把其中的 `control_interface` 段回传给上游 VLA 卡片的 `config` 即可。

于是 VLA 卡片的配置里**不该有** `embodiment: "g1_dex1"` 这种字符串。它 `start` 时做一次协商：

1. 从连线拿到下游 descriptor
2. 和 provider 的 `capabilities()` 对账：`action_dim` vs `dof`、`mode`、单位、`control_hz` vs `rate.max_hz`
3. 对不上 → **拒绝启动**，并在错误信息里指明是哪一维对不上（"模型输出 32 维 joint_position，下游只接受 14 维" 而不是 "shape mismatch"）
4. 对得上 → 记录这次协商结果，`info()` 要能回显

这样换机器人是**改连线**，换模型是**改 provider**，两件事互不干扰，也都不需要改代码。

### 3.3 卡片骨架

```python
# actucore/plugins/vla.py
#
# PREFIX 不能含下划线 —— dispatch 用 full_name.partition("_") 拆前缀。

class VLAPlugin:
    PREFIX = "vla"

    TOOLS = [{
        "name": "vla",
        "type": "processor",          # processor/actuator 都会过 ACP barrier
        "multiInstance": False,
        "description": "语言指令驱动的端到端操作策略",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": ["start", "stop", "info", "config"]},   # info 必须有，否则永远离线
                "task":   {"type": "string", "description": "自然语言指令"},
            },
            "required": ["action"],
            "x-completion": {"actions": ["start"], "timeout": 120},
            "x-resource":   ["arm_l", "arm_r"],      # 见 3.5
            "x-hooks":      {"on_interrupt_all": {"action": "stop"}},
        },
        "configSchema": {
            "type": "object",
            "properties": {
                # ── backend 轴 ────────────────────────────────────────
                # enum 在运行时由已发现的 provider 目录生成，不写死（见 4.3.3）
                "provider":    {"enum": ["<discovered>"], "scope": "shared"},
                # 该 checkpoint 的输入特征键映射：我们的 topic → 模型期望的 key
                # LeRobot 系必填，随数据集走（observation.images.<cam> / observation.state）
                "feature_map": {"type": "object", "scope": "shared"},
                # 照 LLM 的做法：只有 endpoint + key + model，没有部署相关的任何东西
                "endpoint":    {"type": "string", "scope": "shared"},
                "api_key":     {"type": "string", "scope": "shared"},
                "model_dir":   {"type": "string", "default": "/models/vla",
                                "scope": "shared"},
                "unnorm_key":  {"type": "string", "scope": "shared"},   # 无默认值，见 3.6
                "fallback":    {"enum": ["none", "local"], "default": "none",
                                "scope": "shared"},
                # ── embodiment 轴 ─────────────────────────────────────
                # 没有 embodiment 字段 —— 动作面由画布连线决定（见 3.2），
                # 下游 descriptor 在 start 时协商得到。这里只放本卡片自己的偏好，
                # 且必须被 descriptor 的 rate.max_hz 夹住。
                "control_rate_hz": {"type": "number", "default": 30, "scope": "instance"},
            },
            "required": ["provider"],
        },
        "topic_in":  [{"format": "video/mjpeg", "desc": "主视角"},
                      {"format": "state/joint", "desc": "本体状态"}],
        "topic_out": [{"format": "control/joint", "desc": "关节指令"}],
    }]

    def get_tools(self) -> list[dict]: ...
    def dispatch(self, name, args) -> dict | None:
        ...   # 必须返回 plain dict，不要返回 [{"type": "text", ...}]
```

### 3.4 并发（actucore 和 perception 一样是 `ThreadingHTTPServer`）

`start` / `stop` / `config` 会并发到达同一个插件实例。规则和 `perception/README.md` 完全一致，这里只列最要命的四条：

1. per-instance 状态字典的读-改-写用 `threading.RLock` 保护
2. **不要**在持锁状态下调 `node.start()` / `node.stop()` / 加载模型 —— 否则 `stop` 排在 `start` 后面，**再也停不掉一个正在驱动电机的控制回路**
3. 节点**先注册再启动**，让并发的 `stop` 能找到它
4. `stop` 先非阻塞 `request_stop()` 置位，**再**去拿锁

对普通 perception 卡片这是"节点泄漏"，对 VLA 这是"停不下来的机器人"。

### 3.5 `x-resource` 必须声明

`parse_resources` 对未声明返回 `None`，语义是**与一切互斥**。VLA 卡片若不声明，一跑起来 TTS 讲解、导航全被 barrier 挡住，机器人连话都说不了。按实际占用的物理通道声明（`arm_l` / `arm_r` / `base` / `waist`），不要图省事写一个 `robot`。

### 3.6 会静默出错的配置必须设为必填

`unnorm_key`（OpenVLA / UnifoLM 系列都有）选错不会报错，只会让**动作尺度全错**。在一个真的驱动电机的系统里，这类配置：

- 不给默认值，`required` 里必须有
- `start` 时向 provider 查询可用值并校验，对不上直接拒绝启动
- `info` 要把当前生效值回显出来

---

## 4. Provider：本地与云统一

### 4.1 接口

```python
class VLAProvider(Protocol):
    def capabilities(self) -> dict:
        """握手。返回 {model, action_dim, chunk_size, control_hz,
        needs_state, n_cameras, image_size, supports_rtc, unnorm_keys}。
        卡片用它校验 EmbodimentProfile，不匹配就拒绝 start。"""

    def infer(self, obs: Observation, *, inference_delay: int = 0) -> np.ndarray:
        """返回 (T, D)，已按 EmbodimentProfile 反归一化到工程单位。
        单步模型（OpenVLA base）返回 T=1。"""

    def health(self) -> bool: ...
    def close(self) -> None: ...
```

```python
@dataclass
class Observation:
    images: dict[str, np.ndarray]   # {"main": HWC uint8, "wrist": ...}
    state:  np.ndarray | None       # 本体状态，OpenVLA base 为 None
    prompt: str
    t_capture_ms: int               # 观测采集时刻，用于 RTC 的 inference_delay
```

**`infer` 是唯一跨本地/云的抽象。** 上面的卡片逻辑（watchdog、限幅、发布、ACP）对两种 provider 完全一样。

### 4.2 LocalProvider

进程内加载模型，`infer` 直接推理。

- 只在 `provider: local` 时才 import torch / lerobot —— 依赖放在 Dockerfile 自己的 `RUN` 层，不要进基础层（actucore 镜像刻意做薄）
- **单独进程还是进程内**：如果本地模型用 ONNX Runtime，必须起独立子进程。同进程两份 ORT 共享一个 provider 桥，第二个 CUDA session 必挂（jp6.1 异常 / jp5.11 SIGSEGV），`perception/plugins/kokoro_worker.py` 有完整记录
- 显存/内存要在 `capabilities()` 里自检，装不下就明确报错，不要 OOM 到把 perception 一起拖死
- 权重按 §4.5 lazy download（COS + size/SHA256 pin）+ lazy load（`start()` 里加载，先报 `loading`）

### 4.3 RemoteProvider

按后端各写一个薄 adapter。它们只有传输和 payload 形状不同：

| adapter | 传输 | 说明 |
|---|---|---|
| `openpi_ws` | WebSocket | openpi `serve_policy.py`，用官方 `openpi-client` |
| `lerobot_async` | gRPC/队列 | LeRobot `PolicyServer` / `RobotClient`，自带 action queue |
| `unifolm_http` | HTTP | 宇树 `run_real_eval_server.sh`，README 给的是 ssh -L 隧道 |
| `openvla_http` | HTTP JSON/msgpack | `vla-scripts/deploy.py`，单步、无 state |

共同要求：客户端侧 resize（π 系预训练常用 224）、uint8、连接在回合外建立、断连即触发 watchdog。

### 4.3.1 LeRobot 是最高杠杆的一个 adapter

`lerobot_async` 不是"SmolVLA 的一种部署方式"，它是**一次接入换整个 policy zoo**。实测 `huggingface/lerobot` 的 `src/lerobot/policies/` 目录：

```
act  diffusion  eo1  evo1  fastwam  gaussian_actor  groot  lingbot_va
molmoact2  multi_task_dit  pi0  pi05  pi0_fast  rtc  smolvla  tdmpc
vla_jepa  vqbet  wall_x  xvla
```

对照 §6 引的 LIBERO 榜：**EO1（98.2）、X-VLA（98.1）、GR00T-N1.6（97.0）、π0.5（96.9）全都在里面**，`rtc` 也是一等公民。`src/lerobot/async_inference/` 里的 `policy_server.py` / `robot_client.py` 是**策略无关**的，换 checkpoint 不换代码。仓库 Apache-2.0。

所以对"追 SOTA"这个目标，`lerobot_async` 是主力通道，不是备选。它应当从"跟 local 一起顺带做"提到**第二位**。

**但"未来任何 LeRobot 模型都能支持"这句话要限定在传输层。** 准确说法是：

- ✅ **调用与传输通用** —— 起 server、发观测、收 chunk，对 `act` 和对 `xvla` 是同一套代码
- ⚠️ **embodiment 不通用** —— 每个 checkpoint 仍有自己的 `action_dim`、chunk 长度、是否吃语言、norm stats（烘在 checkpoint 里）、以及**输入特征键名**（`observation.images.<cam>` / `observation.state` 的具体命名随训练数据集走）

后者正是 §3.2 协商要解决的事：provider 的 `capabilities()` 把这些报上来，和下游 descriptor 对账，对不上就拒绝启动。所以接一个新 LeRobot 模型的成本 ≈ **一份特征键映射配置**，不是一次开发。这个成本要在卡片 config 里显式暴露（`feature_map`），不要藏在代码里。

**是否用 LeRobot 的 `pi05` 取代 `openpi_ws`？** 两个都留。openpi 是 PI 自家实现、RTC 原作者、文档最全，作为**上游参考**用来判断 LeRobot 移植版是否有行为差异；LeRobot 负责**广度**。如果将来要砍一个，砍 `openpi_ws`，不是 `lerobot_async`。

### 4.3.2 实现顺序

| 顺序 | provider | 阶段 | 覆盖 | 前置条件 |
|---|---|---|---|---|
| 1 | **`mock`** | Phase 0 | 进程内；可配成 T=1 / 不吃 state | 无，零依赖 |
| 2 | **`openpi_ws`**（π0.5） | Phase 1 | WebSocket + chunk + state | 边缘服务器；Apache-2.0 |
| 3 | **`lerobot_async`** | Phase 1 尾 | 策略无关的 gRPC/队列 → policy zoo | 同一台边缘服务器 |
| 4 | **`local`**（SmolVLA） | Phase 2 | 进程内 + 真模型 | Orin NX 实测通过（§6.3），复用 3 的依赖 |
| 5 | **`unifolm_http`** | Phase 2 | HTTP + chunk，G1 embodiment 最优 | 见下 |
| 6 | **`openvla_http`** | Phase 3 | HTTP，退化情形的真实压测 | 目标是 **OFT**，见下 |

**`mock` 排第一，而且不是额外工作。** Phase 0 本来就要一个假策略卡片验证通路和急停。此后它是唯一能在无网络、无 GPU、无边缘机时跑的回归工具——改了 `ControlSink`、改了协商、升了 `schema` 版本都用它验。

**`unifolm_http`：做，但产物要打标。** 它和机队的 embodiment 匹配度最好（训练数据就是 G1 + Dex1，12 个数据集开源，LeRobot v2.1 格式），接上之后 G1 的适配量明显小于 π0.5，值得做评测。唯一要做的事是：`unifolm-vla` 仓库没有 LICENSE 文件，所以这个 adapter 及其权重**在授权明确前不要打进产品镜像**，配置里默认关闭、卡片 `info()` 里回显授权状态。这样评测照做，不会有人误当成可发布能力。

**`openvla_http`：做，但目标是 OpenVLA-OFT，不是 base。** base OpenVLA 不吃本体状态、单步输出 3–5 Hz、固定 7-DoF EEF delta，追 SOTA 的意义不大；真正有竞争力的是 **OFT**——LIBERO 平均 97.1，高于 π0（94.2）和 GR00T-N1（93.9），而且它支持 action chunking（`num_open_loop_steps 25`）、多相机（`--num_images_in_input 3`）和本体状态（`--use_proprio True`）。附带价值是它的 `unnorm_key` 与训练配置必须完全对齐才能加载，正好压测我们 §3.6 "会静默出错的配置必须必填" 那条规矩是否真的兜住了。

### 4.3.3 既然要持续追 SOTA，provider 不能是硬编码 enum

上面六个只是起点。要让"下一个 SOTA 出来就能试"成为常态，provider 层要按 actucore 卡片同样的方式做成**目录发现 + duck typing**：

- `actucore/plugins/vla/providers/<name>.py`，实现 §4.1 的四个方法即可被发现，卡片代码不动
- `configSchema.provider` 的 enum 在运行时由已发现的 provider 生成，不写死
- 一套 **provider 契约测试**：任何 provider 必须通过同一组用例（`capabilities()` 字段完整、chunk 形状与 `action_dim` 自洽、超时行为、`close()` 幂等）。新增一个 provider 的验收标准就是这套测试通过 + 在 `mock` 之外的真硬件上跑一次

有了这两条，加一个新模型的成本落在"一个文件 + 一份特征键映射"，而不是每次都动卡片。

### 4.4 fallback 策略

`fallback: local` 时，远端连续 N 次超时后切本地小模型。**但这不是"降级继续干活"**——本地模型的能力和远端不是一个量级，盲目接管更危险。建议语义是：

- `fallback: none`（默认）→ 断连即 watchdog 接管，机器人停住，向 LLM 报 error
- `fallback: local` → 本地模型只负责**把当前动作安全收尾**（回到 home / 松开夹爪 / 停住），不接着执行任务

这条要显式写进卡片文档，否则会被误当成高可用。

### 4.5 模型与权重的生命周期

**本地和云端是两套完全不同的问题，不要套同一个方案。**

| | 本地（机器人上） | 云端 / 边缘服务器 |
|---|---|---|
| 形态 | 按需拉取、按需加载、停了要还内存 | 长驻进程，启动即加载并预热 |
| 要解决的 | lazy download / lazy load（§4.5.1–4.5.3） | 可达性、预热、版本、并发、可用性（§4.5.5） |
| 约束来源 | 57 GB eMMC、7 GB 内存、和 perception 抢资源 | 网络可达性、多机器人共享、不能中途换模型 |

**lazy 这套只对本地成立。** 云端没有"卡片被创建但未启用"这回事——server 起来就是为了服务推理，按请求加载模型反而是错的。

#### 4.5.1 本地：权重不进 repo、不进镜像

COS 是机器人侧的唯一来源，落盘到 `/opt/embodied/models`（`actucore/deploy/service.yml` 已挂成容器内 `/models`）。镜像刻意做薄的前提就是这个。**直接复用 perception 的下载器，不要重写。**

`perception/utils/model_downloader.py` 已经是这套做法的成熟实现，`ensure_verified_bundle(name, model_dir, base_url, files)` 的语义正是我们要的：

```
已存在 → size 校验 → SHA256 校验 → 直接复用
否则   → flock → 拿到锁后再查一次 → 下载（带重试）→ 校验 → 原子替换
返回 {filename: 绝对路径}
```

它里面有几个**已经踩过的坑**，重写一份必然重踩：

- **check_file 必须在模型完整之后才出现。** 早期直接写最终文件名，第二个调用者一看到文件就去加载，拿到半个模型——现象是 `Load model from ... failed: Protobuf parsing failed`，而同一时刻日志还显示该文件下载到 30%。所以一律 `tempfile` + `os.replace`，且临时文件必须落在**目标目录**里（跨文件系统的 rename 不是原子的）。
- **归档先解到 staging 再 merge**，半解压的归档同样不能提前暴露 check_file。
- **多实例共享 `/models`**，必须 `flock` 串行化，且拿到锁后要**再查一次**——否则冷启动时每个进程各下一份。
- **bundle 的 key 是相对路径不是文件名**（VITS2 就带 `engines/jp61/flow.plan` 这种层级），要拒绝绝对路径、`..`、反斜杠。
- **size 和 SHA256 都要 pin**，只校验其一都不够。

**落地方式**：它现在在 `perception/utils/` 下，actucore 是另一个镜像。建议提到 `phanthymotus/common/model_downloader.py`，两个镜像各自安装，而不是复制一份（复制必然漂移，而漂移的是安全校验代码）。改 Dockerfile 时注意把新增的 COPY/ARG 放到**文件末尾**，否则整个 16 GB 的 perception 镜像要重建。

#### 4.5.2 本地：VLA 权重相对 perception 的三点不同

1. **只有 `local` provider 需要在机器人上下权重。** 远端 provider 一个字节都不用下——边缘服务器上的权重由那台机器自己管（可以用同一个 downloader）。所以机器人侧长期只有 SmolVLA 这一份，这也是选它的又一个理由。

2. **体积大一个量级，要做容量预检。** SmolVLA ~1 GB，π0.5 ~7 GB，UnifoLM 7B 级 ~15 GB，而机器人是 57 GB 的 eMMC（perception 的 `file_intake` 保留策略就是为这个写的）。下载前先查可用空间，不够就**明确报错**，不要把盘写满——盘满之后 SQLite、日志、docker 一起出问题，排查方向会完全跑偏。

3. **无论上游在哪，机器人只从 COS 拉。** 取一次 → 记录上游来源与 revision → 转存 COS 并 pin size/SHA256 → 机器人只认 COS。三个理由：CLAUDE.md 的既定规则（静态资源只放 COS，不 hotlink 不受控的东西）；办公网访问境外站点不可靠；以及上游删改权重不会让现网变得不可复现。上游怎么选见 §4.5.4。

   **许可证边界**：`unifolm` 系权重在授权明确前**不要上传到 COS 的公开路径**——公开 URL 等同于再分发，而 CC BY-NC-SA / 无许可证都不允许。评测期间用私有路径或本地挂载。

#### 4.5.3 本地：lazy load 的三条规矩

1. **模型在 `start()` 里加载，不在 import、不在 `__init__`。** 卡片被创建 ≠ 卡片被启用；在 import 时加载会让一个从未被连线的 VLA 卡片照样吃掉几个 GB。

2. **`start()` 立刻返回 `{"state": "loading"}`，不要返回 ready。** agent-core 认这个状态（`api/config.py` 的 `_settle_loading_item`），会挂一个 watcher 并在 Dashboard 上显示"模型加载中"。TTS 曾经因为直接报就绪，操作员拿到一张声称 ready、25 秒后才能出声的卡片。VLA 的加载时间只会更长，而"就绪了但不动"在一台机器人上比在一个音箱上更容易被误判成故障。下载进度走 `progress_cb`，Dashboard 一行状态即可。

3. **加载在后台线程，且绝不持锁**（§3.4）。否则 `stop` 会排在 `start` 后面——一张正在加载 7 GB 权重的卡片在加载完成前无法被取消，而取消它恰恰是操作员发现配错了之后第一件想做的事。

**卸载同样重要。** `stop` 要真正释放显存：Orin NX 总共 7 GB，一张停了还占着 GPU 的 VLA 卡片会把 perception 挤死。但 CUDA context 一旦建立就不会还给系统（ASR 的 ~1.4 GB context 就是这样），**所以本地 provider 更应该放独立子进程——只有进程退出才真正把内存还回去**。这与 §4.2 出于 ONNX Runtime 冲突得出的结论是同一个方向，这里是它的第二个理由。

#### 4.5.4 上游来源：双发的一律走 ModelScope

**同一个模型在 HuggingFace 和 ModelScope 都有时，一律用 ModelScope** —— 国内带宽差一个量级，而 VLA 权重是 1–15 GB 的量级，这个差别决定的是"十几分钟"还是"一下午"，以及断点重来的次数。

已核实（ModelScope API 返回 200）：

| 模型 | ModelScope | 说明 |
|---|---|---|
| `lerobot/smolvla_base` | ✅ | 本地 provider 的候选，优先走这里 |
| `unitreerobotics/UnifoLM-VLA-Base` | ✅ | 宇树系在 ModelScope 有完整镜像 |

**例外要单独处理：openpi 的权重不在 HF，也不在 ModelScope。** README 明确写了 checkpoint 存放在 **`gs://openpi-assets/checkpoints/...`**（Google Cloud Storage），由 `download.maybe_download()` 自动拉取并缓存到 `~/.cache/openpi`（可用 `OPENPI_DATA_HOME` 改路径）。这从国内基本不可达，所以 π0.5 需要**一次性境外拉取 + 转存**，不能指望边缘服务器直连。这一步要排进 Phase 1，别等到装服务器那天才发现下不动。

优先级写死成：**ModelScope > HF > 上游自有存储（`gs://` 等）**，且无论从哪来，最终都转存 COS 并 pin size/SHA256。

#### 4.5.5 云端：不是 lazy，是另外五个问题

> 这一节是写给**推理服务 repo** 的需求，不是 `phanthymotus` 的实现计划（§5.4）。列在这里是为了说明"lazy 那套只对本地成立"，以及在定协议时要为这些留出字段。

server 是长驻进程，模型在**进程启动时**加载一次。这里没有 lazy 的余地，要解决的是另一组：

1. **可达性与预取。** 见 §4.5.4 —— π0.5 的 `gs://` 源是硬障碍。服务器部署流程里要有一步"权重已就位"的校验，而不是首次请求时才发现没有。

2. **预热，且健康检查要等预热完成。** 首次推理明显慢于稳态（perception 侧的可比数据：ASR 在 GPU 上首推 ~1.7 s vs 稳态 58 ms，来自懒加载 kernel、cuDNN autotune、内存池）。Jetson AI Lab 给的 π0.5 数字都是稳态值。**`/capabilities` 在预热完成前不该返回 ready**，否则第一台连上来的机器人替所有人吃掉冷启动，而它正举着手臂。

3. **版本与可复现。** §5.2 响应里的 `model` 指纹必须落到 checkpoint 粒度并进日志。换 checkpoint 时**不能让正在执行的 session 中途换模型**——`session_id` 要绑定 checkpoint 版本，旧 session 跑完在旧版本上，新 session 才落到新版本。动作策略中途换底座，表现出来是动作突然不连续，而日志上什么都看不出来。

4. **并发与隔离。** 一台 server 服务多台机器人时：不做 batching 的话吞吐就是 `1/latency`，按机器人数量算容量，别按"够用"感觉估。单请求必须有服务端超时，一台机器人的慢请求不能拖垮队列——机器人侧虽然有 watchdog 兜底（停住），但停住的是别人。

5. **可用性。** server 挂掉 = 所有连着的机器人停（watchdog 生效，安全但停工）。需要健康检查 + 告警，且 server 重启后机器人要能自动重连，不需要人去每台机器上点一次。

---

## 5. 云端设计与接口

### 5.1 部署拓扑

| 形态 | 用途 | 评价 |
|---|---|---|
| **局域网边缘服务器**（一台带 RTX 的机器跑 policy server） | 实时闭环 | **主路径**。网络 RTT 个位数 ms + 推理 20–50 ms，落在 RTC 的舒适区，链路自控 |
| 公网云 | 开发、评测、数据回流、离线微调 | 实时闭环仅限准静态任务。本环境网络状况不适合（办公 WiFi 有 MAC 认证门禁、机器人地址常变） |
| 端上 | 小模型 / 兜底 | 见第 6 节 |

### 5.2 线协议

即使用各家现成 server，**我们自己这一侧的契约要固定下来**，否则换后端就要改卡片。建议 RemoteProvider 对外统一成下面的形状，各 adapter 负责翻译：

```jsonc
// POST /infer  (或 ws 的一帧)
{
  "schema": "motus.vla/1",           // 版本号，必填，不匹配即拒绝
  "session_id": "uuid",              // 一次 start 一个，服务端可据此缓存/复位
  "seq": 128,                        // 单调递增，用于丢弃乱序回包
  "t_capture_ms": 1789234567123,     // 观测采集时刻（不是发送时刻）
  "prompt": "把红色方块放到黑色区域",
  "images": { "main": "<base64 jpeg>", "wrist": "<base64 jpeg>" },
  "state":  [0.12, -0.34, ...],      // 工程单位，服务端负责归一化
  "inference_delay": 3,              // RTC：以 timestep 计的预期延迟
  "unnorm_key": "g1_dex1"
}
```

```jsonc
// 响应
{
  "schema": "motus.vla/1",
  "seq": 128,
  "actions": [[...], [...]],         // (T, D)，工程单位
  "action_dim": 14,
  "chunk_size": 50,
  "t_server_recv_ms": ..., "t_server_done_ms": ...,   // 用于延迟归因
  "model": "pi05-ki@2026-02-01",     // 模型指纹，进日志
  "warnings": []
}
```

```jsonc
// GET /capabilities —— start 前握手一次，用于校验 EmbodimentProfile
{ "schema": "motus.vla/1", "model": "...", "action_dim": 14, "chunk_size": 50,
  "control_hz": 30, "needs_state": true, "n_cameras": 2, "image_size": 224,
  "supports_rtc": true, "unnorm_keys": ["g1_dex1", "libero_spatial"] }
```

设计要点：

1. **`schema` 版本号必填**。动作维度变了而客户端不知道，等于随机驱动电机。
2. **`t_capture_ms` 而非发送时刻**。RTC 的 `inference_delay` 是按观测的年龄算的，不是网络 RTT。
3. **`seq` + 丢弃乱序**。网络抖动时旧 chunk 后到，直接丢，不要覆盖新的。
4. **服务端不持有机器人状态**。每次请求自包含，服务端可随时重启、可水平扩容。`session_id` 只用于缓存，不用于正确性。
5. **超时由客户端定，且必须短于 watchdog**。`timeout_ms < watchdog_ms` 是硬约束。
6. **不做重试**。一次 infer 超时就让它过去，下一帧观测更新鲜；重试只会让动作更陈旧。
7. **单位在客户端侧统一**。服务端收工程单位、回工程单位，归一化是服务端的内部事务。归一化跨进程分摊是最容易出错的地方。

### 5.3 安全

云 policy server 是一个**能直接驱动电机的外部端点**，信任级别等同于 `operator` 角色的 peer。要求：

- 传输 mTLS，证书 pin 住；endpoint 白名单，不接受运行时任意改
- **watchdog 和 e-stop 的代码路径不得经过网络**，也不得在 actucore 里（见 5.5）
- 每次 start / 每次 fallback 切换都上活动流（参照 `peer_tool_call` / `peer_tool_result` 的做法）
- 画布绑定仍然是唯一的人工授权点：不连线，LLM 就调不到
- 相机帧持续外发，公网部署前要过隐私/合规

### 5.4 代码与仓库边界：按部署位置划分，不按"是不是推理"划分

**机器人本机上的推理，项目自己实现；机器人之外的推理，只消费。**

前半句是既成事实，不是新规定：perception 整层就是在机器人上实现推理——ASR、TTS、OCR、人脸全部进程内加载模型并执行，还要自己处理 ONNX Runtime 冲突、CUDA context 不归还、GPU/CPU 选型。§4.2 的 `LocalProvider`（SmolVLA）属于同一类，必须在项目内实现。

后半句才是本节要定的事。区分标准不是"是不是推理"，而是：

> **谁为这份推理的资源竞争负责。**

本机推理和 perception 抢同一块 7 GB 内存、同一颗 GPU、同一条 eMMC——「ASR 和 VLA 谁该拿那 2 GB」这个权衡**只有具身项目做得了**，因为只有它知道机器人此刻在干什么。所以它必须在项目内，和调度它的卡片放在一起。

机器人之外的推理不和任何机器人组件争资源。它的权衡是吞吐、成本、扩缩容——那是 serving 的学科，和"机器人该不该说话"没有任何关系。

**远端这一半，本项目已经做过一次而且做对了——LLM。** agent-core 消费 LLM 的全部代码是：

```python
# src/config.py —— 配置里只有一组 {url, key}
'client': {'llm': [{'url': ..., 'key': ..., 'model': ...}]}

# src/client/llm.py
openai.AsyncOpenAI(base_url=config_it['url'], api_key=config_it['key'], ...)
```

整个仓库**没有一行 vLLM / SGLang / TensorRT-LLM 的部署代码**，没有 GPU 编排，没有批处理调度，也没有人在这里看 p99 和显存利用率。注意这和 perception 里满是模型加载代码并不矛盾——LLM 跑在机器人之外，ASR 跑在机器人上，两者适用不同的规则。

**VLA 的远端一半应当照 LLM 办理。** 具体地：

| 内容 | 放哪 |
|---|---|
| **本地推理实现**（`LocalProvider`、模型加载、显存管理、lazy load） | **`phanthymotus/actucore/`** —— 和 perception 同类 |
| 协议 spec（`motus.vla/1`，即 §5.2） | `phanthymotus/docs/` —— 是**文档**，不是库 |
| 远端客户端实现（provider adapter）+ 契约测试 | `phanthymotus/actucore/` |
| 配置项 `{endpoint, key, model}` | `phanthymotus` 的 configSchema |
| **远端服务端实现、部署编排、权重清单、预热、批处理、指标、扩缩容** | **`phanthymotus-cloud`**（GitLab `embodied-ai/phanthymotus-cloud`，`master`，当前为空仓库） |

**仓库拆分不等于抽象拆分。** 两侧共用 §4.1 的同一个 `VLAProvider` 接口，卡片不知道自己连的是进程内的 SmolVLA 还是机房里的 π0.5——`infer()` 的语义、单位、`capabilities()` 协商、契约测试全部一致。拆的是**谁维护、在哪部署、按什么 KPI 考核**，不是接口。

**上一版本文档在这里提出的 `phanthymotus/edge/` 目录作废。** compose、权重清单、预热逻辑、metrics exporter 全部属于服务端 repo，不属于这里。本文 §4.5.4 / §4.5.5 里关于云端权重与预热的内容，是写给那个 repo 的**需求**，不是这个 repo 的实现计划。

#### 为什么必须分开

两个 repo 的 KPI、工程学科、值班对象都不一样：

| | `phanthymotus-cloud` | `phanthymotus` |
|---|---|---|
| KPI | p99 延迟、吞吐、GPU 利用率、可用性 | 机器人行为正确与安全 |
| 测试 | 压测、并发、故障注入 | 真机验证、假时钟、契约测试 |
| CI | 需要多卡 GPU runner | 无 GPU，或单张 Jetson |
| 发布 | 随时扩容、灰度切 checkpoint | 跟着机器人现场走 |
| 出事找谁 | 值班的 SRE | 在机器人旁边的人 |

混在一起的直接后果是双向的：一个只能在多卡机上跑的压测进了机器人仓库的 CI；或者反过来，服务端为了配合机器人仓库的发布节奏而不能随时扩容。

注意本机推理**不适用**这张表：它的 CI 就是 Jetson，它的发布就是跟机器人走，它出事找的也是机器人旁边的人。所以它留在 `phanthymotus` 里是一致的，而不是例外。

#### 一处不能照搬 LLM

LLM 有 OpenAI 这个既成标准可以对齐，**VLA 没有**（§4.3.4）。所以我们得自己扮演"写 spec 的那一方"。但归属方式仍然照 OpenAI 的模式：

- **spec 是文档，不是共享库。** OpenAI 写 spec，vLLM / SGLang / Ollama 各自实现，客户端 SDK 独立演进——三方从不共享一个代码包。
- 对应到我们：spec 在 `phanthymotus/docs`，客户端实现在 actucore，服务端实现在独立 repo 各自对着 spec 写，**契约测试两边各跑一份**。
- 这比抽一个共享 schema 包更耐用：共享库会让两侧耦合到同一份实现细节，而 `schema` 版本号（§5.2 必填）本来就是用来解耦发布节奏的。

#### 服务端 repo 的边界

`phanthymotus-cloud` **不是 "vla-server"**。同一套 serving 基础设施会同时托管 VLM（如 ER-1 一类的具身推理模型）、VLA 策略、以及将来的世界模型——按"推理服务"命名，不要按某一类模型命名，否则第二类模型进来时又要开一个 repo。它对外会有不止 `motus.vla/1` 一个接口。

#### key 与计量

同样照 LLM：**key 由服务端签发与校验**——只有它在数据路径上，也只有它能执行限流。账单与对账可以回流 `resource-center`（那里已有账号体系、Prisma、共享 DB），走**带外**通道：服务端异步推 usage 事件（`session_id`、`model` 指纹、推理次数、耗时，§5.2 的响应里已经全有），请求本身不经过计费系统。这样计费挂了只影响账单，不影响机器人。

**控制回路绝不能穿过计费网关**（§5.3），这条不因为拆了 repo 而改变。

附带的两个好处，不是拆分的理由但顺带解决了：许可证受限的模型（`unifolm` 系）天然不会出现在开源仓库里；服务端凭据也不必再考虑怎么躲开 `phanthymotus` 的公开仓库。

### 5.5 Watchdog 放在驱动侧，不在卡片里

即 §2.4 的 `ControlSink`：

```
VLA 卡片 ──motus.control/1──> /robot/arm/cmd ──> Driver 命令卡片
                                                    │
                                         ┌──────────┴───────────┐
                                         │ ControlSink          │
                                         │  descriptor 对账      │
                                         │  ttl / obs_stamp 丢弃 │
                                         │  max_delta 限幅       │
                                         │  watchdog → hold/停   │
                                         └──────────────────────┘
```

理由：卡片崩了、容器被 OOM kill 了、网络断了——这三种情况下卡片里的 watchdog 都不会执行。**只有驱动侧的超时才是无条件生效的。** 限幅与对账同理：不信任上游给的任何一个数，包括维度。

---

## 6. 本地模型选型

### 6.1 硬件事实

```
$ ssh nvidia@10.100.121.16 'cat /proc/device-tree/model; free -g'
NVIDIA Jetson Orin NX ... Super
Mem:  total 7   available 3      # 已在跑 perception
```

Orin NX 8GB，跑着 perception 时可用约 3 GB。这是选型的硬约束。

### 6.2 候选

| 模型 | 规模 | 本地可行性 | 备注 |
|---|---|---|---|
| **SmolVLA-450M** | 450M（含 ~100M action expert） | **唯一现实候选** | flow matching、action chunk、LeRobot 原生异步推理与 RTC。官方称可在 CPU 上跑 |
| π0 / π0.5 | ~3.3B | ✗ | bf16 权重 6.6 GB，装不下 |
| UnifoLM-VLA-0 | Qwen2.5-VL-7B + head | ✗ | 同上，且要 CUDA 12.4 + flash-attn |
| UnifoLM-WLA-1.0 | 6B | ✗ | 且 WLA-Base 权重尚未放出 |
| OpenVLA-7B | 7B | ✗ | 且 base 不吃本体状态、单步输出 |
| ACT / Diffusion Policy | ~10–100M | ✓ | 无语言条件，只能做单任务，作为兜底动作或对照组 |

### 6.3 SmolVLA 的证据与空白

- 架构上为边缘做了减法：视觉塔跳过一半层；action expert 取中间层（~L/2）特征而非最后一层；推理时不做 image tiling，只喂 global image + pixel shuffle
- 异步推理：响应快 ~30%，固定时间内完成任务数约 2×；关键参数 `actions_per_chunk`、`chunk_size_threshold`
- 支持 RTC（10 步 flow matching 时官方建议 guidance 10.0）

**公开资料里没有 Orin NX 上的 SmolVLA 延迟/内存数字。** 最接近的是 NanoVLA 在 Orin Nano Super 8GB 上对 SmolVLA 的对比（LIBERO-Goal，绝对 Hz 只在图里）。另一个量级参考：量化后的 LiteVLA-Edge 在 **AGX** Orin 上约 150.5 ms（~6.6 Hz）。Orin NX 带宽和算力都低于 AGX，要更保守。

**所以本地这条路必须先做一次实测**，不要按公开数字排期。实测项：单帧延迟 p50/p99、内存峰值、与 perception 共存时的相互影响、连续跑 30 min 的热降频。

### 6.4 本地模型的定位

按 4.4 节，本地模型的第一用途**不是**替代云端跑任务，而是：

1. 安全收尾（断连时把动作停在安全状态）
2. 低风险单任务（抓取固定物体、桌面整理）不必依赖网络
3. 作为云端结果的合理性对照（可选，后期）

---

## 7. 模型许可证（选型前必须过的一关）

| 模型 / 仓库 | 许可证 | 可商用 |
|---|---|---|
| openpi（π0 / π0.5） | Apache-2.0（仓库 LICENSE 已核对；权重条款以官方说明为准） | ✅ |
| LeRobot / SmolVLA | Apache-2.0 | ✅ |
| UnifoLM-WMA-0 | CC BY-NC-SA 4.0 | ❌ 非商用 + ShareAlike |
| UnifoLM-WLA-1.0 | CC BY-NC-SA 4.0 | ❌ |
| UnifoLM-VLA-0 | **仓库无 LICENSE 文件，README 未提** | ❌ 默认保留所有权利 |
| OpenVLA | 见其仓库 | 需确认 |

宇树系模型和我们机队的 embodiment 匹配度最好（训练数据就是 G1 + Dex1，12 个数据集开源，LeRobot v2.1 格式），但许可证是硬阻断。**研究和 demo 可用，产品化前必须先谈授权**。

---

## 8. 分阶段计划

### Phase 0 — 通路与安全（不含模型）

1. 定稿 `motus.control/1`（§2.1 descriptor + §2.2 消息），写进 `phanthymotus-driver/README_dev.md`
2. 实现 `phanthymotus-driver/common/control/`（`ControlSink` + 检查链），**带单元测试**：§2.4 那张表里的八个用例全部通过（假时钟 + 假 `apply`，不需要机器人也不需要 ROS）
3. 在**一个**驱动上接入它，命令卡片声明 descriptor、`topic_in: control/joint`
4. agent-core：`_resolve_input_topics` 增加 descriptor 反向传递；`_topic_clash` 在双方都声明 `priority` 时放行
5. 加 `control/*` 的 Dashboard 渲染器
6. 用一个假策略卡片（正弦轨迹，无模型）端到端验证：画布连线 → 协商 → 发布 → 驱动执行 → 拔网线后 watchdog 停机

**这一阶段不涉及任何 VLA 模型，但它是全部风险所在，也是唯一一段所有后续工作都依赖的。**

### Phase 1 — 卡片 + 远端 provider

7. 写 `actucore/plugins/vla.py`（第 3 节骨架）+ provider 目录发现与契约测试（§4.3.3）
7b. **π0.5 权重预取**：`gs://openpi-assets` 国内不可达，需一次性境外拉取 + 转存（§4.5.4）。先做，别等装服务器那天
8. `openpi_ws` adapter；局域网边缘服务器跑 π0.5，实测 RTT 分布（p50/p99，不要看均值）。server 起来先验证"预热完成前不报 ready"（§4.5.5）
9. 接 RTC，`inference_delay` 按实测配
10. 声明 `x-resource` / `x-completion` / `x-hooks`，验证"喊停能停"
11. `lerobot_async` adapter —— 一次接入换整个 policy zoo（§4.3.1）。用 `pi05` 与 8 的结果对照，确认移植版行为一致

### Phase 2 — 本地 provider 与广度

12. 把 `model_downloader` 提到 `phanthymotus/common/`，两镜像共用；SmolVLA 权重从 **ModelScope**（`lerobot/smolvla_base`）转存 COS 并 pin size/SHA256（§4.5.1 / §4.5.4）
13. Orin NX 上实测 SmolVLA（6.3 的实测项），含冷启动下载耗时与磁盘占用
14. 按结果决定 `LocalProvider` 是进程内还是独立子进程（§4.5.3 的显存归还是第二个判据）
15. 实现 fallback 的"安全收尾"语义
16. `unifolm_http`（默认关闭、`info()` 回显授权状态，权重走私有路径，见 §4.3.2 / §4.5.2）

### Phase 3 — 数据闭环与压测

17. 观测-动作对录制（LeRobot v2.1 格式；可直接用 `lerobot_record.py` 的格式定义，与训练侧闭环对齐）
18. ACP 进度通道，让 LLM 在长任务期间不再是瞎的
19. `openvla_http`（目标 OpenVLA-OFT），用它压测退化情形与 `unnorm_key` 必填规则

---

## 9. 待决问题

1. **`motus.control/1` 的 `mode` 枚举要覆盖到哪一步**？本文列了 `joint_position` / `joint_velocity` / `joint_torque` / `eef_pose` / `twist`。torque 和 eef_pose 涉及各家 SDK 差异很大的部分，第一版是否只做 `joint_position` + `twist`。
2. **仲裁的两层如何分工**：agent-core 的 ACP `x-resource` 锁（调用前，防止 LLM 和 VLA 同时启动）与驱动侧 mux（执行时，逐帧）。两者都要有——前者防冲突发生，后者兜住已经发生的。需要确认是否接受"两处都要维护优先级定义"的代价。
2b. **`priority` 的取值由谁定**：写死在卡片里、放 configSchema、还是由画布上的连线顺序决定。
3. **边缘服务器是哪台机器**，谁维护，断电/重启策略是什么。
4. **宇树模型是否去谈授权**。若谈成，G1 上的 embodiment 适配工作量会显著小于 π0.5。
5. **π0.6 / UnifoLM-WLA-Base 未放出**，不进本期计划，只做跟踪。

---

## 附录：外部参考

- openpi 远端推理：<https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md>
- RTC（Real-Time Chunking）：<https://arxiv.org/abs/2506.07339>，LeRobot 实现 <https://huggingface.co/docs/lerobot/rtc>
- LeRobot 异步推理：<https://huggingface.co/docs/lerobot/async>
- SmolVLA：<https://huggingface.co/blog/smolvla>
- Jetson AI Lab，π0.5 on Thor（TRT FP8/NVFP4，132 ms → 54 ms → 49 ms）：<https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/>
- openpi 边缘部署 issue：<https://github.com/Physical-Intelligence/openpi/issues/657>
- 宇树 UnifoLM-VLA-0：<https://github.com/unitreerobotics/unifolm-vla> ｜ WMA-0：<https://github.com/unitreerobotics/unifolm-world-model-action> ｜ WLA-1.0：<https://github.com/unitreerobotics/unifolm-wla>
