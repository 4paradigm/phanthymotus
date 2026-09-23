# ActuCore — 执行模型层

ActuCore 是 Perception 在执行侧的对称层。Perception 把原始数据流变成语义；ActuCore 把意图/目标变成运动指令。

```
Hardware → Driver·Sensor → Perception → Agent Loop → ActuCore → Driver·Actuator → Hardware
                                                     ↑ 这一层
```

执行模型（VLA 策略、导航、抓取策略、locomotion、whole-body control）以**卡片**的形式挂在这里，聚合成一个 MCP HTTP server，由 Agent Core 通过 MCP JSON-RPC 调用。

**当前有一张卡片：`vla`，默认开启。** `enabled` 只决定这张卡片出不出现在工具列表里，不决定它动不动 —— 真正的门槛在画布连线、协商和驱动侧的检查链，见下。

## `vla` 卡片

一张卡片 + 可插拔 provider，**不是每个模型一张卡**。两条轴刻意保持正交，因为换模型和换机器人是两件无关的事：

- **provider（换模型）** —— 从 `plugins/vla/providers/` **扫目录发现**，卡片里没有枚举。加一个后端就是加一个文件：暴露 `PROVIDER`，实现 `capabilities` / `infer` / `health` / `close` 四个方法，`configSchema` 的 enum 在运行时生成。
- **embodiment（换机器人）** —— 完全不配置。卡片驱动的是**画布上连到它输出口的那张驱动命令卡片**，和 `canvas_binding.py` 决定 agent 能够到哪些 MCP 是同一套逻辑。agent-core 在启动时向那张卡片要 action space，作为 `control_interface` 传进 `start`。没有 `embodiment: "g1_dex1"` 这种字符串可以写错，也不会把 URDF 误当成动作接口。

启动时**协商一次**：provider 的 `capabilities()` 和下游 descriptor 对账（动作维度、频率），对不上直接拒绝启动并说明是哪两个数字对不上——而不是启动后在 30 Hz 上一条条失败，那时候操作员看到的是一台停住的机器人和没有原因。

Provider 的组织方式是**非对称的**，而且是刻意的：

| provider | 说明 |
|---|---|
| `mock` | 正弦轨迹，无模型、无网络、无 GPU、无 torch。默认值 |
| `smolvla` | LeRobot SmolVLA，本机推理。**只有 JetPack 6.1 的镜像有**，见下 |
| `vla_cloud` | 任何跑在别处的模型。只配 `{endpoint, api_key, model}` |

**本地一个模型一个文件，按模型名命名** —— 和 `perception/plugins/` 一样（`asr.py`、`tts.py`、`vop.py` 各自管自己的权重、下载和加载）。一个笼统的 `local` 会变成一个按模型族分支的 switch，SmolVLA 的动作 padding、π0 的 JAX 栈、UnifoLM 的 flash-attn 构建全堆在它后面。共用的部分（发现、四方法契约、与机械臂的协商）在卡片和 `providers/__init__.py` 里。

**远端只有一个文件。** 模型跑在别处时，它自己的那些麻烦就不是机器人的事了：回来的是一个 action chunk，唯一变化的是地址。配置形状和 agent-core 配 LLM 完全一样——这个项目里 `config.main['client']['llm']` 就是一组 `{url, key, model}`，旁边没有一行 serving 代码。

### 选哪个 checkpoint

`provider` 选"哪个模型族、跑在哪"，`model_name` 选"具体哪一份权重"：

```yaml
provider: smolvla
model_name: smolvla_base     # 本机 provider 从 models: 里挑
models:
  smolvla_base:              # 上游原版，6 维（SO-100/SO-101）
    model_dir: /models/vla/smolvla_base
    weights: {base_url: [...], files: {...}}
    feature_map: {...}
  smolvla_tianyi:            # 为某台机器人微调过的，另起一份
    ...
```

公开发布的 checkpoint 沿用**上游原名**；为某台机器人微调过的用**带机器人名的名字**（`smolvla_tianyi`、`smolvla_q5`）——因为"它适配的动作空间"正是操作员必须搞对的东西，而名字是唯一会被读到的地方。

选了一个没注册过的名字会**直接拒绝并列出已暂存的**，不会退回默认值：悄悄加载另一份 checkpoint 会得到一个能跑、会动、但是错的策略，正是整条协商链路存在的理由。

表单里 `endpoint` / `api_key` / `timeout_ms` **只在 `provider: vla_cloud` 时出现**（`x-show-when`）。本机 provider 旁边填着一个 endpoint 看起来像配好了，实际被忽略，没有任何东西会说明这件事。

需要说清楚的一个后果：`vla_cloud` 说的是**我们自己的 `motus.vla/1`**，不是 openpi 的 msgpack-over-WebSocket，也不是 LeRobot 的 gRPC。指向一个原始的上游服务器不会work——翻译属于服务端（`phanthymotus-cloud`），那里本来就住着吞吐和扩缩容的问题。这和 OpenAI 生态的分工是同一个：spec 是文档，vLLM 和 SGLang 各自实现，客户端不背每种服务器一个适配器。

`smolvla` 的三条规矩都在 `providers/smolvla.py` 里：**懒 import**（torch/lerobot 在用到它们的函数里才 import，所以没装 lerobot 的镜像照常启动、照常提供 `mock` 和 `vla_cloud`）、**懒下载**（COS + size/sha256 pin，复用 perception 的 `model_downloader`，不重写）、**懒加载且不占调用线程**（`__init__` 只读 checkpoint 的 config —— 便宜，且足够回答 `capabilities()` 让卡片先完成协商 —— 权重在后台线程加载，期间 `health()` 为 False，卡片报 `loading` 而不是 ready）。

有一件事它替你做不了，而且值得把话说准：**限制来自 checkpoint，不是架构。**

SmolVLA 的天花板是 `max_action_dim: 32` —— 训练时把动作补到 32 过投影层，推理时按 `action_feature.shape[0]` 裁回数据集的维度。维度在微调时**从 LeRobot 数据集自动推断**，投影层跟着 resize，不用手改配置（只有 DoF > 32 才要动架构）。天轶的 26 维落在 32 以内，所以这个模型族对天轶是可行的目标。

不行的是**这个 checkpoint**：`smolvla_base` 在 SO-100/SO-101 上预训练，它的动作头只学过六个槽，归一化统计量也是那条臂的。指到 26 维上它会**吐出数字而不是报错**——那比拒绝更糟，也正是协商比对的是 checkpoint 的宽度而不是网络宽度的原因。

路径是**在天轶数据上微调**，不是改配置。动手前值得知道：LeRobot 的 issue 区里有微调 loss 收敛、曲线正常、评测成功率却是 0% 的案例，原因是 state/action 布局和录制时对不上。

`mock` 的用途不是演示，是**在接任何模型之前验证整条通路**——画布连线、协商、消息构造、驱动侧检查链、watchdog、拔网线。它按各关节半行程的比例构造，因此**在结构上就出不了限位**，默认幅度很小。

安全上这张卡片自己只做很少的事：

- **不自启动。** bundle 生命周期的 `start()` 故意什么都不做，策略不能因为容器重启就继续跑。
- **不声明 `x-completion`。** 策略跑到被停为止，没有"完成"；声明了会让一个 ACP pending 挂住整张卡片的生命周期，把其它所有 actuator 堵在 barrier 后面。
- **`x-resource` + 两个 interrupt hook** 都指向 `stop`。
- 推理失败时**发布空**而不是重发上一条：接收端的 watchdog 会让机械臂停住，而重发会让它继续按一个已经不在跑的策略动。

真正保证安全的东西在驱动侧 —— `common/control.ControlSink` 逐条检查、静默即 hold、受力即停。这个分工是刻意的：actucore 这个进程可能崩溃、被 OOM kill、或者丢掉网络，这三种情况都不能是"机器人靠它停下来"。

协议见 `phanthymotus-driver/README_dev.md` §「Continuous Control」，设计见 `docs/vla-integration.md`。

```bash
cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q
```

| | |
|---|---|
| MCP HTTP | `http://<host>:15730/mcp` |
| SSE（ACP 完成事件） | `http://<host>:15730/sse` |
| 容器 | `embodied-actucore` |
| 镜像 | `<registry>/<namespace>/actucore:<tag>` |
| 注册类别 | `actucore` |
| `serverInfo.name` | `actucore-bundle` |

端口只有一个：ActuCore 没有 Perception 那种音频流场景，所以不开 WebSocket（15731 预留）。

## `navi` 卡片

看到目标就朝它走过去。接 `vop` 的检测结果和一路深度图，输出 `motus.control/1`
的 **twist**（底盘速度）到一张驱动的底盘命令卡片。

```
camera ─┬─→ vop ────────────(data/json 检测+bbox)──→┐
        │                                           ├─→ navi ─(control/velocity)─→ 驱动的底盘命令卡片
        └─→ visual_depth ──(image/depth-zlib)──────→┤
                                                    │
           驱动的 loco_state ──(state/odom，可选)───→┘
```

`navigate_to(target="椅子")` 之后：朝目标走 + 同时转向对正 → 按深度三段减速、
必要时侧向让开 → 到 `stop_distance_m` 停下并推 ACP 完成事件。

**这是反应式视觉伺服，不是导航栈。** 没有地图、没有全局路径、绕不过 U 形障碍，
也不会后退（深度图对身后一无所知）。要建图规划是另一条路线。

### vx / vy / wz 是同一拍发出去的

第一版是串行的：目标不在画面中央就 `vx = 0`，先转正再走。对能横移的底盘没有
理由这么做 —— 接近速度按目标方位角（`half_fov_rad`）拆成 vx 和 vy，机器人**一边
转向一边沿直线走向目标**。去掉的正是「停—转—走」那套步态，它在 r1_sz 上的表现
就是每次修正方位都顿一下。

不能横移的底盘配 `align_min_scale: 0` + `use_lateral: false` 即可退回原来的走法；
而如果驱动的 descriptor 把 `vy` 钉成 `lower == upper == 0`，`adopt_limits` 会自己
读出来并关掉这个轴，不需要额外配置。

横移是有门槛的：深度图朝前，侧面正是它最不了解的方向，所以要迈的那一侧的深度段
必须**已知且空旷**——未知直接不迈，而不是默认放行。

### 目标跟踪：一帧起步、十帧放弃，方向是反的

第一版用**最新一帧**挑目标：置信度过 0.35 就开始追，丢 10 帧才放弃。于是一帧
假阳性就足以让人形机器人往前走 —— 而且那一帧还顺便定死了后续搜索往哪边扫，因为
搜索方向取自「最后看见它在哪一侧」。真机上的表现是「先转一下、往前挪一点、然后
才开始逆时针扫描」，三段全部来自同一帧垃圾检测。

现在走 `plugins/navi/track.py`：一条轨迹，带生命周期。

| 态 | 条件 | 驱动底盘？ |
|---|---|---|
| tentative | 最近 `confirm_window`(5) 帧里出现 `confirm_hits`(3) 帧 | **否** |
| confirmed | 达标 | 是 |
| coasting | 本帧没看到，按自身运动推算 | 是，上限 `max_coast_s`(1.2s) |
| reacquiring | 滑行中有东西落进门内，但还不够信 | 是，仍按预测走 |
| lost | 超时或不确定度越门 | 否，转入搜索 |

**复活一条轨迹和新建一条一样要 `confirm_hits` 帧。** 原先只要一帧，而那一帧恰恰
出现在门开得最大的时候 —— 滑行期间协方差一直在长，1.2 s 之后门的半径到了 3.7 m。
r1_sz 上一条追人的轨迹就这样被重捕到一个灭火器/路锥上（vop 有一帧把它标成了
`person`），卡片随即报告「已到达」。门另外还被一条物理约束封顶：目标最快也就
`max_target_speed`，自运动已经在预测里了，所以残差是目标自己走的距离 —— 四米外的
人不会突然出现在手边。

三条思路都是抄的，出处写在 `track.py` 的模块注释里：SORT 的生命周期、ByteTrack
的低置信度第二轮关联（`sustain_confidence` 以上的检测不能**新建**轨迹，但可以
**维持**一条已有的，前提是落在门内）、OC-SORT 的重捕回溯（目标重新出现时，速度
从遮挡两端的两次真实观测重建，而不是沿用滑行期间一直在累积误差的那个）。

**状态是米制的、机体系的，不是图像框。** 这是没有直接用现成跟踪库的原因：它们的
卡尔曼状态是像素里的 (x, y, 宽高比, 高)，我们的深度没有地方放；而遮挡期间预测框
位置上的深度读到的是遮挡物，所以距离无论如何得单独外推 —— 于是会变成一个像素滤
波器和一个米制滤波器估同一个物体、且可以互相矛盾。

**自身旋转才是目标在画面里移动的主因。** 死区让 `wz` 只有 0 和 ≥1.0 两档，换算
成 63° 视场下的 640 px 就是约 58 px/帧的位移瞬间出现或消失，图像空间的恒速模型
预见不到。而 BoT-SORT 用光流去估的那个量，`motus.odom/1` 直接报给我们了 —— 所以
卡片现在读 odom 的 `vx/vy/wz` 三个轴，不再只读 `vx`。没接 odom 时退回用**指令**
推算，同时放大过程噪声：指令不是测量（dry_run、姿态被拒、死区归零都会让两者对不
上），这时滤波器应该更信它看见的。

### 避障走廊是米制的，不是画面的三等分

画面的固定扇区在不同距离覆盖**不同的实际宽度**。63° 镜头下中间那份是
`±0.204 × 距离`：

| 距离 | 中间带覆盖半宽 | R1 半宽（357 mm 整宽） |
|---|---|---|
| 0.8 m（`obstacle_stop_m`） | ±0.16 m | ±0.18 m ❌ |
| 1.1 m | ±0.22 m | ±0.18 m ⟵ 交叉点 |
| 2.0 m | ±0.41 m | ±0.18 m（白白减速） |

**距离越近覆盖越窄，而近处才是要停的地方。** 偏轴 0.25 m 的门框会被归到「左」，
而左带从来不停止前进 —— 肩膀撞上去，深度图却报告前方通畅。

现在 `depth.corridor()` 逐像素按自身深度算横向偏移，只保留落在机器人宽度里的，
取 5 分位。半宽由底盘在 descriptor 的 `footprint` 里声明（和 `min_magnitude` 同
一个套路：机器人知道自己多宽，策略不该写死），`adopt_limits` 读走；没有声明就按
保守值走并在 `degraded` 里明说 —— 这个数所有的出错方向都是单边的，以为自己更宽只
是多减速，以为自己更窄就是把肩膀送进门框。

声明的是**手臂放下时的静态外廓**，所以卡片自己再加 `clearance_margin_m`：真正会
撞的是摆动的手臂和迈出去的腿。

### 「不知道」不等于「没东西」

深度图有洞，而造成洞的东西 —— 椅子腿、桌沿、玻璃 —— 正是会挂到肩膀的那些。无效
像素会悄悄从最小值里掉出去，所以**满是空洞的走廊和空旷的走廊给出同一个数**。

`corridor()` 因此同时返回 `coverage`：走廊在 `obstacle_stop_m` 处投影到画面上的那
块区域里，有多少像素带真实读数。低于 `coverage_min` 就不往前走并说明原因，介于
`coverage_min` 和 `coverage_full` 之间按比例减速。分母算的是走廊的**完整角度范
围**，包括落在镜头之外的部分 —— 近处机器人比视场还宽（0.5 m 处 ±0.33 m 的走廊约
67°，63° 的镜头够不到两边），只数可见列会报出一个「全部测到了」而其实有三分之一
的车宽从没进过画面。

横移的判据也换成了同一个东西：问的不是「画面右三分之一空不空」（那描述的是前方偏
右两米处），而是**要迈进去的那条走廊**测到了没有、空不空。落在镜头外的走廊 coverage
为 0，因此自动被拒 —— 不需要特判。

### 机器人动不了那么慢，这件事决定了上面所有东西

足式底盘要凑一整个步态周期才能动，所以低于某个速度它根本不动：R1 是 0.4 m/s 和
1.0 rad/s，低于这个值 SDK 收下指令、返回 0、机器人站着不动。于是比例控制**没法慢慢
修小误差，只能短促地修**。每个轴因此都是一个带迟滞的开关（`Gate`），一旦打开，幅值
被抬到至少地板值（`_lift`）。地板值来自下游 descriptor 的 `limits.min_magnitude`，
不是这里写死的常数。

由此而来的两个坑，都真实发生过：

| 症状 | 原因 |
|---|---|
| 转向一顿一顿 | `wz_max` 0.8 低于 1.0 的地板，策略**整个输出范围都执行不了**，每条转向指令都被吸附成 0 或 ±1.0 —— 一个 10 Hz 方波。`adopt_limits` 现在在启动时把上限抬离地板，并把调整写进 `info().degraded` |
| 走到目标前 0.67 m 就停住，然后按「长时间没有进展」失败 | `k_fwd * (d - stop)` 早在到达之前就掉到地板以下。前进轴的 Gate 现在会一直保持地板速度，直到真的进入停止距离 |
| 明明还离得远，却报告「已到达」 | 到达判定排在避障之前，于是**挡在路上的东西可以冒充目标**。见下 |
| **走到门旁边特地转向门、然后肩膀刮到门框** | 避让时「往哪边躲」是按**角度三等分**选的 —— 而门洞恰恰是这个判据最坏的情况：能看穿过去，所以它在三等分里读数**最远**，于是被判成「更开阔的一侧」，机器人朝唯一够得着的东西扑过去。而且是 `wz_max` 全速偏航 + `vy_max` 全速横移，且转身这一项**完全没有把关**（只有 `vy` 过 `_may_strafe`）。现在两侧都按**米制偏移走廊**判，两侧都不可用就停下不转 |
| **过不了门** —— 走到门口就转向，接着原地转身 | 视场角配小了。走廊是米制的，每一帧要把半宽反算成画面上的像素列，这一步用 `half_fov_rad`。r1_sz 上配的 0.55 对实测 0.888，切片偏宽 1.6 倍，**走廊 1.86 m 宽、比任何一扇门都宽** → 门框永远算在正前方 → 离门 0.6 m 触发 `obstacle_stop` 转向旁边 → 人出画、轨迹丢失 → 搜索扫描（那个"转身"）。**而深度图全程报告前方通畅。** 现在这个数由相机声明，见下 |

### 「到达」和「被挡住」是两个结论，不许互相冒充

用户在 r1_sz 上看到的：人还在几米外，机器人在一个障碍物前停下，然后报告到达。
两处成因，都改了：

1. **到达必须建立在活的观测上。** 轨迹处于滑行/重捕状态时位置是推算出来的 ——
   推算足以支撑「继续走」，不足以支撑「走完了」。现在滑行中不会判到达，而是
   落到避障逻辑，由它说出前面到底是什么。
2. **挡住是一个结果，不是一种情绪。** `_avoid` 会先尝试绕 —— 转向更开阔的一侧
   并朝那一侧横移。但转向算运动，所以 `idle_timeout_s` 永远抓不到一台原地挪了
   十二秒的机器人，调用方什么都得不到。`blocked_timeout_s`(12s) 之后任务以
   「前方被挡住了 N 秒，绕不过去：<具体原因>」失败。

同一件事在驱动侧也有一条：**步长钳不能比同一轴的死区更细**，否则起步的头几拍全在
机器人执行不了的区间里。见 `phanthymotus-driver/README_dev.md` §
`limits.min_magnitude`。

### 相机参数不是配置项，是相机声明的

`half_fov_rad` 曾经是这张卡片配置里的一个数，要人手填。它是**相机**的属性，而填的人
没有任何办法发现自己填错了 —— r1_sz 上那个 0.55 前一天就已经量成 0.888，结论进了报告
没进配置，没有任何东西报警，症状是机器人过不了门（见上面症状表）。

现在走 `motus.camera/1`：相机卡片在 `info()` 里声明，agent-core 在 `start` 时按连线交给
下游，这张卡片用 `policy._adopt_camera` 采纳。完整规范在
`phanthymotus-driver/README_dev.md` § Camera Parameters，消费侧实现是
`plugins/navi/camera.py`。

和 `min_magnitude`（死区）、`footprint`（肩宽）完全同一条规矩：**知道的那一方声明，
策略启动时采纳，采纳了什么写进 `info().degraded`。** 相机参数是最后一个例外。

三件跟着来的事：

- **没声明不是错误。** 仓库里几乎没有卡片听说过这个格式，兜底值照旧生效，只是会在
  `degraded` 里说出症状（「如果机器人在门口反复转向……」），省掉下一个人重查一遍。
- **声明里没有视场角 ≠ 视场角是 0。** 相机可以诚实地报 `null`；这时也用兜底值，并提示
  用 `tools/measure_fov.py` 量一次、填到**那颗镜头的驱动声明**里，而不是填回这张卡片。
- **两路输入必须来自同一颗镜头，现在会检查了。** vop 报的是归一化横向偏移，深度图是
  一格格的距离，这张卡片用**同一个**视场角把两者都换成米 —— 只有同源才成立。而输入是
  按载荷内容绑定的（刻意如此），所以把 A 相机的 vop 接到 B 相机的深度上一直是接得通的，
  会算出看起来合理但错的距离，且任何日志里都没有痕迹。两边都声明 `id` 之后，这就是
  一次比较。原来的 navi 计划承诺过这条检查，从没实现。

### 第二个输出口：它看到了什么、决定了什么

`image/jpeg`，默认 `/actucore/navi/view`，5 Hz。画在深度图上：**走廊**（画在它当前
读到的那个距离上）、**目标框与距离**、**三个轴的指令箭头**。

存在的理由不是好看。这张卡片至今每一个 bug 都是同一个形状 —— **数字看着都合理，机器人
做了别的事**：视场角把走廊撑到 1.86 m、角度三等分读的是镜筒而不是房间、横移被死区放大
9 倍。三次都在 `info()` 里看不出来，而站在机器人旁边看一眼就明白。

三个设计点：

- **「没有读数」画成平灰，不上色带。** 自信地错和读不到，在任何色带里长得一模一样，而
  分开这两者正是镜筒遮罩存在的目的 —— 灰色就是那个遮罩在工作的样子。
- **走廊画在它当前清到的那个距离上**，不是固定一对线。走廊是米制的，同样的半宽在不同
  距离覆盖画面的不同比例，这恰恰是最久才变得可见的那个事实。`corridor_edges()` 是纯函数，
  和决策用的是同一套米→像素映射（测试里钉住了，因为画一张和决策不一致的图比不画更糟：
  它会替机器人做过的任何事背书）。
- **给人看的图不允许把机器人弄停。** 渲染整段被包住、单独限流，出错就关掉这个口并记一
  条日志，指令流照跑。

文字是 ASCII —— `cv2.putText` 只有 Hershey 字体、没有中文，中文原因会渲染成方块，所以
它留在 `info().last.reason` 里。

### 启动门槛

和 `vla` 一样，`enabled` 只决定它出不出现在侧边栏里。真正的门槛：

1. 输出必须连到一张驱动的底盘命令卡片（`control/velocity` 端口），否则拿不到
   下游 descriptor，直接拒绝启动；
2. 输入必须有 vop 的检测结果，以及深度图或深度摘要中的**至少一路**；
3. 协商不过就拒绝 —— 这张卡片只产生 `twist`，接到关节卡片上维度可能恰好也对，
   靠 `mode` 拦下来；
4. 每条指令还要过驱动侧 `ControlSink` 的检查链。

### 两档降级，`info()` 里看得见

少接一根线不会让卡片起不来，但会让它变笨，而**变笨和坏掉从下游看是一样的**。
所以 `info().degraded` 会明说当前在哪一档：

| 少了什么 | 后果 |
|---|---|
| 只有深度摘要，没有深度图 | 目标距离退化成「它所在的那三分之一画面里最近的东西」，那是最近的**障碍**而不是目标。精度明显变差，且障碍进入停止距离时会被当成「到达」 |
| 没接 `state/odom` | 没有卡死检测。撞上东西不会自己停 —— 而单目深度贴近平面墙时恰恰最不可靠，这是它最需要兜底的场景 |
| vop 没开 `publish_bbox` | 距离只能在目标中心取一小块，而不是整个目标区域取分位 |
| 相机没声明 `camera_info` | 视场角退回卡片里的兜底值，**避障走廊的宽度按它算**。兜底值刻意偏小 —— 偏小则走廊偏宽，机器人拒绝走得过的缝（卡住，但不撞）；偏大则走廊偏窄，肩膀撞门框。卡住能恢复，撞了不能 |

### 最重要的一条性质

**说不清楚的时候什么都不发。** 观测过期、深度解不开、目标丢太久、判定卡死、
已到达 —— 全都返回 `values=None`，卡片一条指令都不发，由驱动侧 watchdog 在
`watchdog_ms` 内把底盘停住。

刻意不发显式的零：零同样能停住机器人，但它会**继续喂饱 watchdog**，于是一个
已经死掉的策略会留下一台「自以为正在被驱动」的机器人。沉默是坏掉的上游唯一
伪造不了的信号。

（唯一的例外是「到达」：那里会先发一帧显式的零再安静下来。到达是成功，应该停
在一条指令上而不是停在超时上，否则驱动日志里看着像链路断了。）

### 标定工具（`tools/`）

走廊的几何建立在两个从来没在这台机器人上量过的量上，两个脚本各管一个。**两个都
只读传感器，不发任何指令**，所以在站着、躺着、或者正在干别的的机器人上跑都安全
（机器人由人来开）。它们随镜像发布，不用 `docker cp` —— 热拷进去的文件会一直和镜
像发散，而一次出处不明的测量值不了多少钱。

```bash
docker exec -it phanthy-motus-actucore-1 python3 /work/tools/measure_fov.py wall
docker exec -it phanthy-motus-actucore-1 python3 /work/tools/measure_fov.py object --width 0.60 --target box
docker exec -it phanthy-motus-actucore-1 python3 /work/tools/measure_odom_drift.py static --seconds 60
docker exec -it phanthy-motus-actucore-1 python3 /work/tools/measure_odom_drift.py landmark --target chair --seconds 120
```

**`measure_fov.py`** 回答两个问题，不只一个。`half_fov_rad` 是个配置值不是读数，
而整条走廊的横向换算都按它的 `tan` 缩放；同时 `corridor` 算的是
`lateral = tan(θ) × depth`，这在深度是**到成像平面的垂直距离**（z-depth）时才对，
如果深度是**沿射线的距离**（range）就该用 `sin(θ)` —— 31° 处差 17%，正好落在肩膀
所在的角度上。

`wall` 方法两个问题一起回答，而且不需要卷尺：正对一面平墙，z-depth 读出来是平的，
range 读出来是 `d0/cos θ`。形状本身就是答案，而且如果是 range，曲线的陡峭程度直接
给出 `half_fov`。**两种都拟合不上**也是一个结论 —— 那说明这个深度源不满足针孔几何，
换多少 `half_fov_rad` 都救不回来。

`object` 方法需要一把卷尺和一个已知宽度的矩形目标，用两条边（不是半宽）算，所以目
标不必摆在画面正中。建议在两个距离各测一次：结果差超过几个百分点，出问题的就不是
视场角而是镜头畸变或深度标定。

**`measure_odom_drift.py`** 量的不是抽象的位姿漂移，而是**跟踪器真正消费的那个量**：
滑行 T 秒之后，一个静止参照物的预测方位和实测方位差多少。而且是**和「什么都不做」
对比着量** —— 如果用里程计补偿并不比假设世界没动更准，那补偿就没有挣到它的复杂度，
局部记忆栅格（`docs/visual-navigation.md` 取舍八）也就不用考虑了。这个比值是结论，
不是原始误差。

推算那一步直接调 `track.Tracker._predict`，不另写一份：要量的是**跑在机器人上的那
个变换**，第二份拷贝可能是对的而线上那份是错的。

`static` 模式最便宜：机器人站着不动，凡是累积出来的都是偏置。足式速度估计的偏置通
常是主导项，而且它会稳定地朝一个方向攒。

### 文件

| 文件 | 内容 |
|---|---|
| `plugin.py` | 卡片：工具声明、生命周期、订阅、定时发布 |
| `policy.py` | 控制律。纯函数，无 ROS —— 所有行为分支都在这里 |
| `depth.py` | 深度解码与采样。纯函数，无 ROS |
| `track.py` | 单目标跟踪：生命周期、自运动补偿、遮挡外推。纯函数，无 ROS |
| `odom.py` | `motus.odom/1` 的读取侧。**故意不跨仓库 import** —— 协议是文档不是共享库，两边各自实现、各自跑契约测试 |
| `../tools/` | 标定脚本，见上。只读传感器 |

`policy.py` 和 `depth.py` 不碰 ROS 是有意的：这样「目标被人挡住的同时左边还有
个障碍会怎样」是一条测试，而不是一下午的真机调试。

## 构建与运行

只有 Jetson GPU 版 —— 执行模型（VLA、抓取策略、locomotion）都要 GPU，没有 CPU 变体。

```bash
./deploy/build_actucore.sh                    # JetPack 5.11（默认，与 build_perception.sh 一致）
./deploy/build_actucore.sh --jp-version 6.1   # JetPack 6.1，带本机推理
./deploy/build_actucore.sh --mirror tuna      # 指定 pip / apt 源
```

**同一份 Dockerfile，两个 base**，应用层逐字节一样：

| | base | 可用 provider | 大小 |
|---|---|---|---|
| jp5.11（默认） | `jetson-base`（共享的那个） | `mock` + `vla_cloud` | ~13.8 GB |
| jp6.1 | `jetson-base-actucore`（CUDA torch 2.9 + lerobot） | 全部 | ~18.6 GB |

这个差别是**被迫的，不是取舍**：jp5.11 是 CUDA 11.4，而 lerobot 要 `torch >= 2.2.1`，PyTorch 官方矩阵里 torch 2.2 的最低 CUDA 是 11.8 —— 那条线上**不可能**有本机推理。`smolvla` 在那里会在 start 时直接拒绝并说明原因。完整调研见 `deploy/prepare_actucore_base.sh`。

jp6.1 的 base 由 `deploy/prepare_actucore_base.sh` 构建（只支持 6.1）。加本机模型卡片时，如果它的依赖不在 base 里，放在它自己的 `RUN` 层，不要预装在共享基础层里。

部署走 Dashboard 的服务部署页，或直接把 `deploy/service.yml` 合并进 `/opt/phanthy-motus/docker-compose.yml`（Agent Core 会从镜像里抽这个片段，见 `agent-core/src/api/drivers.py`）。

## 卡片契约

卡片是 duck typing 的，没有基类、没有 ABC、没有注册装饰器。一个卡片是 `plugins/<name>.py`，或者 `plugins/<name>/` 包（`__init__.py` 里 re-export 类）。

### 必需成员

| 成员 | 签名 | 说明 |
|---|---|---|
| `PREFIX` | 类属性 `str` | 工具名前缀，也是 dispatch 的路由键 |
| `__init__` | `(self, plugin_cfg: dict, executor)` | 需要 ROS 命名空间的卡片用 `(self, plugin_cfg, namespace, executor)` |
| `get_tools()` | `-> list[dict]` | 返回工具 metadata 列表 |
| `dispatch(name, args)` | `-> dict \| None` | 返回 `None` 会让 MCP 报 `-32601` |

可选：`start()` / `stop()` —— 只有在 `main.py` 的注册块里显式调用才会被执行。

### 四个必须知道的坑

1. **`PREFIX` 不能含下划线。** `dispatch()` 用 `full_name.partition("_")` 拆前缀，所以 `PREFIX = "vla"` 可以，`PREFIX = "grasp_policy"` 永远匹配不上。
2. **`inputSchema.properties.action.enum` 必须包含 `"info"`。** Agent Core 靠它探活 —— 它会挑出 action enum 里有 `info` 的工具，用 `{"action": "info"}` 调一次（`agent-core/src/api/mcp_manage.py`）。没有 `info` 的卡片会一直是离线状态。
3. **`dispatch()` 必须返回 plain dict**，例如 `{"state": "running"}`。MCP HTTP handler 会自己包成 JSON-RPC 的 content 格式。**不要**返回已经包好的 `[{"type": "text", ...}]`，那会二次编码并让前端解析失败。
4. **`x-completion` / `x-hooks` 放在 `inputSchema` 里面**，不是工具顶层。

### 工具 metadata

```python
TOOLS = [
    {
        "name": "vla",                # 等于 PREFIX 时不加前缀，否则暴露为 "{PREFIX}_{name}"
        "type": "processor",          # sensor | actuator | processor | resource
        "multiInstance": False,       # True = 每个输入 topic 一张卡片 / 一个 ROS 节点
        "description": "…",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "stop", "info", "config"]},
                "input_topic": {"type": "string"},
            },
            "required": ["action"],
            # 可选：长时动作声明 ACP 完成回调
            # "x-completion": {"actions": ["goto"], "timeout": 120},
            # 可选：系统 hook 绑定（打断等）
            # "x-hooks": {"on_interrupt_goto": {"action": "cancel"}},
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "model_dir": {"type": "string", "default": "/models/vla", "scope": "shared"},
                "speed":     {"type": "number", "default": 0.5, "scope": "instance"},
            },
            "required": [],
        },
        "topic_in":  [{"format": "data/json",  "desc": "goal"}],
        "topic_out": [{"format": "control/velocity", "desc": "motion command"}],
    }
]
```

**`type` 的含义** —— 它决定 Agent Core 怎么调度这个工具：`sensor` 连续调用会被批量并行；`actuator` 和 `processor` 要过 ACP barrier，dispatch 前会等所有 pending 动作完成；`resource` 是静态资源（如 URDF）。没声明 `type` 的工具默认按需要 barrier 处理（安全侧）。判定逻辑在 `agent-core/src/event/llm.py` 的 `_needs_barrier()`。

**`configSchema` 的 `scope`** —— `shared` 是整个卡片共享一份配置，`instance` 是每张画布卡片一份。

**`topic_out[0].format` 决定 Dashboard 用哪个渲染器**（波形 / 视频 / 点云 / KV 面板……）。格式清单见 `phanthymotus-driver/README_dev.md` 的 "Data Format & Dashboard Rendering"。执行侧常用 `control/velocity`、`control/joint`、`data/json`。

## 加一张卡片

1. 写 `plugins/<name>.py`（或 `plugins/<name>/` 包），实现上面四个必需成员
2. 在 `config.yaml` 的 `plugins` 下加 `<name>: {enabled: true, ...}`
3. 在 `main.py` 的 `ActuCoreBundle.__init__` 注册块里加一个 if 分支：
   ```python
   if plugins_cfg.get("<name>", {}).get("enabled", False):
       from plugins.<name> import XPlugin
       self._plugins.append(XPlugin(plugins_cfg["<name>"], executor))
       log.info("XPlugin loaded")
   ```
4. 该卡片需要的依赖加到 `Dockerfile`（以及 `Dockerfile.jetson`，如果要跑 GPU）
5. 重建镜像、重新部署，确认 Dashboard 侧边栏「执行」分区里出现了它

需要 ROS 命名空间的卡片（topic 里要带机器人名）多一步：namespace 为空时用 hostname 兜底，写法参照 `perception/main.py` 里 vop 的注册块。

完整的、带 ROS 节点的卡片实现可以直接看 `perception/plugins/vop.py` —— 它是最干净的范例。

### 要发 `motus.control/1` 指令流的卡片

别自己拼消息、也别自己写协商 —— 用 `plugins/control_stream/`：

```python
from ..control_stream import build as build_message
from ..control_stream import negotiate

problems = negotiate.check(self._capabilities(), descriptor, label="策略")
if problems:
    return self._error("与下游动作空间不匹配：" + "；".join(problems))
rate = negotiate.effective_rate(capabilities, descriptor, self._cfg.get("rate_hz"))
ttl  = negotiate.ttl_ms(rate, descriptor)
```

`label` 是报错里对上游的称呼，默认「模型」。导航策略不是模型，告诉它的操作者
「模型输出 6 维」会把人送去找一个不存在的 checkpoint。

这两个模块原本在 `plugins/vla/` 下，第二个消费者出现时提到了 `control_stream/`；
`plugins/vla/` 留了两个 re-export shim，所以照旧从那里 import 的代码不用改。
