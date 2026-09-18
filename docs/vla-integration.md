# VLA 接入方案

执行模型（VLA 策略、导航、抓取、locomotion）怎么接进这套系统：架构长什么样、接一个
新模型要做什么、接一台新机器人要做什么，以及**对 servo 卡片的要求**——包括现在必须
满足的和将来会加的。

代码里引用这份文档的地方：`phanthymotus-driver/README_dev.md` §Continuous Control、
`actucore/plugins/vla/negotiate.py`、`actucore/plugins/vla/message.py`。

> **这份文档描述已经存在的东西。** 2026-09 有过一版同名方案，前提是「actucore 是空的、
> 从它到电机的路不存在」——那一版已作废（PR #220），因为两个前提都不成立了。下面凡是
> 写「已实现」的，都能在 main 上找到对应文件；没实现的单独列在最后一节。

## 现状

| 环节 | 状态 | 在哪 |
|---|---|---|
| `vla` 卡片（协商、分块下发、暂停/打断、watchdog 配合） | ✅ | `actucore/plugins/vla/plugin.py` |
| provider 协议 + 三个实现（`mock` / `smolvla` / `vla_cloud`） | ✅ | `actucore/plugins/vla/providers/` |
| 启动协商 | ✅ | `actucore/plugins/vla/negotiate.py` |
| `motus.control/1` 消息与 descriptor | ✅ | `message.py` + `phanthymotus-driver/common/control/` |
| 驱动侧安全链（新鲜度、仲裁、限幅、限位、watchdog） | ✅ | `phanthymotus-driver/common/control/sink.py` |
| servo 卡片实例 | ✅ 2 台 | RealMan RM75、天轶 2.0 |
| 权重下载（size+sha256 pin、多源择优、进度上报） | ✅ | `perception/utils/model_downloader.py` |
| 云端推理服务 | ❌ 未实现 | `phanthymotus-cloud` 仍是空仓库 |
| 碰撞再校验、动作块连续性 | ❌ 未实现 | 见「已知缺口」 |

## 架构

一条从像素到电机的链路，跨三个仓库：

```
 相机/本体状态            actucore                      phanthymotus-driver
 ┌──────────┐      ┌────────────────────┐        ┌──────────────────────────┐
 │ image/*  ├─────▶│  vla 卡片           │        │  servo 卡片               │
 │ state/*  │      │   ├ 协商(一次)      │        │   ├ descriptor (info)     │
 └──────────┘      │   ├ provider.infer  │        │   ├ ControlSink (每条)    │
                   │   └ 按 rate 拆块下发 ├───────▶│   └ 厂商 SDK              │
                   └─────────┬──────────┘ control/│                          │
                             │            joint   └──────────────────────────┘
                   ┌─────────▼──────────┐   (DDS)
                   │ provider           │
                   │  mock / smolvla    │  ← 本机推理，权重走 model_downloader
                   │  vla_cloud         │  ← 远端推理，只有 {endpoint,key,model}
                   └────────────────────┘
```

两条轴**刻意正交**——换模型和换机器人是两件无关的事：

- **换模型 = 换 provider。** 目录扫描发现，卡片里没有枚举。
- **换机器人 = 换连线。** 卡片驱动的是画布上连到它输出口的那张驱动卡片，启动时向它要
  action space。没有 `embodiment: "g1_dex1"` 这种能写错的字符串。

**控制平面与数据平面分开**：`tools/call` 是低频、请求/应答、要授权的控制平面，适合「LLM
决定把手臂移到某处」；执行模型每秒产生几十条命令、每条都不是一个问题，所以走数据平面的
`control/*` DDS 话题。和扬声器的 `topic_in: audio/pcm-16k` 与它的 start/stop 分开是同一个
道理。

## 通用控制接口 `motus.control/1`

**权威定义在 `phanthymotus-driver/README_dev.md` §"Continuous Control"**，那里有完整字段
表。这里只讲为什么是这个形状。

分两半：**descriptor** 是驱动在 `info()` 里声明的「我接受什么」，**message** 是话题上流过
的一条命令。两者在每条命令上对账。

**URDF 不是 descriptor，也替代不了它。** URDF 没有单位、没有控制频率、没有说明接受绝对
还是增量位置、没有归一化范围。它作为 FK 和碰撞几何的补充从 `urdf_ref` 引用，但动作接口不
能从它推导。

**`force_torque` 必须出现，哪怕是 `null`。** `parse_descriptor` 会拒绝省略它的 descriptor
——省略正是一台机器人被误以为有力保护的方式。

### 协商发生在 start，不是每条命令

`negotiate.check()` 把 provider 的 `capabilities()` 和下游 descriptor 对账：动作维度、频率
上限、chunk 大小。对不上就拒绝启动，并说明是**哪两个数字**对不上（「模型输出 32 维动作，
下游只接受 14 维」）。

替代方案是启动后在 30 Hz 上一条条失败——那时机械臂已经动了，或者驱动在全量拒绝而操作员
看到的是一台停住的机器人和没有原因。所有问题**一次性收齐**返回，而不是修一个重启一次再
发现下一个。

### 频率与 ttl 的由来

`effective_rate()` 取三者最小：卡片配置的偏好、模型的 `control_hz`、驱动的
`rate.expected_hz`，再被 `rate.max_hz` 夹一次。**比模型自己的频率发得更快不会让机械臂更
平滑**，只会让每条动作的含义和它被生成时略有出入。

`ttl_ms()` = 两个周期，下限 50 ms，上限是驱动的 `watchdog_ms`。**ttl 设得慷慨等于把保护
去掉而看起来还在**：命令过期不丢弃，机器人就会在暂停之后拿一张旧图上算出来的动作恢复运
动。比 watchdog 还长没有意义——驱动那边已经放弃了。

`stamp_ms`（命令何时生成）和 `obs_stamp_ms`（它依据的观测何时采集）**不是同一个数**，差
值就是全部意义所在：一条刚生成的命令完全可能来自 800 ms 前的画面，这是远端推理的典型故
障，只有第二个时间戳能抓到。两处都填 `stamp_ms` 在任何不涉及延迟的测试里都是对的，而在
生产环境里等于关掉了接收端的陈旧性检查。

## 接一个新模型

加一个 provider 就是加一个文件（`actucore/plugins/vla/providers/<name>.py`）：

```python
def PROVIDER(descriptor: dict, config: dict | None = None, on_status=None):
    ...     # 返回实现 capabilities / infer / health / close 四个方法的对象
```

四件事按这个顺序想清楚：

1. **`capabilities()` 必须便宜且诚实。** 它在权重加载完成之前就要能回答，协商才能在几个
   GB 落盘之前就否掉一个动作空间对不上的 checkpoint。不要声称没实现的能力——`supports_rtc`
   就是例子：LeRobot 为流匹配策略提供了 RTC，但 `smolvla` provider 没实现它需要的 prefix
   conditioning，所以它必须报 `False`，否则卡片会传进一个没人处理的 `inference_delay`。
2. **权重按规范下载。** 复用 `perception/utils/model_downloader.py`，遵守
   `perception/README.md` §"Model downloads: the two rules"：size+sha256 pin、多源按实测
   速度择优、**必须上报进度**（`on_status` 就是为此存在的）。SmolVLA 的 checkpoint 906 MB
   加 backbone 约 1 GB，是全系统最大的下载。
3. **懒 import、懒下载、懒加载。** torch / lerobot 在用到的函数里才 import，否则没装它们
   的镜像连 `mock` 都启动不了。权重在后台线程加载，期间 `health()` 为 False，卡片报
   `loading` 而不是 ready。
4. **`on_status` 即使没有下载也要接。** 卡片不问自己构造的是哪个 provider。

## 接一台新机器人

给驱动加一张 **servo 卡片**。现成的两个可以照抄：`realman/rm75_6f_v/servo.py`（单臂 7 自
由度）、`x-humanoid/tianyi2.0/servo.py`（双臂 + 手，用 descriptor 的 `groups`）。

要做的是两件事：**`build_descriptor()`** 声明这台机器接受什么，**`ControlSink`** 跑每条命
令。安全属性来自 sink 而不是你的文件——新鲜度、多来源仲裁、步长限幅、硬限位、watchdog、
升级停机都在 `common/control/sink.py` 里，且脱离机器人可测。

**不要自己写这些检查。** 详细规范和 `ControlSink` 的用法见 `README_dev.md`
§"Continuous Control"。

## 对 servo 的要求

分两部分：现在就必须满足的（否则 `parse_descriptor` 或协商会拒绝），和随着模型能力上来
会要求的。

### 现在（硬性）

| 要求 | 为什么 |
|---|---|
| `descriptor` 用工程单位，`joint_names` 的**顺序就是** `values` 的含义 | 顺序错了不会报错，只会让机器人做出另一件事 |
| `limits.lower/upper` 必填 | sink 的硬限位依赖它 |
| `rate.max_hz` / `expected_hz` / `watchdog_ms` / `max_obs_age_ms` 全填 | 前两个进协商，后两个是两条独立的保护 |
| `force_torque` 必须出现，没有就写 `null` | 省略 = 被误以为有力保护 |
| 反馈与指令出自**同一份** `build_descriptor()` | 两条路径不能对这台机器能做什么有分歧 |
| 单位换算只在边界做一次 | rm75 的 descriptor 是弧度、SDK 吃角度，转换紧贴 SDK 调用 |
| 运动使能/人工确认在 `start` 时检查 | 命令频繁不是跳过这道门的理由，start 是有人在场的那一刻 |

### 将来（会加，但还没有）

这些不是设想，是当前实现已经能看到边界的地方：

- **力控与阻抗。** `mode` 已经预留了 `joint_torque`，`FORMATS` 也映射了
  `control/joint-torque`，但没有任何 servo 卡片实现它，`ControlSink` 也没有力维度的限幅
  语义。真要做接触任务时，descriptor 需要的不只是 `force_torque` 上限，还有刚度/阻尼的可
  配置范围。
- **更高的控制频率与抖动指标。** 现在两张卡都按 30 Hz 下发，上限分别是 100 Hz（rm75）和
  50 Hz（天轶 2.0）。rm75 的厂商高跟随模式要求周期低于 10 ms（即 ≥100 Hz），我们跑的是低
  跟随；而 descriptor 目前**只声明频率上限，不声明抖动**。
  一个能跑 100 Hz 但偶尔卡 40 ms 的通路和一个稳定 50 Hz 的通路，对策略的意义完全不同，将
  来需要 `rate.jitter_p99_ms` 这一类字段。
- **真实的观测时刻。** `obs_stamp_ms` 的保护只有在它是**传感器采集时刻**而不是「驱动读到
  的时刻」时才成立。现在没有任何机制强制这一点，也没有跨机器的时间同步——远端推理一旦上
  线，这会是第一个暴露的问题。
- **动作块级的接受/拒绝回执。** 现在的拒绝是逐条的（sink 的 `Verdict`），卡片按 rate 把
  块拆开下发，驱动不知道这一批属于同一个块。要做碰撞再校验或块级回滚，需要驱动能对**一
  个块**表态。
- **e-stop 与 `pause` 的语义差别要落到硬件。** 当前实现里暂停是「停止发送」，机器人保持
  位置并在下一条有效命令到达时**无预警恢复**。`ttl_ms` 是唯一拦住它拿旧命令恢复的东西。
  真正的 e-stop 应该是驱动侧的、需要人工复位的状态，而不是上游停止发送。

## 云端边界

`phanthymotus-cloud`（closed source，**目前仍是空仓库**）放的是跑在机器人**之外**的模型服
务。划分标准不是「是不是推理」，而是**谁负责资源竞争**：

- 机上推理和 perception 抢同样的 7 GB、同一块 GPU——这个取舍只有本项目能做，所以
  `perception/` 和 actucore 的本机 provider 留在这里。
- 机外推理不和机器人抢任何东西，它的取舍是吞吐、成本、弹性伸缩——另一门手艺、另一套 KPI。

机器人这一侧永远只有 `{endpoint, key, model}`，和 agent-core 配 LLM 一模一样。两条不变量：
**控制回路不经过计费网关**（用量事件异步推给 resource-center，计费挂掉不能让机器人停），
以及**仓库拆分不等于抽象拆分**（本地和远端在同一个 provider 接口后面）。

VLA 没有 OpenAI 那样的现成协议——一次请求要带多路图像 + 关节状态 + 任务串，返回的是整个
动作块——所以协议 `motus.vla/1` 由我们自己定义，但沿用 OpenAI 的**所有权模型**：规范是一
份文档，不是一个共享库。不要抽出共享 schema 包，那会把两个仓库耦合到实现细节上；强制的
`schema` 版本字段才是让它们发布节奏解耦的东西。

## 已知缺口

**碰撞再校验没有实现。** MoveIt Pro 会把动作块的每个点对着带 padding 的 planning scene 检
查，并持续监视场景以便中途出现的物体能让机器人停下。我们没有 planning scene——只有 URDF，
没有场景表示、没有 FK/碰撞运行时。在有之前，缓解手段是流程性的：先在仿真里跑到你信任这
个策略，然后第一次上真机降速运行，有人在场并且有物理急停。

**连续性是控制器的职责。** 平滑、加密、块间融合、已提交窗口的大小，都发生在轨迹被执行的
地方。有一条推论属于这里：**已提交窗口至少要覆盖 p99 推理延迟**，否则机器人会在两个块之
间停顿；而窗口越大，急停真正停下来所需的时间越长。

**一台机器人上多个命令来源的仲裁只做了一半。** `ControlSink` 按 `source` + `priority` 仲
裁（高优先级来源只要还在发就一直持有通道），但优先级是**发送方自己填的**——`vla` 卡片从
配置里读，默认 50，configSchema 里就是一个操作员可改的整数字段。没有系统级的分配表来保证
「遥操作一定压得过策略」，两张卡片配成同一个数字时，赢的只是 seq 更新的那条。
