# ActuCore — 执行模型层

ActuCore 是 Perception 在执行侧的对称层。Perception 把原始数据流变成语义；ActuCore 把意图/目标变成运动指令。

```
Hardware → Driver·Sensor → Perception → Agent Loop → ActuCore → Driver·Actuator → Hardware
                                                     ↑ 这一层
```

执行模型（VLA 策略、导航、抓取策略、locomotion、whole-body control）以**卡片**的形式挂在这里，聚合成一个 MCP HTTP server，由 Agent Core 通过 MCP JSON-RPC 调用。

**当前卡片：`navigation`**，公开工具名 `ControlledSemanticSpatial` —— FAST-LIVO2 建图/里程计 + Nav2 规划/控制 + 语义航点，三者由卡片在**本容器内**作为 ROS 子进程托管（不用 companion 容器、运行时不碰 docker socket），对外只发布 bounded `velocity_proposal`，物理执行仍归 Driver。完整 action / topic / 配置 / 构建 / 许可证见 [plugins/navigation/README.md](plugins/navigation/README.md)。

该卡片的公开契约可复用于不同机器人；当前 runtime adapter 使用兼容性
`ubuntu` namespace，各本体由 Driver 提供符合契约的 topic、frame、标定和执行器。
**另一张卡片：`vla`，默认开启。** `enabled` 只决定这张卡片出不出现在工具列表里，不决定它动不动 —— 真正的门槛在画布连线、协商和驱动侧的检查链，见下。

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

## 构建与运行

只有 Jetson 版 —— 执行模型多数要 GPU（VLA、抓取策略、locomotion），没有 CPU 变体。`navigation` 卡片本身不用 GPU，但和它们共用这一个镜像。

```bash
./deploy/build_actucore.sh                 # JetPack 5.11（默认）
./deploy/build_actucore.sh --mirror tuna   # 指定 pip / apt 源
./deploy/build_actucore.sh --jp-version 6.1 # 上游 VLA 本地推理，不含 navigation
```

JetPack 5.11 默认继承仓库锁定的 `@sha256` 基础镜像，无需额外
环境变量，提供 navigation 与 VLA mock / vla_cloud。JetPack 6.1 保留上游
`jetson-base-actucore` 的 torch / lerobot 本地推理，通过 `Dockerfile.vla` 构建；
该镜像禁用 navigation（对应基座尚未发布），不接受 navigation base 覆盖。
`Dockerfile.vla` 保留上游 `bdeec4e` 原 `Dockerfile.jetson` 的 JP6.1 路径：
colcon/empy 和 audio_msgs 的构建步骤是原有层，不是导航新增依赖。独立文件让
JP6.1 继续使用 torch/lerobot 平台基线，避免进入 JP5.11 的导航多阶段构建。
该文件没有 APT 消费者，因此不修改继承的 APT 源或签名校验。
JP6.1 只复制 VLA 插件，构建产物从共用配置生成 navigation-disabled 变体；
最终镜像直接构造未启动的 Bundle，断言其公开工具仅为 `vla`、navigation 配置
关闭且插件不可导入。构建检查不启动 ROS、HTTP、模型推理或卡片。
5.11 的导航基础镜像预编译了锁定版本的 FAST-LIVO2、Nav2
和系统依赖，仅作为日常构建的 builder。仓库自有 ROS 包使用普通 install
编译后，最终阶段从同一个干净、锁定 digest 的 Jetson 平台镜像重新开始，
只复制第三方和自有 ROS install space 及应用代码。源码、build/log 目录和
navigation base 中额外的编译层不会进入可部署镜像。临时验证另一个基线时
可显式设置 `ACTUCORE_NAVIGATION_BASE_IMAGE`，覆盖值仍必须是精确的
`@sha256` 引用。只有导航依赖锁、补丁或系统依赖变化时，镜像维护者才
重新构建并推送基础镜像：

```bash
GIT_MIRROR_PREFIX=https://ghfast.top/ \
  ./deploy/build_actucore.sh --base --mirror tuna
```

最终镜像阶段会检查 FAST-LIVO2 与 Nav2 原生节点的动态链接、加载配置中的全部
BT 插件，并导入 ActuCore 主入口；JP6.1 同样检查主入口导入。缺失共享库或 Python
模块会使构建失败，不能用 builder 阶段或挂载源码后的测试代替最终镜像检查。

在 8 GB 等小内存 ARM64 构建机上可设置 `BUILD_JOBS=2`；该值会同时传给
navigation base 和日常 ActuCore 的 C++ 编译步骤。

基础镜像不是可部署服务，也不注册到 Resource Center。加新卡片时仍把普通依赖
放在 `Dockerfile.jetson`；只有稳定且可复用、已经成为构建瓶颈的第三方导航栈
才进入 `Dockerfile.navigation-base`。基础镜像必须在原生 ARM64 构建，脚本拒绝
再次走耗时且容易超时的 x86 QEMU 交叉编译。

当前镜像与固定导航基线的按 digest 体积测量、依赖增量及测量口径见
[导航镜像体积证据](../docs/plans/controlled-semantic-spatial-image-size-evidence.md)。

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
            # 可选：Canvas topic 输入转换为同工具 action
            # "x-topic-actions": [{"port": "goal", "action": "goto", ...}],
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

**`type` 的含义** —— 它决定 Agent Core 怎么调度这个工具：`sensor` 连续调用会被批量并行；`actuator` 和 `processor` 要过 ACP barrier，按上游 `x-resource` 和调用者顺序等待冲突动作；未声明资源时保守互斥；`resource` 是静态资源（如 URDF）。没声明 `type` 的工具默认按需要 barrier 处理（安全侧）。判定逻辑在 `agent-core/src/event/llm.py` 的 `_needs_barrier()`。

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
4. 该卡片需要的依赖加到 `Dockerfile.jetson` 自己的 `RUN` 层
5. 重建镜像、重新部署，确认 Dashboard 侧边栏「执行」分区里出现了它

需要 ROS 命名空间的卡片（topic 里要带机器人名）必须在注册块中校验类型和该
runtime adapter 的支持范围；不要用 hostname 猜测机器人 namespace。

本层最完整的范例是 `plugins/navigation/`：一张卡片对外只暴露一个工具名，内部拆成 mapping / planning / semantic 三个子组件，并在同容器里托管 ROS 子进程。只需要单个 ROS 节点的简单卡片可以看 `perception/plugins/vop.py`。
