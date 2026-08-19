# ActuCore — 执行模型层

ActuCore 是 Perception 在执行侧的对称层。Perception 把原始数据流变成语义；ActuCore 把意图/目标变成运动指令。

```
Hardware → Driver·Sensor → Perception → Agent Loop → ActuCore → Driver·Actuator → Hardware
                                                     ↑ 这一层
```

执行模型（VLA 策略、导航、抓取策略、locomotion、whole-body control）以**卡片**的形式挂在这里，聚合成一个 MCP HTTP server，由 Agent Core 通过 MCP JSON-RPC 调用。

**当前不带任何卡片** —— 这一版是骨架加全链路打通。`tools/list` 返回空数组，服务照样注册、探活、在 Dashboard 侧边栏「执行」分区里显示（count 0）。

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

只有 Jetson GPU 版 —— 执行模型（VLA、抓取策略、locomotion）都要 GPU，没有 CPU 变体。

```bash
./deploy/build_actucore.sh                    # JetPack 5.11（默认）
./deploy/build_actucore.sh --jp-version 6.1   # JetPack 6.1
./deploy/build_actucore.sh --mirror tuna      # 指定 pip / apt 源
```

镜像刻意做薄 —— 除了 MCP server 本身，只保留 base 镜像自带的 CUDA torch 和 ROS2 环境。加卡片时把该卡片的依赖放在它自己的 `RUN` 层，不要预装在基础层里。

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

需要 ROS 命名空间的卡片（topic 里要带机器人名）多一步：namespace 为空时用 hostname 兜底，写法参照 `perception/main.py` 里 htmsg / vop 的注册块。

完整的、带 ROS 节点的卡片实现可以直接看 `perception/plugins/htmsg/plugin.py` —— 它是最干净的范例。
