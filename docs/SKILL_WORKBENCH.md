# 内嵌技能工作台

在现有执行卡上选择「录制技能」并点击原有的「执行」，打开 Driver 提供的页面。没有专用全局按钮、额外的全局样式或机器人选择页。Core 复用已有 modal 样式，提供通用卡片页面容器、受限通信和本机技能导入；页面样式、示教状态、机器人适配和动作资产属于 Driver。既有 Agent Loop、技能激活、提示词与 ACP 不变。

原有卡片只能显示参数表单和文本结果；现有技能安装接口从资源中心获取定义，不能安装本机生成的未发布定义。因此新增上述两项通用能力。若后续统一的卡片页面协议或本地安装接口覆盖它们，应迁移并移除重复实现；不在 Core 添加各机器人专用流程。

## 扩展声明

声明放在 `inputSchema` 内，避免被现有 MCP 注册字段白名单丢弃：

```json
{
  "x-workbench": {
    "protocol": "motus.workbench/1",
    "kind": "skill-authoring",
    "title": "机器人示教",
    "entry": "ui",
    "actions": ["ui", "info", "new", "check", "begin", "record", "finish", "inspect", "trial", "save", "run", "stop"]
  }
}
```

用户通过卡片原有执行按钮调用 `entry` 声明的动作，预期返回 `{protocol, html}`。HTML 最大 250,000 字符，需自包含。对声明工作台的卡片，动作菜单只显示页面入口和 `x-action-params` 中的公开动作，中文标签取自 action 的 `oneOf[].title`；内部编写动作只在工作台使用，`ui` 无需暴露给 LLM。`info` 必须无运动副作用，且提供 `busy`，用于关闭前检查。停止须可独立调用。只打开页面不触发运动。

宿主 iframe 使用 `sandbox="allow-scripts"`，不授予同源、弹窗、表单或顶层导航权限。宿主前置 CSP 禁止联网和外部脚本。页面无法读取 Core 的 token 或父页面存储。

## 通信

iframe 加载后，父页面向其发送 `{protocol: "motus.workbench/1"}` 并转移一个 `MessagePort`。子页面须验证 `event.source === parent`。后续消息均走该端口：

```js
port.postMessage({id: 'request-1', method: 'call', args: {action: 'info'}});
// 返回 {id, result} 或 {id, error}
```

- `call`：仅转发到当前卡片的 MCP/工具和固定 instance_id，action 必须在声明中。不能指定任意 MCP、工具或 URL，也不能直接调用 save。
- `save`：仅向声明 `kind=skill-authoring` 的工具开放。调用该工具的 `save`，要求返回标准 `skill` 定义；宿主再调用本机导入 API，并返回 `installed` 与 `context`。字段验证和保存条件由 Driver 执行。打开既有技能列表时自动读取最新数据。
- `context`：只读查询当前工具是否已通过画布执行器连线连接到决策核心，返回 `{conversation_ready, tool}`。不自动增加执行权限。

每条请求最多 40,000 字符；写操作串行，停止可以并行抵达。没有通用 fetch、任意父页面操作或 token 通道。关闭会撤销原端口。浏览器关闭不是机器人急停；驱动仍须提供独立的停止与输入看门狗。

示教流程首个实现见 Driver 仓库 `common/teaching/` 和 `agilex/piper/teaching.py`。

## 本机技能导入

`POST /api/skills/import` 使用既有 Core API 鉴权，直接写入 ConfigDB 的 `skills.installed`。无需资源中心账号。

必填：`slug`、`name`、`oneLiner`、`instruction`。可选：`description`、`category`、`version`、`requiredTools`。仅接收这些字段，限制大小和类型，不接受 HTML 图标、远程资源、任意配置或 active 覆盖。总定义不超过 32 KiB。

新技能默认对 LLM 可见；仍沿用 `activate_skill` 的现有指令加载机制。相同定义重试幂等，不重置用户后来修改的可见性；同 slug 不同定义返回 HTTP 409。导入不是自动执行，也不会自动授权未连接的工具。

用户需要把示教执行卡接到决策核心一次，之后同一机器人的新录制技能沿用该连接。工作台会显示这项状态，未连接时不会宣称保存后即可对话调用。

## 本次实现的调用边界

工作台中的操作绑定打开页面的 MCP 服务、工具和卡片实例。导入接口仅接收已有 Skill 格式，不会自动创建执行器连线、增加运行权限或给网页增加聊天框；自然语言入口仍由部署已有的语音或消息渠道提供。

首个 Driver 生成的 Skill 目前只记录工具名与资产 ID，尚未固定来源 MCP 服务 / 卡片；多个同名工具服务同时接入时的选择仍待完善。卡片「运行技能」也仍是带资产 ID 的参数表单，尚无按保存名称选择并运行的界面。模拟浏览器验收覆盖单个绑定设备的发现、激活和 MCP 调用，未验证真实 LLM 决策或真机操作。这些限制不应被表述为已完成完整的多设备网页调用体验。

## 验证

本仓：`python -m pytest -q tests/test_skill_import.py`（在 agent-core 目录，使用既有依赖及 pytest）。

Driver 的 `tests/test_teaching_browser.py` 提供跨仓浏览器验收：真实 MCP 代理、SQLite 导入、现有技能列表、模拟机器人、起点拒绝、试运行和沙箱隔离；使用方式见 Driver 的 `common/teaching/README.md`。
