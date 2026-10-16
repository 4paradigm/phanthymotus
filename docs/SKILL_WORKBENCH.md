# 内嵌技能工作台

在现有执行卡上选择「录制技能」并点击原有的「执行」，打开 Driver 提供的页面。没有专用全局示教按钮或额外的全局样式。Core 复用已有 modal 样式，提供通用卡片页面容器、受限通信和本机技能导入；页面样式、示教状态、机器人适配和动作资产属于 Driver。既有 Agent Loop、技能激活、提示词与 ACP 不变。

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

必填：`slug`、`name`、`oneLiner`、`instruction`。可选：`description`、`category`、`version`、`requiredTools`、`compatibility`、`attachment`、`origin`。后三项为有界对象，attachment 校验协议、工具和资产 ID，origin 校验 MCP 与卡片实例；不接受任意外链附件。仅接收这些字段，限制大小和类型，不接受 HTML 图标、远程资源、任意配置或 active 覆盖。总定义不超过 32 KiB。

新技能默认对 LLM 可见；仍沿用 `activate_skill` 的现有指令加载机制。相同定义重试幂等，不重置用户后来修改的可见性；同 slug 不同定义返回 HTTP 409。导入不是自动执行，也不会自动授权未连接的工具。

用户需要把示教执行卡接到决策核心一次，之后同一机器人的新录制技能沿用该连接。工作台会显示这项状态，未连接时不会宣称保存后即可对话调用。

## 本次实现的调用边界

工作台中的操作绑定打开页面的 MCP 服务、工具和卡片实例。导入接口仅接收已有 Skill 格式，不会自动创建执行器连线、增加运行权限或给网页增加聊天框；自然语言入口仍由部署已有的语音或消息渠道提供。

支持动作包的 Driver 返回型号元数据和本地资产附件声明，宿主补入来源 MCP/卡片；导出及更新说明固定该来源。既有 Agent 路由仍按画布连接选择工具，多个同名服务同时接入时的自然语言选择未新增强制限制。卡片「运行技能」也仍是带资产 ID 的参数表单，尚无按保存名称选择并运行的界面。模拟浏览器验收覆盖单个绑定设备的发现、激活和 MCP 调用，未验证真实 LLM 决策或真机操作。这些限制不应被表述为已完成完整的多设备网页调用体验。

## 验证

本仓：`python -m pytest -q tests/test_skill_import.py`（在 agent-core 目录，使用既有依赖及 pytest）。

Driver 的 `tests/test_teaching_browser.py` 提供跨仓浏览器验收：真实 MCP 代理、SQLite 导入、现有技能列表、模拟机器人、起点拒绝、试运行和沙箱隔离；使用方式见 Driver 的 `common/teaching/README.md`。


## 每动作必填参数

`mcp_client._to_openai_schema` 支持 `x-action-params.<action>.required`，仅将声明的该动作参数列为必填；未声明时沿用工具级 required。解决兄弟动作参数不同、拆分后 run 的参数全为可选的问题。Driver 仍须自行执行参数和状态校验；此变更不修改 Agent Loop、提示词或执行许可。

## 完整动作文件

现有技能管理的「已安装」中可导入动作包；本机动作 Skill 详情可下载动作包或更新调用说明。导出与更新使用原来源 MCP/卡片，旧 taught UUID Skill 仅在可唯一确定原工具服务时兼容。Core 复用已有鉴权和 MCP 调用，传输不会自动连线或执行动作。

新增 `src/api/skill_packages.py` 的 `/api/skills/package` 路由：

| 接口 | 行为 |
| --- | --- |
| GET `/targets` | 当前画布中声明 `x-skill-package.schema=motus.teaching.package/1` 的 HTTP MCP 工具 |
| POST `/inspect` | 有界读取上传文件，返回 manifest 与设备卡片；不接触机器人 |
| POST `/install` | 按 card_id 发送有界分块，由 Driver 验证兼容并生成草稿；仅本机已有相同验证资产时可复用安装，保持原可见性 |
| GET `/pending` / POST `/resume` | 聚合各卡片待验证导入；从技能列表恢复对应工作台 |
| GET / POST `/session/{slug}` | 查询、设置或撤销本机此技能的限时调用授权；设置必须有网页 Bearer token 与显式条件确认 |
| POST `/approval/consume` | 只允许回环来源消费 30 秒一次性凭据；不能创建授权 |
| GET `/export/{slug}` | 从原卡读取完整包，验证长度与传输 SHA 后下载 |
| POST `/refresh/{slug}` | 由原 Driver 校验资产并重新生成说明，校验相同 slug/资产后更新；保留可见性和安装时间，拒绝并发覆盖 |

文件包含 `manifest.json` 和 `trajectory.jsonl`，最大 16 MiB，解压轨迹最大 64 MiB；Core 检查格式边界并负责传输，Driver 完成轨迹校验、型号/关节/模型匹配和重新绑定。导入创建的新草稿必须本机试运行并确认效果，随后走已有 save/import 才成为可调用 Skill。拒绝替换尚未保存的录制草稿。

导入后直接打开自动检查页面，一次点击确认本次条件并试运行，完成后确认效果启用。关闭可从技能列表「待验证 → 继续验证」恢复。目标设备已有完全相同且验证成功的本机资产时复用原资产；不采信包内授权，不重复试运行，也不重置既有 Skill 可见性。

技能详情可「允许本次直接调用」，最长 30 分钟，Driver 继续执行每次新预览及运动检查。Core 的 `skill_packages.py` 负责已登录网页的一次性授权凭据，`auth.py` 仅对回环消费端点增加精确豁免，供同机 Driver 核验；创建权限仍必须验证 Bearer token。无需给 Driver 增加网页 token，也不把授权动作暴露给 Agent。授权成功自动刷新旧 Skill 调用说明。停止、驱动重启、反馈重连、模式/标定或动作变化及执行异常由 Driver 撤销；授权不导出，不持久化。该轮没有修改 Agent Loop。

`web/index.html` / `web/js/skills.js` 复用现有技能弹窗和样式提供文件入口和型号详情，`workbench.js` 在保存时补入原 MCP/实例身份。没有机器人型号判断或运动算法放进 Core。

完整附件的资源中心发布尚未实现。线上公开客户端当前只展示文本/配置发布流程；在拿到服务端附件实现前，本 PR 不能声称动作包已可上传、审核、下载并从技能广场安装。也不会将轨迹塞进 instruction 或 configSchema。

本地验证：Skill 导入、文件路由、旧定义更新、action required，以及真实浏览器下的导出→更换模拟设备身份→导入→试运行→重新保存。该闭环硬件为模拟，未调用真实 LLM。

2026-10-16 已在上海部署：基于现场 `release.261015.0897a8d` 应用完整 PR 增量，保留基线 MCP 更新，构建 `phanthymotus-core:workbench318-20261016`（`0609067d81ce`）。候选镜像内 schema、Skill 导入、动作包、MCP 合并及错误处理五组测试 **52 passed**。真实网页已完成旧“抬手”说明更新、型号展示、完整动作包下载和上传文件检查，原 Skill 保持启用，页面无脚本错误；没有上传资源中心。

真实 Agent 随后已正确发现并调用 `g1_teaching.preview(asset_id)`，但一次物理回放遗留的 G1 交权超时导致 `motion_busy`。该问题在 Driver #412 的 `arm_stream.py` / `teleop_control.py` 修复，无新增 Core 修改。现场完成停止交权并更新 Driver 后，多次预览成功、busy=false；未再次执行动作。真实 Agent 确认→执行→完成的整段流程仍待现场验证，不将这些只读检查表述为整次物理验收成功。
