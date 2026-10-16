# 内嵌技能工作台与技能包

一张执行卡可以附带一个由 Driver 提供的交互页面。Core 给它一个隔离容器、一条受限
通道、一个本机技能导入接口和一个自动执行策略存储；页面内容、设备适配、资产校验和
运动逻辑全部属于 Driver。既有 Agent Loop、`activate_skill`、提示词生成和 ACP 不变，
没有新增全局按钮，也没有新增全局 CSS。

## Core 知道什么，不知道什么

这是本文档最重要的一节，因为前几个版本正是在这里出了问题。

**Core 不知道**：这是什么类型的设备、包里装的是什么、协议叫什么名字。具体地：

| Core 不持有 | 由谁决定 |
| --- | --- |
| 协议标识符 | 卡片的 `x-skill-package.schema`，任意字符串 |
| 包内文件名 | Driver。Core 只要求有 `manifest.json`，其余条目名不限 |
| `compatibility` 的字段含义 | Driver。Core 只读其中的 `tool` 用于选路，其余原样转发/展示 |
| 资产 ID / 绑定标识的格式 | Driver。Core 只做长度与字符集边界检查 |
| 「还在释放中」的状态词 | 工具的 `inputSchema['x-stop']`（见下） |

早期版本把 `motus.teaching.package/1`、`trajectory.jsonl`、`taught-` 前缀、32 位
十六进制资产 ID、64 字符绑定、以及一整套 `hold`/`handback`/`sdk_pending`/
`ownership_held` 状态词硬编码在 Core 里。每一项都让「接入第二种设备」等于「改
Core」，这与 `x-skill-package` 这个声明机制的目的正相反。前端一度还把
`dual_arm` 映射成「双臂」、`fixed_hand` 映射成「固定手」—— 运动学词汇出现在宿主
页面里，第二种形态只能靠继续编辑那个文件来支持。

`tests/test_skill_packages.py` 用一套故意与任何已发布 Driver 都不相同的标识
（`example.device.package/1`、`recorded-` 前缀、非 16 进制资产 ID、非 64 字符绑定）
守住这条边界：只要有任何一项需要和 Core 里写死的字符串对上，那组测试就会红。

**Core 知道**：传输边界（大小、压缩方式、条目名不得逃出目录）、谁有权开启自动执行、
包该送到哪张卡、以及本机技能库的读写。

## 扩展声明

声明放在 `inputSchema` 内，避免被现有 MCP 注册字段白名单丢弃。

### `x-workbench` —— 页面容器

```json
{
  "x-workbench": {
    "protocol": "motus.workbench/1",
    "kind": "skill-authoring",
    "title": "技能录制",
    "entry": "ui",
    "actions": ["ui", "info", "new", "check", "begin", "record", "finish",
                "inspect", "trial", "save", "run", "stop"]
  }
}
```

用户通过卡片原有执行按钮调用 `entry` 声明的动作，预期返回 `{protocol, html}`。
HTML 最大 250,000 字符，需自包含。对声明工作台的卡片，动作菜单只显示页面入口和
`x-action-params` 中的公开动作，标签取自 action 的 `oneOf[].title`；内部编写动作
只在工作台使用。`info` 必须无运动副作用且提供 `busy`，用于关闭前检查。停止须可
独立调用。只打开页面不触发运动。

iframe 使用 `sandbox="allow-scripts"`，不授予同源、弹窗、表单或顶层导航权限；宿主
前置 CSP（`default-src 'none'; connect-src 'none'`）禁止联网和外部资源。页面读不到
Core 的 token，也读不到父页面存储。

关闭时宿主先问设备 `info`：`busy` 为真则拒绝关闭；设备**读不到**（不可达、超时）
则说明原因并让操作者选择是否仍然关闭。两种情况都不会把页面锁死 —— 早期版本在关闭
失败时留着模块级的 `opened` 标志，于是整页在刷新前再也打不开任何工作台。Esc 也能
触发同一套关闭流程。

### `x-skill-package` —— 技能包能力

```json
{
  "x-skill-package": {
    "schema": "example.device.package/1",
    "pending": true,
    "persistent_execution": true
  }
}
```

`schema` 是这张卡能收的包的协议标识，必须与包内 `manifest.schema` 一致，否则 Core
在接触设备之前就以 409 拒绝。其余字段都是可选能力位；没有声明的能力，对应入口就
不出现。

### `x-stop` —— 释放协议（可选）

```json
{
  "x-stop": {
    "pending_states": ["handing_back"],
    "pending_flags": {"authority_released": false},
    "failed_states": ["refused"]
  }
}
```

Core 自己只认两个「还在停」的状态（`running`、`stopping`）和三个失败状态
（`error`、`failed`、`fault`）。需要确认交权、等输出排空的设备自己声明补充规则。
声明格式不合法时整条声明被忽略并回落到通用规则 —— 一个 Driver 不能靠发垃圾字段
搞坏全机队每张卡的停止。

两条有意为之的规则：

- **`unknown` 不是失败。** 它是设备在遥测过期时的回答（engineai/t800），把它读成
  「停止失败」会把一台已经停好的设备报成卡住。
- **被拒绝的动作不是失败的动作。** 无生命周期的插件（手势、回桩、喇叭）回答
  `{"error": "unknown action: stop"}`，意思是「我没有东西需要解除」。
  phanthymotus-driver 的 30 个 bundle 里只有 6 个用 `common/lifecycle` 自行规范化
  了这个回答，其余 24 个仍然原样返回；把它读成失败，正是那个仓库的
  `common/lifecycle.py` 记录过的、曾经让一台机器人整个 start-project 因为一条回桩
  命令而回滚的误读。
- **`error` 只在没有 `state` 时才定胜负。** 这是同一文件里 `tool_state_of` 早已写明
  的规则（`{"state": "running", "error": "last frame dropped"}` 是警告不是失败），
  停止路径现在复用 `tool_state_of` 而不是自己再写一遍。

另外，轮询间隔是 0.25s 且不超过每卡死线的 1/4。0.1s 会把 5 秒死线变成每张卡最多
50 次 MCP 调用，还是所有卡片同时扇出。

## 通信

iframe 加载后，父页面向其发送 `{protocol: "motus.workbench/1"}` 并转移一个
`MessagePort`。子页面须验证 `event.source === parent`。后续消息均走该端口：

```js
port.postMessage({id: 'request-1', method: 'call', args: {action: 'info'}});
// 返回 {id, result} 或 {id, error}
```

- `call`：仅转发到当前卡片的 MCP/工具和固定 `instance_id`，action 必须在声明中。
  不能指定任意 MCP、工具或 URL，也不能直接调用 save。
- `save`：仅向声明 `kind=skill-authoring` 的工具开放。调用该工具的 `save`，要求
  返回标准 `skill` 定义；宿主再调用本机导入 API，并返回 `installed` 与 `context`。
  字段验证和保存条件由 Driver 执行。
- `context`：只读查询当前工具是否已通过画布执行器连线连接到决策核心，返回
  `{conversation_ready, tool}`。不自动增加执行权限。

每条请求最多 40,000 字符；写操作串行，停止可以并行抵达。没有通用 fetch、任意父页面
操作或 token 通道。导航和关闭都会撤销原端口。**浏览器关闭不是机器人急停**；驱动仍须
提供独立的停止与输入看门狗。

## 本机技能导入

`POST /api/skills/import` 使用既有 Core API 鉴权，直接写入 ConfigDB 的
`skills.installed`。无需资源中心账号。

必填：`slug`、`name`、`oneLiner`、`instruction`。可选：`description`、`category`、
`version`、`requiredTools`、`compatibility`、`attachment`、`origin`。后三项为有界
对象：`attachment` 校验协议标识的**形状**（`<ns>/<ver>`）、所声明的 tool 必须出现在
`requiredTools` 里、资产 ID 在长度与字符集范围内 —— 不校验具体是哪个协议或哪种 ID
格式。`origin` 校验 MCP 与卡片实例。总定义不超过 32 KiB，不接受 HTML 图标、远程
资源、任意配置或 `active` 覆盖。

相同定义重试幂等；同 slug 不同定义返回 409。幂等判断按字段类型取缺省值：字符串和
列表有归一化的空形式（缺失等价于空），字典没有 —— 早期版本对三个字典字段也用 `''`
作缺省，于是「重新导入一个在这些字段出现之前就装好的技能」会撞上 409 并被要求改
slug，而不是预期的幂等 200。

导入不是自动执行，也不会自动授权未连接的工具。用户需要把该执行卡接到决策核心一次，
之后同一设备的新技能沿用该连接；工作台会显示这项状态，未连接时不会宣称保存后即可
对话调用。

## 技能包传输

`src/api/skill_packages.py` 挂在 `/api/skills/package`：

| 接口 | 行为 |
| --- | --- |
| GET `/targets` | 画布中声明了 `x-skill-package.schema`（任意值）的 HTTP MCP 工具 |
| POST `/inspect` | 有界读取上传文件，返回 manifest 与**已按协议和 tool 选路过**的卡片；不接触设备 |
| POST `/install` | 按 `card_id` 发送有界分块；协议不符或 tool 不符在接触设备前拒绝 |
| GET `/pending` / POST `/resume` | 聚合各卡片待验证导入；从技能列表恢复对应工作台 |
| GET `/export/{slug}` | 从原卡读取完整包，验证长度与传输 SHA 后下载 |
| POST `/refresh/{slug}` | 由原 Driver 校验资产并重新生成说明；保留可见性和安装时间，拒绝并发覆盖 |
| GET `/execution/{slug}` | 读取实际绑定后的有效选择、是否继承卡片及绑定变化提示 |
| POST `/execution/{slug}` | 网页开启/关闭单技能自动执行，并撤销该技能旧临时授权 |
| POST `/execution/read` | 精确回环、只读；Driver 每次 preview/run 读取策略，无授权写入能力 |

边界：整包 ≤ 16 MiB，`manifest.json` ≤ 16 KiB，单个条目解压 ≤ 64 MiB，全部条目合计
≤ 128 MiB，拒绝加密与非 deflate/stored 压缩，拒绝绝对路径和 `..`。**整条传输有
180 秒死线** —— 16 MiB 是最多 683 次分块调用，只给每块 30 秒超时的话一次上传能把
请求挂住几小时，而这个仓库里其他每一条面向设备的路径都有死线。

导入创建的新草稿必须本机试运行并确认效果，随后走已有 save/import 才成为可调用
Skill。不采信包内的验证声明。目标设备上已有完全相同且验证成功的本机资产时复用原
资产，不重复试运行，也不重置既有 Skill 的可见性。关闭后可从技能列表「待验证 →
继续验证」恢复。

## 持久自动执行设置

用户可在保存/导入启用时选择，也可在「我的 → 技能广场 → 已安装 → 技能 → 对话调用」
勾选「允许自动执行」。默认关闭，正常停止、停止项目及容器重启后保留。

**这个设置是「每技能 × 每卡片 × 每份录制」的，没有整卡形式。** 曾经有过：一个复选框
就授权该设备**以后**验证通过的所有技能 —— 整个功能里最宽的一次授权，而为了实现它要
去拦截通用的 `PUT /api/canvas/tool-config`、在一次配置保存里做设备 I/O、并把一个私有
字段存进用户自己的配置 blob。单技能同意覆盖真实需求，整卡形式已移除。

只有声明 `x-skill-package.persistent_execution` 的工具使用此能力。三道闸：

1. **已登录网页。** `prepare_execution_config` 和 `POST /execution/{slug}` 都要求
   Bearer token，模型直接调 MCP `config` 不授予权限。调用方送来的
   `_execution_binding` 一律丢弃并由 Core 向设备重新取得。
   部署若没有配置 `ACCESS_TOKEN`，这道闸永远打不开 —— 此时返回 403 并说明原因，
   而不是让操作者去找一个那台机器上不存在的登录入口。
2. **绑定匹配。** 策略记录设备绑定与资产绑定；换绑定后旧选择不匹配，按「manual」
   处理并提示核对。设备返回的绑定里**只有三个字段被读取**，其余（型号、几何、标定
   摘要等）原样忽略 —— 早期版本把设备回复直接 splat 进一个三参数函数，于是第一个
   按自己协议多带一个字段的 Driver 就是一个 500。
3. **画布绑定。** 工具必须接在本机的 `decision_core` 上。

开启时会先刷新该技能的调用说明，使定义学到自动执行规则，然后才落盘。

每次执行仍由 Driver 预览、检查设备/起点/控制权并等待真实回执。策略只存本机，不包含
在导出的技能包中。

## 每动作必填参数

`mcp_client._to_openai_schema` 支持 `x-action-params.<action>.required`，仅将声明的
该动作参数列为必填；未声明时沿用工具级 required。解决兄弟动作参数不同、拆分后参数
全为可选的问题。

同处的 start 前置检查改为复用既有的 `missing_required_config`：只在真正缺必填项时
阻止启动（与 `api/mcp_manage.py` 早已采用的判断一致），并且**报出真正缺的字段名**。
原来的文案是 `provider/url/key` —— 那是某个 LLM 适配器的字段表，把模型指向一张摄像头
卡片上不存在的表单。

## 已移除与临时项

三项能力在合并前被删掉了，因为每一项都是在为不存在的东西做铺垫：

| 移除的 | 为什么 |
| --- | --- |
| **30 分钟限时授权**（`session_challenge`/`authorize`/`revoke`、一次性凭据表、以及消费它所需的回环鉴权豁免） | 它被描述为「兼容早于持久设置的旧适配器」。没有这样的适配器 —— 唯一实现过这些 action 的驱动尚未合并，所以这是在跟一个从未发布的前任做兼容。顺带移除了 `/approval/consume` 那条豁免和那个带锁/过期/容量上限的内存凭据表。 |
| **旧 slug 前缀解析器** | 用于把 `attachment` 字段出现之前安装的技能映射到设备。恰好一台测试机上有这样一行；一条 DB UPDATE 能替掉两条代码路径，而且 slug 命名约定本来就不是身份机制。 |
| **整卡总开关** | 见上一节。 |

两项保留但**是临时的**，不要在它们之上再建东西：

- **浏览器上传/下载**（`/inspect`、`/install`、`/export` 和导入弹窗）是人工搬文件 ——
  在资源中心还存不了附件之前，这是技能跨设备移动的唯一手段。目标路径是
  **RC → Core → 设备**，所以 RC 附件上线后这几条会移除。
- **`refresh/{slug}`** 是「没有版本机制」的替代品。资源中心存上带版本的技能包之后，
  重录一次产出的是新版本而不是改写当前这一份，这条路由随之消失。

## 验证

```bash
cd phanthymotus && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest agent-core/tests -q
cd phanthymotus && node --test "agent-core/web/js/*.test.mjs"
```

本轮状态（2026-10-16，`fix/skill-package-decoupling`）：

- `agent-core/tests` **1592 passed, 1 skipped**；前端 **186 passed**；
  `actucore/tests` + `agents/pr_review/tests` **484 passed, 21 skipped**。全部在本机
  实测，不是推断。
- 修掉了上一版带来的 3 个回归：`test_layout_stop_removed` 的替身返回值缺 `code`，于是
  删卡清理把每张卡都报成停止失败。替身已改成 `mcp_call_tool` 真正返回的形状。
- 修掉了一条**什么都没验证**的断言：`test_skill_packages` 里检查「伪造的
  `_execution_binding` 打到通用 canvas 配置端点会被拒」的那条，用的 fixture 只挂了
  `skills.router`，所以请求是 404 而不是 401。新增 `full_client` fixture 同时挂载
  canvas 路由，并补上「伪造绑定被丢弃」「无关卡片保存不接触设备」两条。

**尚未验收**：本轮没有在真机上跑过。上一版在上海 G1 上完成的五次「抬手」、限时授权
与导入闭环是**那一版**的证据，不能作为本轮的实机验收 —— 本轮改动了停止路径的判定
规则、技能包的协议解析和自动执行的落盘顺序，这三处都需要重新在真机上过一遍。资源
中心的完整附件上传/审核/下载/广场安装仍未接通；本机技能不等于已发布到广场。

历史实机记录、截图与动作 ID 见
[Driver #412](https://github.com/4paradigm/phanthymotus-driver/pull/412)。
