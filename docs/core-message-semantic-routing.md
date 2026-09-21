# Core Jev 消息接入与路由

## 开启方式

1. 准备 TypeSafe 或 OpenRouter API Key；可以在下面的卡片配置中填写，也兼容 **Core 服务环境**中的 `TYPESAFE_API_KEY` 或 `OPENROUTER_API_KEY`（别名 `OPEN_ROUTER_KEY`）。不要将密钥写进 Solution 或 Git。
2. 停止智能控制并取得 Canvas 编辑权，打开 `decision_core` 配置。
3. 打开「启用 Jev 语义接入与消息路由」。默认关闭；旧项目不会自动启用。
4. 在 Jev Base URL 中选择 TypeSafe（`https://api.typesafe.ai/v1`）或 OpenRouter（`https://openrouter.ai/api/alpha`），在 API Key 密码框填写对应密钥。只有一份已保存 Key，不按服务商分别保管；切换服务时请替换为对应 Key，留空保留现有值。保存后不回显，已保存值优先于所选服务的环境变量。旧配置缺省 TypeSafe，不自动改用 OpenRouter。模型保持 `jev-latest` 即可：TypeSafe 原样使用，OpenRouter 映射为已实测的 `typesafe/jev-1.13`；不会调用聊天接口。密钥不进入卡片配置或 Solution；导入方案不会替换本机密钥。存储使用现有 SQLite（非加密保险库），应保护数据库及备份权限。默认读取 Core 内 `./resource/memory/identity.md` 的完整 UTF-8 正文；可修改为 Core 可读的路径，空值恢复默认。配置页不再显示诊断块，诊断见 Activity 或状态接口。
5. 免唤醒词场景将上游 ASR 的 `trigger_mode` 设为 `vad`。保留 `asr_kws` 会先过滤掉没有唤醒词的语音；Core 不自动修改其他卡片。
6. 保存后重新开始智能控制。卡片编辑锁和停止要求与现有 Canvas 一致。

开启后，完整身份、主 Loop 原有可见的文本历史、当前 turn 和待判断消息会发送给所选 Jev 服务（OpenRouter 会经其网关转给模型提供方）；不发送原始音频/图像、环境配置密钥。工具结构中已知凭据字段会脱敏，但不能承诺自动发现对话中任意秘密；需由操作者确认数据外发适用性。

## 行为

语音先判断是否面向机器人。同一次 Jev 请求同时判断来源/对象（`human_to_robot / robot_echo / human_to_other / uncertain`）及 addressed；只有明确为 human_to_robot，且 addressed、该选项概率与 confidence 均达到语音阈值才接入。其他对象、回声、不确定、缺失或无效结果均拒绝，不得回退成 steer。通过的语音，以及网页/Channel 交互文字，再使用 `steer / interrupt / followup` 判断。空闲时正常开始处理；忙碌时沿用既有三种模式的执行逻辑。`interrupt` 不等于同步停播，更不是急停。

Core 记录本机 MCP `on_notify` 绑定的普通工具调用及 Hook 下发文本作为近期播报参考（最多 16 条，每条最多 4096 字，按长度保留至多 120 秒，停止项目时清空）。明确失败的下发移除；网络超时可能已执行，保留到过期。它不是扬声器实际播放确认，也不是 AEC。规范化后至少 6 字、整条 ASR 完全包含于参考的片段可直接拒绝；短口令及含额外内容的混合文本交给 Jev，错字回声也依赖 Jev。人类逐字复述播报可能误拒绝，无明确称呼的旁人对话仍可能无法仅凭文本区分；不声称替代声学消回声或说话人识别。未通过 Core 下发的机器人声音没有这一参考。

ACP 完成、传感器、调度器和子 Agent 回执不调用 Jev。Channel 权限、来源、回复关联和机器人消息的中断限制继续生效。Jev 不能授予权限。

只有附件而没有可判断文本的交互消息直接沿用旧链路，不丢失附件。已确认面向机器人的短句（例如“停”）不再被旧的纯时长 backchannel 规则二次过滤；ASR 去重及原有取消/播放停止流程保持不变。

初始模型 `jev-latest`，语音阈值 0.5、路由 confidence 阈值 0.5、总判断预算 2 秒（包含排队与重试）；这些是实验默认值，不是校准后的质量保证。路由不确定则使用当前已配置 interrupt mode（通常为 steer），不是固定的另一套默认值。

单 worker，等待队列最多 8 条。从 Core 接收候选消息开始计算唯一截止时间，预算为 `jev_timeout_s`（最大 5 秒）；排队、上下文准备和至多一次重判共用预算，每次 HTTP 请求只获得剩余时间，不重新计时。预算耗尽时语音不放行，文字回退旧链路；这不是后续 Core 处理或扬声器停播的时延承诺。预算内语音已经确认接入、仅模式无效时回退当前默认模式。结果绑定当时 turn，避免迟到中断新任务。

关闭开关后停止新请求，失效在途结果；待判断语音丢弃，未交付文字回到旧链路一次。停止智能控制时丢弃候选，不在恢复时重放。

开关切换不是全局 FIFO 屏障：配置持久化成功并发布到内存后，新消息按新开关状态处理，不等待旧判断取消或旧文字排空。因此关闭过程中，新直通文字可能先于旧的回退文字入队；取消中的旧文字先于旧等待队列回退，均保留原权限与来源。稳定关闭状态完全走旧入口，无 Jev 调用或语义标记；跨切换的诊断可不同。队列满时的文字直通也可能超越候选队列。取消与回退共享入队边界的提交标记，保证每个候选至多入队一次（不对上游重复发送的独立事件去重）。

替换 Solution 时，不带 Jev 配置的方案将其关闭；带配置的方案在替换前同步校验身份/密钥。画布布局、卡片配置与 Jev 设置在同一数据库事务中替换，成功后发布运行时状态；校验或数据库失败时保留旧画布及配置，不依赖后台 MCP 下发来启用 Jev。Jev 字段必须位于 decision_core 共享配置，不能放入实例配置。导出不包含 API key，身份路径按敏感本地引用清空。

## 排查与接口

- Canvas 配置仍经 `PUT /api/canvas/tool-config/agentcore/decision_core` 保存；无效开关配置返回 HTTP 400，原配置不变。
- Jev 字段只允许共享配置；实例配置、Solution 实例配置和带 instance_id 的 MCP 配置调用会拒绝这些字段（400），包括 API Key。Solution 先校验原始实例字段，再剔除共享配置中的导入密钥；拒绝时不改变画布、凭据或运行时。删除实例配置不重置共享设置或密钥，数据库失败返回 503。
- 共享保存返回 `persisted` 与 `runtime_applied`。若持久化成功但后续运行时应用失败，HTTP 200 同时返回警告，页面提示重新保存重试；此时并非回滚，Jev 设置和密钥已保存。确认重试成功后再开始智能控制。
- Core 卡片启动前会重新应用保存配置；校验/写入失败则返回失败，不订阅话题或报告启动成功。项目启动前及成功边界使旧 Jev 判断失效，停止先关闭接入再停止卡片；启动途中收到停止请求不会被启动完成覆盖。
- Solution 的原子性仅指数据库：画布、配置、Jev 设置及待同步记录同一事务保存；设备停止/配置下发不是分布式事务。响应中的 `applied.canvas.persisted` 与 `runtime_applied` 分别表示保存和同步结果，失败时页面显示警告。未完成的停止卡片及配置引用保存在 `canvas_runtime_pending`（不复制密钥），重载方案或再次启动会重试；恢复失败禁止启动，不遗失旧实例的停止目标。
- 配置校验/持久化在线程中执行，串行合并并以事务更新 Jev 与卡片配置；删除卡片配置时同时重置 Jev。数据库写入失败返回 HTTP 503，事务回滚，不把部分删除说成成功。配置请求取消不会取消已开始的数据库事务，重连后应刷新读取实际状态。
- `GET /api/canvas/semantic-routing` 返回只读状态及最近 100 条诊断，不返回身份正文、历史或密钥。
- `decision_core info` 同样提供 `semantic_routing` 状态。
- Activity 的 `semantic_routing` 区分判断结果与 `dispatch` 实际分流；可能出现 `not_addressed`、`robot_echo`、`human_to_other`、`uncertain_audience`、`invalid_audience`、`uncertain_default`、`queue_full`、`project_stopped` 或错误类型。拒绝判断带 `actual=reject`；完全匹配播报的快速拒绝带 `method=recent_speech_exact`。实时推送全局最多每秒一次，突发期间合并为最新一条，`coalesced` 表示本次合并省略的条数；不是逐消息审计流。即使客户端很慢，也只保留一个推送任务和一个待发样本，不堆积任务。检查具体消息的判断/分流应读取上述接口的最近 100 条诊断；旧记录会被覆盖，需要现场留证时及时读取。
- `api_ms` 只代表 Jev 调用，不是从说话到停播的时延。`queue_ms` 为 Core 判断队列等待。
- 原 ASR 监视画面仍显示识别文本；显示出来不代表 Core 已接受。VAD `on_hearing` 也不是语义唤醒成功。

## 验证边界

当前天轶部署：2026-09-21 21:52（北京时间）Core 已更新为 `release.260921.9d717a6`（代码 `388e71b0`），含自声/第三人门禁与服务选择。镜像测试 1397 通过、0 失败；用户明确允许不等待 Bot 审查先测试。只重建 Core，其他容器未变；控制仍关闭，Jev 原有 TypeSafe 配置与密钥保留。部署健康通过，真实语音效果待现场测试。详见 [部署记录](plans/tianyi-core-jev-deployment.md)。下文测试与旧部署条目保留各自历史时间点。

本地自动化覆盖配置、异步接入、模式、失败与失效路径。真实 API 在新增身份/历史条件下的语义效果、ROS/DDS 联调、真机回声和停止时延需要独立验证；旧离线数据指标不能直接视为本方案验收。

自声/第三人修复（尚未部署）：Core 全量 `1381 passed, 10 subtests passed`，41.96 秒；退出时仍有既有 event-loop 析构 warning。定向 `70 passed, 2 subtests passed`。真实 Jev 合成探针绕过本地精确匹配，直接调用新问题及真实 `parse_result`：11 条返回均符合预期，7 条拒绝（第三人 4、回声 3，含 ASR 错字），4 条接受（正常提问、联系第三人、停止、混合停止）。模型 `jev-1.13.0`，1086–3910 ms；5 条超过默认 2 秒预算。此探针为观察模型使用独立 5 秒上限，不修改生产配置；超预算结果不能算默认线上接入通过。此前探索性 10 条调用有 1 条超时，其余 9 条符合预期；超时不算语义判断成功。

可复现探针：在隔离 `DB_PATH`、已设置 `TYPESAFE_API_KEY` 的 Core Python 环境执行 `python agent-core/tests/semantic_routing_live_probe.py`。仅发送脚本内合成文本，不读真实身份/历史、不启动设备；逐条输出分类、接入结果及预算标记。未做真实音频验收或新修复 Bot 审查。

2026-09-21 验证记录：

- 初次 114 项 unittest：新增 semantic_routing，以及 LLM 配置、schema 条件、Channel 权限/历史、unittest 风格 interrupt 和 ACP 回归通过。此命令未执行 pytest 函数式的 `test_interrupt_all_fallback`，不能据此声称该文件通过。
- 33 项 pytest：Canvas 编辑锁、Solution、项目启动解析通过。Python 3.13 退出时出现事件循环析构警告；现有 `test_start_project_resolution.py` 的 helper 创建 loop 后未 close，本任务未修改该测试。
- 36 项 Node 测试通过（topic-derive、json-util），三个改动 JS 语法检查、10 个改动 Python 文件的 3.10 AST 检查及 `git diff --check` 通过。
- 浏览器使用 `tests/semantic_routing_preview.py`：真实注册 schema、配置 API 与现有 Canvas 弹窗，验证默认关闭、开启显示字段、非法路径拒绝、空路径恢复默认、保存刷新保持、关闭隐藏。隔离临时数据库和假密钥，不启动 Core loop 或硬件。
- 真实 TypeSafe API 仅使用合成身份/历史：直接称呼机器人通过接入（0.98）；停止讲解选择 interrupt（confidence 0.99）；讲完再做选择 followup（0.98）。返回模型 `jev-1.13.0`，单次 1236 / 952 / 984 ms；不是准确率或真机端到端时延评测。
- 2026-09-21 21:04（北京时间）已仅更新天轶 Orin Core 为 `release.260921.557124c`（d5064f42，镜像测试 1388 通过、Bot No issues found）；其他三个服务未更新/重启。只读验证启动、密码框 schema、身份文件可读、DDS 隔离及智能控制关闭通过。机上 TypeSafe key 尚未配置，上游 ASR 仍需切换 vad；未做真实 Jev/语音链路联调、TTS 停播或硬件验收。详见 [部署记录](plans/tianyi-core-jev-deployment.md)。

PR 首轮镜像测试为 1330 通过、12 失败、1 跳过，不能用上述定向测试代替全量。审查修复增加配置内存快照（启动加载，保存成功/替换 Solution 时更新）、诊断工作线程及相关回归；12 项打断测试改为显式 patch 模块。全量执行还发现原先跳过的部署进度模拟测试参数错误，已修正并改成普通 pytest 可执行入口，不涉及真实部署。

审查修复后本地 Python 3.10.20 / pytest 9.1.1 全量结果：1346 passed、8 subtests passed、零失败/跳过，41.53 秒，退出码 0；36 项 Node 测试通过。退出仍出现 event-loop 析构告警，未隐藏。镜像内结果与 Bot 审查结论以 PR 最新提交对应记录为准，不以本地结果替代。

`f7118d0a` 镜像内全量 1346 通过、0 失败。第二轮针对异步配置写入/事务问题修复后，本地 Python 3.10 全量 1349 passed、8 subtests passed，35.07 秒，退出码 0，无失败/跳过，本次未出现析构告警；新增真实 SQLite 故障回滚、并发写入、取消请求一致性测试。

外部判断使用 HTTPS 和默认服务端证书校验；不下载或执行模型代码。API 返回没有另行提供应用层签名，信任所选服务端及 TLS 链路；类型校验不构成对判断正确性的保证。

默认部署通过卡片密码框配置凭据，无需改 Compose 或重启。对应服务的环境变量回退仅适用于已显式注入 Core 进程环境的变量；宿主 `.env` 不会自动透传。本 PR 不修改部署模板、依赖或 Dockerfile。保存值及对应环境变量均无密钥时，保存启用 Jev 会报错并保留旧配置。

DDS 启动与挂载沿用主干，本 PR 的 Compose 文件与基线逐字一致。Core 的现有 profile 生产/修复链路未变；DDS 部署机制重设计不混入 Jev 功能。现场更新仍须验证实际 loopback 隔离，不能仅凭服务存活放行。

配置旁路修复后本地验证：Python 3.10 全量 1357 passed、8 subtests passed，42.12 秒；Jev/DDS 定向 62 passed；全部前端 Node 测试 93 passed，sidebar 语法检查及 diff whitespace 检查通过。新增覆盖实例字段拒绝、删除数据库失败及共享凭据保留、持久化后同步/异步下发失败和重试成功。此记录不代表新镜像已部署。

全量复测（Python 3.10、安装 Core 依赖和 pytest，从仓库根目录执行；先将 `DB_PATH` 设为本次创建的临时数据库）：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest agent-core/tests -q -ra
node --test agent-core/web/js/topic-derive.test.mjs agent-core/web/js/json-util.test.mjs
python agent-core/tests/semantic_routing_preview.py
```
