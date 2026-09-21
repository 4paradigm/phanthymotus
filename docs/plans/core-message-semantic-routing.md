# Core 消息接入与 Jev 智能路由

## 已确认范围

- 本次 Core/Jev 功能单独提交新 PR，不再跟进已关闭的 PR 74；用户已授权提交与推送，不部署或执行硬件动作。
- decision_core Canvas 配置增加 Jev 总开关，默认关闭，关闭时保持原行为。
- 语音先判断是否面向机器人；通过的语音与网页、Channel 等交互消息统一选择 steer / interrupt / followup。
- ACP、传感器、调度器、子 Agent 回执不经过 Jev。保留原权限、回复来源、取消及动作完成契约。
- 读取完整 identity 文件（默认 Core 文件，可覆盖路径），复用主 Loop 分层历史及当前 turn；不另建历史。
- 路由不确定时使用运行中的默认 interrupt mode；不改变现有 interrupt 的停播机制。

## 实现与接口

1. event_bus 正式入队前异步 admission；拒绝的语音不进入 recent、collector ring 或主 Loop。
2. 独立队列最多 8 条，单 worker；2 秒 API 总超时、5 秒消息有效期。语音无法确认则拒绝；非语音失败回退原链路，至多交付一次。
3. 一次 Jev 请求：语音 addressed Noul + 条件 route Choice；文字只 route。实验阈值均 0.5，模型 jev-latest，记录实际版本。
4. collector 消费内部事件级路由，不修改全局模式；session/turn/身份/配置变化使旧结果失效，有限重判，不能中断新的 turn。
5. Canvas 配置保存前验证；密钥只从 TYPESAFE_API_KEY 读取。默认身份路径自动填入，状态接口展示可读性、密钥是否配置、上游 KWS 告警与诊断记录。
6. 保留 Canvas 停止智能控制后编辑与编辑锁要求；这是既有 UI 约束，不为新开关绕过。
7. Activity 展示建议模式、实际模式、原因与耗时，不记录完整身份、历史或密钥。
8. 仅附件消息回退原链路；已确认接入的语音不再被纯时长 backchannel 规则二次丢弃。Solution 替换清理旧开关，不能残留不可见的启用状态。

## 验证

- 临时 DB_PATH：开关/持久化、语音隔离、共享上下文、模式选择、权限回归、队列溢出、超时、失效/取消与至多一次交付。
- HTTP 配置与事件契约测试；浏览器配置字段、读取失败与开关演示。
- 真实 API、DDS、停播和真机结果分别报告，mock 不替代现场验收。
- 同步用户操作文档；旧离线评测保留为独立工具，不将旧指标当本方案验收。

## 当前证据

2026-09-21 已实现。114 项 unittest、33 项 pytest、36 项 Node 测试通过；Python 3.10 AST、JS 语法及 diff 检查通过。pytest 退出有既有测试未关闭 event loop 的析构警告，未掩盖或修改该测试。

浏览器已验证真实 Canvas 配置弹窗、路径错误、默认路径恢复和开关持久化；隔离 fixture 没有主 Loop/硬件。

三条合成对话的真实 API 冒烟成功，返回 jev-1.13.0；语音接入通过，停止/后续分别判为 interrupt/followup。调用约 0.95–1.24 秒，不代表真实语音端到端延迟。

操作与完整验证命令见 `docs/core-message-semantic-routing.md`，README_zh 入口已同步。旧离线评测计划仍描述独立实验，无需改写。未修改其他 owner 的 STATUS。当前交付范围为从最新主分支创建独立功能分支，提交实现、测试和文档，推送并创建新 PR；不包含旧离线评测文件，未部署。ROS/DDS、真实长历史语义质量与物理停止时延尚未验证。
