# Core Jev 消息接入与路由

## 开启方式

1. 在 **Core 服务环境**配置 `TYPESAFE_API_KEY`。不要将密钥写进 Canvas、Solution 或 Git。
2. 停止智能控制并取得 Canvas 编辑权，打开 `decision_core` 配置。
3. 打开「启用 Jev 语义接入与消息路由」。默认关闭；旧项目不会自动启用。
4. 默认读取 Core 内 `./resource/memory/identity.md` 的完整 UTF-8 正文。可修改为 Core 可读的文件路径，空值恢复默认。配置栏显示实际路径、可读性、密钥是否配置和当前默认模式；最近诊断见 Activity 或状态接口。
5. 免唤醒词场景将上游 ASR 的 `trigger_mode` 设为 `vad`。保留 `asr_kws` 会先过滤掉没有唤醒词的语音；Core 不自动修改其他卡片。
6. 保存后重新开始智能控制。卡片编辑锁和停止要求与现有 Canvas 一致。

开启后，完整身份、主 Loop 原有可见的文本历史、当前 turn 和待判断消息会发送给 TypeSafe；不发送原始音频/图像、环境配置密钥。工具结构中已知凭据字段会脱敏，但不能承诺自动发现对话中任意秘密；需由操作者确认数据外发适用性。

## 行为

语音先判断是否面向机器人。通过的语音，以及网页/Channel 交互文字，再判断 `steer / interrupt / followup`。空闲时正常开始处理；忙碌时沿用既有三种模式的执行逻辑。`interrupt` 不等于同步停播，更不是急停。

ACP 完成、传感器、调度器和子 Agent 回执不调用 Jev。Channel 权限、来源、回复关联和机器人消息的中断限制继续生效。Jev 不能授予权限。

只有附件而没有可判断文本的交互消息直接沿用旧链路，不丢失附件。已确认面向机器人的短句（例如“停”）不再被旧的纯时长 backchannel 规则二次过滤；ASR 去重及原有取消/播放停止流程保持不变。

初始模型 `jev-latest`，语音阈值 0.5、路由 confidence 阈值 0.5、请求总超时 2 秒；这些是实验默认值，不是校准后的质量保证。路由不确定则使用当前已配置 interrupt mode（通常为 steer），不是固定的另一套默认值。

单 worker，等待队列最多 8 条，消息从 Core 收到开始有效 5 秒。接入不确定、排队超时或 API 故障时，语音不放行；文字回退旧链路。语音已经确认接入、仅模式无效时也回退当前默认模式。结果绑定当时 turn，避免迟到中断新任务。

关闭开关后停止新请求，失效在途结果；待判断语音丢弃，未交付文字回到旧链路一次。停止智能控制时丢弃候选，不在恢复时重放。

替换 Solution 时重置 Jev 为关闭，新配置需通过身份/密钥校验后才启用，避免旧方案残留不可见的开启状态。导出不包含 API key，身份路径按敏感本地引用清空。

## 排查与接口

- Canvas 配置仍经 `PUT /api/canvas/tool-config/agentcore/decision_core` 保存；无效开关配置返回 HTTP 400，原配置不变。
- `GET /api/canvas/semantic-routing` 返回只读状态及最近 100 条诊断，不返回身份正文、历史或密钥。
- `decision_core info` 同样提供 `semantic_routing` 状态。
- Activity 的 `semantic_routing` 区分判断结果与 `dispatch` 实际分流；可能出现 `not_addressed`、`uncertain_default`、`queue_full`、`project_stopped` 或错误类型。
- `api_ms` 只代表 Jev 调用，不是从说话到停播的时延。`queue_ms` 为 Core 判断队列等待。
- 原 ASR 监视画面仍显示识别文本；显示出来不代表 Core 已接受。VAD `on_hearing` 也不是语义唤醒成功。

## 验证边界

本地自动化覆盖配置、异步接入、模式、失败与失效路径。真实 API 在新增身份/历史条件下的语义效果、ROS/DDS 联调、真机回声和停止时延需要独立验证；旧离线数据指标不能直接视为本方案验收。

2026-09-21 验证记录：

- 114 项 unittest：新增 semantic_routing，以及 LLM 配置、schema 条件、Channel 权限/历史、现有 interrupt 和 ACP 回归通过。
- 33 项 pytest：Canvas 编辑锁、Solution、项目启动解析通过。Python 3.13 退出时出现事件循环析构警告；现有 `test_start_project_resolution.py` 的 helper 创建 loop 后未 close，本任务未修改该测试。
- 36 项 Node 测试通过（topic-derive、json-util），三个改动 JS 语法检查、10 个改动 Python 文件的 3.10 AST 检查及 `git diff --check` 通过。
- 浏览器使用 `tests/semantic_routing_preview.py`：真实注册 schema、配置 API 与现有 Canvas 弹窗，验证默认关闭、开启显示字段、非法路径拒绝、空路径恢复默认、保存刷新保持、关闭隐藏。隔离临时数据库和假密钥，不启动 Core loop 或硬件。
- 真实 TypeSafe API 仅使用合成身份/历史：直接称呼机器人通过接入（0.98）；停止讲解选择 interrupt（confidence 0.99）；讲完再做选择 followup（0.98）。返回模型 `jev-1.13.0`，单次 1236 / 952 / 984 ms；不是准确率或真机端到端时延评测。
- 未做机上部署、ROS/DDS 联调、真实 TTS 停播或硬件验收。

定向复测（从仓库根目录；先将 `DB_PATH` 设为本次创建的临时数据库）：

```sh
PYTHONPATH=agent-core/src:agent-core/tests python -m unittest \
  test_semantic_routing test_llm_config_sync test_config_schema_show_when \
  test_channel_authorization test_channel_role_authorization test_channel_chat_history \
  test_interrupt_fallback test_interrupt_all_fallback test_acp_barrier_finish \
  test_acp_resource_barrier test_acp_ordering
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  agent-core/tests/test_canvas_editor_idle.py \
  agent-core/tests/test_solution_test_block.py \
  agent-core/tests/test_start_project_resolution.py -q
node --test agent-core/web/js/topic-derive.test.mjs agent-core/web/js/json-util.test.mjs
python agent-core/tests/semantic_routing_preview.py
```
