---
name: multi-robot-registration
description: 在 PhanthyMotus 多机器人环境中快速完成现场登记。HR 通过 channel_request 开始、查询、更新或停止活动时使用；现场人员询问登记、明确参加、修改记录，或用“我也参加”“轮到我了吗”等自然表达参与意图时也使用。通过镜像内预置的 Runtime 8.0 保存状态，并由 leader 通过 peer_delegate 协调 member。
metadata:
  version: "3.2"
---

# 多机器人协同登记

## 分工与硬约束

Agent 只负责理解意图、自然对话、TTS、人脸识别、`peer_list`、`peer_delegate` 和 leader 的 `channel_reply`。文件状态、锁、幂等、校验、快照、统计、归档及清理由预置 Runtime 完成。

```text
RUNTIME=/work/skills/multi-robot-registration/scripts/mrr_runtime.py
DATA=/work/resource/multi-robot-registration
```

- Runtime 已随 Agent Core 镜像构建。不得生成、安装、复制、修改或展开 Python 源码，也不做日常 `--version` 预检。
- 直接执行本次需要的 Runtime 命令。如果文件不存在或 Python 无法执行，停止本次事务并如实报告“当前 Agent Core 镜像缺少登记 Runtime”；不要现场修复。
- 每台机器人文件系统独立。leader 不能读取 member 路径；跨机器人只使用 `peer_delegate` 的请求和返回。
- member 不向 HR 发消息。所有飞书业务回复只能由 leader 使用现有 `channel_reply`；禁止 REST API、Webhook、SDK 或独立飞书机器人。
- 不主动控制底盘、头部、手臂、手势或姿态。所有面向现场人员的话实际调用 TTS。
- 身份可靠后才能提交；只有 `commit.status="ok"` 且返回 `event_id` 后才能说登记或修改成功。

## Runtime 调用

每个命令使用一次 Bash 调用，JSON 从标准输入传入，不创建临时脚本：

```bash
python3 /work/skills/multi-robot-registration/scripts/mrr_runtime.py COMMAND <<'JSON'
{JSON}
JSON
```

常用命令：

| 命令 | 输入与用途 |
|---|---|
| `inspect` | `{}`；只读当前活动，不创建 session |
| `begin` | `{}`；校验活动并创建交互锁，返回 `session_id` 和完整配置 |
| `session-update` | `{"session_id":"...","data":{...}}`；仅长时间或多题交互保存中间进度 |
| `commit` | `{"session_id":"...","employee":{"stable_id":"...","name":"..."},"operation":"register|modify","answers":{...}}` |
| `abort` | `{"session_id":"...","reason":"..."}` |
| `control-leader` | leader 本机执行 `START/STATUS/SNAPSHOT/UPDATE/STOP/CLEANUP` |
| `control-member` | delegated subagent 在 member 本机执行同一控制协议 |
| `store-page` | leader 保存并校验 member 的完整 SNAPSHOT 响应 |
| `aggregate` | 汇总本机记录和已经保存的远端快照 |
| `archive-final` | 验证最终数据并写入 `archive.json` |
| `archive-read` | 读取历史归档 |

禁止绕过 Runtime 手动写 `current.json`、记录、锁、快照、响应缓存或归档。

## 员工现场热路径

根据完整语境理解意图，不做关键词机械匹配。允许简短寒暄、自然称呼和连贯回应，不朗读工具名或内部步骤。

### 明确要登记或修改

例如“我要登记”“我也参加”“帮我改一下”。若活动语境明确，直接调用一次 `begin`，不先 `inspect`，也不再次询问是否参加：

1. `begin.status="ok"`：使用其返回的活动配置自然接话，再进行身份识别和提问。
2. `busy`：请对方稍候，不创建第二个 session，不做人脸识别。
3. `unavailable/error`：说明当前无法登记，不猜测活动状态。

### 询问或语义不明确

例如“这里可以登记吗”“现在还能报名吗”“轮到我了吗”。先调用 `inspect`：

- `ready`：自然说明活动；仅在对方尚未明确参加时问一句是否现在登记。
- `no_activity/unavailable/error`：说明真实状态，不调用 `begin`，不做人脸识别。
- 对方确认参加后再调用 `begin`。

### 身份、提问和提交

- `begin` 成功后才做人脸识别。连续两次失败可询问姓名或工号，但必须得到可靠、唯一的 `stable_id`。
- 按 `begin.config.questions` 提问。简单、明确且合法的单选直接采用；只有模糊、矛盾、越界、多选超限或复杂自由文本才追问或复述确认。
- 单题短登记不调用 `session-update`。多题或明显较长的交互才按需续租和保存进度。
- 答案齐全且身份可靠后调用一次 `commit`。不要在提交前重复 `inspect`，提交后也不要用 `STATUS` 二次确认。
- `commit` 成功后自然告知完成；失败则说明尚未保存成功。员工取消、离开或持续无答复时调用 `abort`。

默认最短路径：

```text
明确参加：begin → 身份识别/自然提问 → commit → TTS 成功
先询问：inspect → 简短回答/确认参加 → begin → 身份识别/自然提问 → commit → TTS 成功
```

## 控制协议

所有机器人控制请求使用完整 JSON 信封：

```json
{
  "marker":"MRR_CONTROL_V1",
  "protocol":"mrr-v1",
  "command":"START",
  "request_id":"全局唯一 UUID",
  "round_id":"本轮 UUID",
  "leader":{"peer_id":"g1","name":"g1"},
  "config_version":1,
  "sent_at":"ISO-8601",
  "payload":{"target":{"peer_id":"g1","name":"g1"}}
}
```

命令为 `START|STATUS|SNAPSHOT|UPDATE|STOP|CLEANUP`。重试同一逻辑请求时必须复用相同 `request_id` 和完全相同的信封；新事务使用新 UUID。核对返回的协议、请求、轮次、命令和 `responder.peer_id`。只有 `status=ok` 且 `effect=applied|already_applied` 才算控制成功；`pending/conflict/error` 不得被口头改写为成功。

delegated goal 保持短小：

```text
在目标机器人执行一次 mrr-v1 本地控制事务。禁止 activate_skill、tts、channel_reply、动作、再次 peer_delegate，以及安装或生成代码。将下方完整 JSON 原样通过标准输入交给：
python3 /work/skills/multi-robot-registration/scripts/mrr_runtime.py control-member
最后只返回：MRR_RESPONSE {Runtime 的完整单行 JSON}
请求：{完整信封}
```

## HR 启动

只处理已配置 HR 渠道的 `channel_request`：

1. 从 HR 已提供的信息整理活动名称、题目/选项/约束、身份方式、重复规则、报告字段和机器人名单。只询问真正缺失且会改变执行的内容；一次性确认，不分多轮逐项确认。
2. 需要 member 时调用一次 `peer_list`，让 HR 确认名单；不自行猜选。
3. HR 确认后执行 leader `START`。payload 含 leader target、完整 `team_members` 和完整 `config`。
4. leader 成功后，对每位 member 各做一次 `peer_delegate(START)`；可并行时并行。每位 member 使用不同 `request_id`。
5. 校验返回。超时可用原请求重试一次；仍失败则标记未确认，不阻塞其他机器人。
6. 最后用一次 `channel_reply` 汇报确认配置、成功机器人和未确认机器人。

不要在启动成功后额外逐台调用 `STATUS`。

## 当前汇总与更新

当前汇总不先逐台 `STATUS`：

1. 每位 member 使用独立 `snapshot_id` 请求 `SNAPSHOT(mode="current", cursor=0, limit=50)`。
2. 每页使用新 `request_id`，复用 `snapshot_id`，按 `next_cursor` 继续。
3. 核对 responder、round、snapshot、cursor、页计数和 invalid 计数；每页通过 `store-page` 成功后才请求下一页。
4. 所有页连续、总数相符且无无效文件后，执行一次 `aggregate(local_mode="current", remote_snapshots=[...])`。
5. 用一次 `channel_reply` 报告截止时间、统计、逐人结果和每台机器人的完整性。只有诊断异常时才补 `STATUS`。

配置更新只接受 HR：确认完整新配置并递增 `config_version`，先执行 leader `UPDATE`，再向 member 下发同一完整配置。`pending` 时有限次查询状态，不能无限轮询。新增 member 用 START；移除 member 必须先 STOP 并取得其 final SNAPSHOT。

## STOP、归档与清理

这一流程不能为提速省略：

1. leader 本机执行 `STOP`，立即关闭新人接待；不等待本机会话结束就向所有原选定 member 发送 STOP。
2. `pending` 时有限查询，最终确认每台机器人的 `runtime_state="stopped"`。即使 START 曾超时，也要尝试 STOP。
3. 每台 stopped member 使用新的 `snapshot_id` 获取完整 `SNAPSHOT(mode="final")`，逐页 `store-page` 并验证游标、计数、无效文件和停止时记录数。
4. leader stopped 后执行 `aggregate(local_mode="final")`，再调用 `archive-final`。
5. 只有 `archive-final` 返回 `status=ok`、`archived=true`、`data_complete=true` 且有 `archive_digest` 时，才逐台 member CLEANUP，最后 leader CLEANUP。否则保留全部原始数据。
6. leader 使用一次 `channel_reply` 如实报告结果、完整性、归档和清理状态。

## 不可牺牲的正确性

- 多机器人独立存储、请求幂等、员工稳定身份唯一性、交互锁、STOP 控制、SNAPSHOT 完整性、最终归档和安全清理必须保留。
- 修改登记追加新事件，不覆盖旧事件；汇总按 `event_id` 去重，并按稳定员工 ID 应用 HR 确认的重复规则。
- 失败时不得声称已经启动、保存、停止、归档或清理。远端数据不完整时必须明确报告。
- 不向 HR 发送原始人脸图像、多余敏感信息、内部路径、工具日志或未确认答案。
