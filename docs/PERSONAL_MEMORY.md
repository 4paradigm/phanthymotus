# 个人记忆与心理支持模式

默认关闭。关闭时保留原有 prompt、记忆工具、技能、历史、后台代理和设备功能。开启后进入独立心理支持模式：现有人脸或声纹任一已绑定身份匹配即可自动解锁，无手机确认。

## 数据存放与边界

用户、绑定、记忆和对话存储在机器人本地 `DB_PATH` 指定的 SQLite（默认 `agent-core/resource/data.db`）的 `pm_*` 表中。旧 `chat_*`、子代理结论和 `prompt_memory.md` 不迁移、不删除，也不注入个人模式，避免历史个人内容被当成公共知识。

本地存储不等于内容不出设备：使用云端模型时，对话、必要历史和相关记忆仍随请求发送到模型服务。完全离线需要本地模型。没有新增数据库加密；主机/容器管理员、持有 ACCESS_TOKEN 的运营者和配置的模型提供方属于可信边界。

按用户选择接受生物识别误匹配、照片/录音/合成声音冒用的风险；没有活体或防重放保证，相似度不是身份概率。心理支持不替代专业服务，不提供诊断或药物建议。

## 启用

1. 先配置部署 `.env` 的 `ACCESS_TOKEN` 并重启。个人模式启动和所有个人记忆管理 API 都要求认证；运营者 token 不提供给普通用户。
2. 在感知画布连接 ASR、声纹、人脸识别，记录真实 DDS source 和已绑定本机输出工具的完整名称。源名单只填可信感知 topic，不能填远程消息、channel 或背景旁听 source。
3. 停止服务，在目标部署的 agent-core 目录使用应用的 Python 环境写配置，然后重启：

```python
import sys
sys.path.insert(0, 'src')
import config
config.main['personal_memory'] = {
    'enabled': True,
    'voice_sources': ['dds:/填写实际ASR输出topic'],
    'face_sources': ['dds:/填写实际摄像头输出topic/face'],
    'face_max_age_s': 3,
    'output_tools': ['填写画布已绑定的本机TTS工具完整名称'],
}
```

4. 通过下面的运营者 API 建档并绑定已经登记的感知身份，然后开始智能控制。`remember` 默认 false，只有用户同意长期保存时才设为 true。`onboarding` 控制是否主动提出可跳过的了解问题。

配置、启用或关闭模式都需要停止并重启服务，不支持通过通用配置 API 热切换。看到人只更新新鲜身份依据，不自行开始咨询；ASR 到达时优先用声纹，或最近单人人脸证据。远程聊天不能借用摄像头在场身份。

建议在感知配置中关闭陌生人 auto_enroll，并在同意后注册。本分支不改原有感知默认值。多个 face/voice source 必须属于同一组各自共享身份库的感知实例，不同设备不能混用 p-N。更换模型、重建或替换身份库后先解除旧绑定并重新登记。

## 运营者 API

前缀 `/api/personal-memory`，请求头 `Authorization: Bearer <ACCESS_TOKEN>`。

| 操作 | 方法和路径 | 请求体 |
|---|---|---|
| 建档 | POST `/users` | `{"name":"小王","remember":true,"onboarding":true}` |
| 用户列表 | GET `/users` | 无 |
| 绑定人脸 | POST `/users/{user_id}/bindings` | `{"modality":"face","external_id":"p-3"}` |
| 绑定声纹 | POST `/users/{user_id}/bindings` | `{"modality":"voice","external_id":"p-8"}` |
| 解除绑定 | DELETE `/bindings/{modality}/{external_id}` | 无 |
| 记忆/问答偏好 | PUT `/users/{user_id}/preferences` | `{"remember":false,"onboarding":false}` |
| 删除用户及关联数据 | DELETE `/users/{user_id}` | 无 |
| 公共候选列表 | GET `/public-candidates` | 无 |
| 批准公共知识 | POST `/public-candidates/{id}/approve` | 无 |
| 删除公共候选/已批准条目 | DELETE `/public-candidates/{id}` | 无 |

face:p-3 和 voice:p-3 不自动关联；只有明确绑定同一个 user_id 才属于同一用户。已有绑定不能直接转移到别人。

## 会话和记忆

- 任一已绑定身份匹配后，本人的全部个人记忆都可检索。默认注入最新20条结构化记忆，其余通过 `memory_recall(query, limit, offset)` 分页查询，避免一次塞满上下文。
- 语音到达时固定身份，再排队；排队期间出现的另一张脸不会重新归属旧语音。每轮加载对应用户独立近期历史，清空旧摘要。未知、未绑定、多人或模态冲突时作为访客，不继承上一用户，不保存访客历史。重启后也等待新匹配。
- `remember_personal` 区分 fact（自述）、preference、event、analysis（待核实假设），并绑定当前来源 turn。可用 `forget_personal` 删除错误记忆，再保存更正。第三人的描述只归当前用户，不能写入第三人档案。
- 已识别但不保存长期记忆的用户，仍可在当前连续会话中使用最多18轮的临时上下文，只留内存；换人或下次进入时距离上轮超过5分钟则清除。访客不保留跨轮历史。
- 问题可跳过，也可选择自由聊天；记忆中的分析不是疾病或人格定论。敏感内容应先确认保存意愿。
- `set_memory_preferences(false, false)` 关闭并删除个人记忆、对话和未公开候选。已经人工批准的通用知识需单独删除；删除整个用户会删除其所有关联公共条目。
- `propose_public_memory` 只提交通用知识候选，运营者审核无个人内容后批准；模型不能直接公开私人对话。语义审查由运营者负责。
- 不写旧共享历史，不生成共享咨询日报，不自动吸收旁听内容。

## 与原功能的关系

默认关闭，旧路径不变，独立 worktree 不修改原分支和线上部署。

开启时暂不向咨询会话开放共享任务、后台子代理、任意文件/Shell/Python、跨机器人委派、原始传感器历史、旧共享记忆工具及其他已安装技能。只开放个人记忆工具、finish 和配置的本机输出工具，实际派发也校验权限。原有设备安全/停止通路保留。

这是可切换的独立咨询模式，不是对所有旧工具的多用户改造。需要完整常规机器人功能时关闭并重启。output_tools 只加入经审核的本机输出工具；加入任意读取、跨设备转发工具会扩大信任边界。

本模式不向共享 `/decision_core` DDS topic 广播个人决策或工具结果；停止写 LLM 请求/回复 JSONL、HTTP 请求体调试文件和主循环请求快照，性能日志去掉触发文本和 metadata。运营者仪表盘仍显示活动；perception 原有的原始音频、声纹样本、访问记录及驱动日志仍由原模块管理，部署方需单独制定保留规则。

## 测试、删除与回退

在开发环境 `cd agent-core && python -m pytest -q`。自动测试不使用真实模型、摄像头或麦克风；上线前仍需真机校准误接受/误拒绝、噪声、短语音、遮挡、多人切换、TTS回声和输出绑定。

删除是逻辑删除，不保证磁盘取证不可恢复。离线备份不会随在线删除改写，恢复备份时需重新应用删除记录。

回退：停服务，将 personal_memory.enabled 设为 false 并重启。旧表和旧长期记忆仍保留；不要将未经分类的 pm_* 内容导入共享 prompt。
