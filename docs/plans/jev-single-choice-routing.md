# Jev 单次接入与路由判断

## 已确认范围

- 将一次请求内的 audience、addressed、route 三题合并为一个 Choice：ignore / steer / interrupt / followup / uncertain。
- 不再以 addressed 或 confidence 多重门槛拒绝；ignore 拒绝语音，uncertain 和超时按当前默认模式处理。非语音交互不因语音对象判断丢失。
- 最新用户选择召回优先：仅陌生称呼、ASR 近音字或疑似第三人对话不拒绝；交主 LLM 进一步判断。明确自声和明确非面向机器人的内容才 ignore。不得把这类疑似第三人放行计成“成功过滤第三人”。默认 interrupt 仍可能先中断，LLM 不是之前路由动作的保证。
- 保留本地精确回声、结构校验、权限、失效与停止门禁、至多一次交付；迟到结果不得覆盖默认模式。
- 废弃阈值从卡片 schema 隐藏，旧配置兼容读取但不参与决策。不自动改 identity 或其他卡片。
- 用历史真人正例、机器人播报、明确第三人负例及称呼歧义放行样本作真实 API 回放；对照原方案的接入结果和延迟，不以 mock 宣称准确率。
- 用户授权验证后仅部署天轶 Core；不更新其他容器，不启动智能控制或动作。部署前核对空闲、自动启动关闭、无任务与控制隔离；保留回滚。
- 2026-09-21 现场测试后追加授权：将当前改动范围化提交并推送 PR #254，更新 PR 描述中的实测收益、延迟原因及国内 API 的条件性预期，申请完整 Bot 构建/测试/审查并按意见修复。不包含再次部署、合并 PR 或将回声过滤与开关解耦。

## 执行

1. 修改问题、解析、schema 说明与测试，同步操作文档。
2. 历史样本新旧同输入交替调用，报告错误、误接入、误拒绝及超预算；本地 Core 全量回归。
3. 在个人目录准备可追溯 Core 镜像、备份与回滚，验证目标镜像。现场门禁满足后仅重建 Core，核对其他容器未变化。
4. 报告部署与现场验收分别处于什么状态，用户手动启动实测。
5. 当前源码重新执行 Core 全量回归，更新操作文档和 PR 描述；推送后等待同一 head 的 Bot 结果，修复有效问题并重新测试/推送/申请审查，不能以旧 head 的通过替代。

## 验证与部署结果

- 最新 head 审查先返回不完整的 provider-key 评论，澄清单份 Key 手动切换为用户明确选择后重审又耗尽工具预算；均不算通过。再次申请时发现上游 main 已更新且冲突，合入 `0922d699`：保留上游任务句柄取消/超时与 topic 回退，同时保留 Jev lifecycle epoch；两组新增测试都保留。新增 timeout_s 透传到本 PR 的测试包装器，首轮已复现遗漏导致的失败；不改变上游功能或重新部署。
- 合并后 Core 全量 1411 passed、1 skipped（缺 lark_oapi）、10 subtests passed，37.30 秒；前端 Node 96 passed，diff 检查通过。复核操作文档启动/停止失效及运行时同步说明仍符合当前实现，README 入口未变，无需修改。推送合并后的新 head 后重新申请完整 Bot。
- Bot [审查意见](https://github.com/4paradigm/phanthymotus/pull/254#issuecomment-5762705548) 的外部 spans / 内部 _perf_spans 非列表导致 collector 退出问题均已先测试复现，再统一规范化并过滤非字典项；畸形外部 spans 视为缺失，可回退已有时间戳推导。覆盖 Jev 开/关、坏事件后继续接收、原 ASR/Jev spans 保留和两个拼接分支。补明真实 API 探针付费且仅显式运行，不进自动测试发现。
- 此轮修复本地 Python 3.13.0 全量 1393 passed、1 skipped（缺 lark_oapi）、10 subtests passed，35.41 秒。修改仅容错与测试说明，README/操作接口无需变化；未重新部署，继续请求最新 head Bot。
- PR 等待审查期间自查发现 Solution 实例字段校验遗漏隐藏旧阈值，补全 `CONFIG_KEYS` 一致性；先扩展现有测试到全部配置键并复现失败，再修复入口，保持“Jev 仅共享配置”的契约。此修复不改变实机路由策略，不再次部署。
- 修复后本地 Python 3.13.0 Core 全量 1391 passed、1 skipped（缺 lark_oapi）、10 subtests passed，35.23 秒，diff 检查通过。此前 `66a18f6f` 镜像 `release.260921.c84895a` 全量 1405 passed、0 failed（4m08s）；新增修复需重新申请同 head 的 Bot，不沿用旧结果。README/操作文档原先已规定全部 Jev 字段仅共享配置，本修复恢复既有契约，无需改变用户操作步骤。
- 本次提交前复测：临时独立 `DB_PATH`，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/jev-core-routing-test-20260921/bin/python -m pytest agent-core/tests -q -ra`：1391 passed、1 skipped、10 subtests passed，37.30 秒；唯一跳过为本地缺少 `lark_oapi` 的 Feishu 测试。`git diff --check` 通过；最新 head 的镜像测试及 Bot 审查另待推送后确认。
- 最终召回优先提示词：天轶 Core 内 TypeSafe 新旧各 20 次交替真实请求，使用 10 条现场 ASR 原句（7 人类问话、3 播报）及 10 条合成回归，实际 identity、固定构造上下文；不是当时完整历史快照重放。期望是当前接入策略，不是独立人工标注的真实对象准确率；逐字稿不入仓。
- 新版 15 条应转交内容全部转交（包含 3 条称呼歧义），5 条明确自声/非对话全部拒绝；旧版按同一新策略少转交 5 条。两边 20/20 返回，无请求错误；中位数旧 847.5 ms、新 826.1 ms，最慢旧 3669.7 ms、新 3258.0 ms，各 2 次超过生产 2 秒预算。诊断用 8 秒观察模型结果，不修改线上预算；超预算负例不能算线上过滤成功，精确回声另有本地守卫。
- 先前版本曾出现人名过滤过宽，后按用户最新选择明确转为召回优先；不把策略改变计作第三人过滤准确率提升。新旧耗时基本相当，无显著加速证据。
- 最终提示词 Core 全量：1391 passed、1 skipped、10 subtests passed，34.91 秒。补 Python 3.10 asyncio/builtin TimeoutError 兼容后，本机 Python 3.10 与目标 arm64 镜像均原生 unittest 81 项通过，目标用时 9.916 秒。首次镜像测试因系统 pytest 6.2.5 与 anyio 插件不兼容失败，改用这些测试本来的 unittest 入口，不安装或修改依赖。
- 2026-09-21 22:30 仅部署 Core 本地增量镜像 `local/phanthy-motus/core:jev-single-20260921-fe5b5fc1`，基于既有 `release.260921.9d717a6`，只覆盖四个源码文件及定向测试、VERSION。代码哈希 `fe5b5fc1e5aeb4fbbe73bc28c7f5213f550b00032e2ffe9ce6f4ba01d434a0ef`；没有 git commit/push、镜像发布或新 Bot 审查。
- 读回单题五选项、TypeSafe base、identity 可读、Key 保留；启动期探针连接拒绝后服务正常。智能控制/自动启动关闭、任务空、DDS 隔离；其他容器 ID/镜像/启动时间/重启数均未变。已备份 Compose、SQLite 和容器基线；回滚仅切 Core 回既有基础镜像。
- 上述为部署当时记录。随后用户已执行开关对照，汇总见 [现场观察](../core-message-semantic-routing.md#现场开关对照2026-09-21)；不是同题受控 A/B 或完整硬件验收。README 入口及外部请求路径未变，无需修改；操作文档同步当前策略、诊断字段及实测边界。
