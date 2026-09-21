# Jev 服务地址选择

## 范围

- decision_core 配置新增 jev_base_url，下拉预置 TypeSafe 与 OpenRouter 的 Jev API base；复用现有 schema 表单，不新增诊断块。
- 旧配置缺省 TypeSafe，保持默认关闭。仅支持两个已验证的 HTTPS 地址，不接受任意域名或跳转。
- 请求使用同一 state/questions 与接入策略；按服务选择 systemone 或 decisions，默认模型自动映射到已实测标识。
- 按用户确认复用现有 base_url + api_key 模式，只保存一份 Key，不做服务商隔离；切换地址时用户填写对应 Key，空值保留现有值。兼容旧 api_key 及 TYPESAFE_API_KEY，OpenRouter 支持 OPENROUTER_API_KEY / OPEN_ROUTER_KEY；密码不回显、不进日志和 Solution。
- 校验、事务、重启、Solution、脱敏与 UI 持久化均补回归；真实 API 仅用合成文本验证新请求代码。
- 不更改机器人配置、依赖、ASR/TTS 或既有超时预算；本次实现先本地验证，提交部署沿用独立门禁。

## 验证

- 定向 73 passed + 2 subtests；Core 全量 1384 passed + 10 subtests，39.06 秒，无失败。
- 覆盖两条真实请求契约、模型映射、文字只请求 route、单 Key 切换保留/替换、重启恢复、Canvas 保存读回、数据库失败保持原值及原脱敏/导出规则。
- 隔离浏览器表单：默认关闭隐藏字段；开启显示两个 URL 选项；选择 OpenRouter，保存刷新保留，密钥不回显。隔离 fixture 没有 Core 运行时，保存如实提示持久化成功但运行时应用失败；不将此视为运行时验收。
- 新生产请求函数的真实合成文本探针：OpenRouter 874 ms，正确拒绝第三人对话；TypeSafe 在 5 秒观察上限超时。本轮不改变线上预算。
- 接口和操作文档已同步；README 通用入口与依赖未变，无需调整。未部署、未声称天轶或新版本 Bot 验收通过。

## 部署阶段

用户已明确要求部署，随后明确允许不等 Bot 审查先实机测试。提交 `388e71b0` 已推送 PR #254，镜像 `release.260921.9d717a6` 内 Core 测试 1397 通过、0 失败（评论 5761522169）；21:52（北京时间）仅切换天轶 Core，审查尚待结论。沿用 [部署步骤](tianyi-core-jev-deployment.md)，切换前实时复查空闲和控制隔离，备份 Compose/数据库/容器基线；部署后其他服务 ID、镜像、启动时间及重启次数均未变。未自动填写或切换服务凭据、未开始智能控制。只读接口与启动验收通过，真实语音验收待用户执行。
