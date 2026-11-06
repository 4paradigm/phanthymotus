# PhanthyMotus QQ Channel — 通用 Channel 接入与部署

> **Bumi 机器人独立运行（无需 Windows 电脑）**：参见 [`deploy/qq/README-bumi.md`](../deploy/qq/README-bumi.md)。本 PR 包含 QQ Adapter 与 NapCat 模板，但**不自动安装 NapCat**；机器人必须另外运行 NapCat，再完成断电脑验收。

## 范围与版本

- V1：OneBot v11 **正向 WebSocket**，仅 QQ 私聊纯文本；不支持群聊、语音、图片、文件
- 应用端：`agent-core/src/channel/adapters/qq_onebot.py`，沿用既有 ChannelManager / Canvas / ACL
- 开发基线：公司 `main`（集成时采用的提交 `990257f`）；通过独立功能分支提交
- 自动验证：`python -m pytest agent-core/tests/test_qq_onebot.py agent-core/tests/test_qq_channel_integration.py agent-core/tests/test_qq_end_to_end.py agent-core/tests/test_qq_api_security.py -q`
- 本机真实 QQ 和机器人真机是独立验收阶段，Mock 通过不代表已完成真实测试
- NapCat 官方文档：<https://napneko.github.io/NapCatQQ-Desktop/docs/connections/>、<https://napneko.github.io/guide/boot/Shell>

## 数据流

手机 QQ → QQ 服务器 → NapCat OneBot WS → QQOneBotAdapter → ChannelManager → ACL → `/channel/request/<id>` → `decision_core` → `channel_reply(source_message_id)` → NapCat `send_private_msg` → 原 QQ 私聊。

机器人语音与音乐必须通过既有 Canvas 绑定的工具单独执行，不由 QQ Adapter 直接调用。

## 安装与连接（Windows 11 / WSL Ubuntu 24.04）

1. 在 WSL Docker 中先检查现有实例：`docker ps -a`、`docker images`、`ss -ltn`。**已有 NapCat 不得重复登录相同 QQ**。
2. 若没有 NapCat，优先使用官方支持 amd64/arm64 的 Docker 镜像，在独立目录使用 `deploy/qq/compose.napcat.yaml`。设置仅供本机使用的 `WEBUI_TOKEN` 后运行 `docker compose --env-file .env -f compose.napcat.yaml up -d`。
3. 在 NapCat WebUI 的连接配置中创建并启用 **正向 WebSocket 服务器**，监听 3001。为 **OneBot WS** 设置独立的高强度 Access Token（与 WebUI Token 不同）；不要关闭鉴权。手机扫码登录专用 QQ 账号；不要保存二维码、QQ 密码或登录态到 Git。
4. Agent Core 目前使用 Docker **host 网络**，若 NapCat 将宿主机端口映射到 `127.0.0.1:3001`，则 Agent Core 可用 `ws://127.0.0.1:3001/`。若将来改用 bridge 网络，不能复制这个地址，需用可达的容器服务名/IP。
5. Agent Core 的 Settings → Channels → Add，选择 **QQ (OneBot / NapCat)**，填 `Channel ID=qq_main`、`ws_url`、OneBot `Access Token`，保留 `Private only`，启用。
6. Canvas 为同一 Channel ID 配置 `channel_request → decision_core → channel_reply`；按项目现有方式启动控制。
7. QQ 发送 `/ping`，Agent 应通过真实 `channel_reply` 回复 `pong`；再发送正常问句核对 Activity、LLM 和原会话内回复。若当前 LLM 没有固定 ping 规则，须先配置对应测试指令，不能把未回复误判为 WS 问题。

可选的无明文 DB Token 配置：通过 API 设置 `access_token_env=QQ_ONEBOT_TOKEN` 并将该环境变量注入 Agent Core 容器，而不是直接提供 `access_token`。无论哪一种，**令牌不得写入代码、测试仓库或 Docker Compose**。

## QQ 用户授权

- 新 QQ 用户**强制**以 `viewer` 注册，即使其他 Channel 全局默认角色为 `operator`。
- viewer 可对话回复、读取被允许的只读工具，不能点歌或控制执行器。
- QQ V1 为纯文本通道：即使明确授予 `operator` 或 `owner` 且 `tool_filter=*`，服务端仍仅允许 `channel_reply` 与 `finish`，不执行音频、运动、导航、Shell 或子代理工具。
- TTS、Speaker 和点歌需要独立的后续 PR、授权策略和真机验收；不要仅修改 Prompt 或 ACL 角色就尝试开放。
- 不要通过 system prompt 代替 ACL；撤权应立即生效。HTTP 管理 API 需通过既有 Agent Core 身份验证访问。

## 常见问题

- **connected 但没回复**：先确认 Canvas 的入站卡片已启用、同 ID 的回复卡片已绑定，再看 Activity 与 `source_message_id`。
- **401/403**：检查 OneBot Token，不要混用 WebUI Token；配置不应在 URL query 中明文透传。
- **WebSocket 失败**：先确认 NapCat 登录在线、端口监听、Docker host/bridge 网络关系。
- **发送失败**：`retcode` 非 0 或超时都不能视为 QQ 已送达；超时可能处于不确定状态，不要盲目重发。
- **心跳超时**：确认 NapCat 启用了心跳事件；ChannelManager watchdog 负责重连。
- **风控与兼容性**：NapCat 非 QQ 官方开放平台，存在协议更新/账号风险；建议测试号，不对个人主号做长期无人值守承诺。
