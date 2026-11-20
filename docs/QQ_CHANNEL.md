# QQ 官方机器人

## 扫码连接

1. 在支持 ROS 2 DDS 的 Agent Core 上启用 Dashboard 身份验证，并配置可用的 Decision Core 模型。
2. 打开 **Settings → Channels → Add Channel → QQ**。使用手机 QQ 扫码并确认授权。二维码五分钟后过期，可以取消或重新生成。
3. 扫码授权后会保存渠道。空 Canvas 会生成一套最小聊天连线，有唯一 Decision Core 的闲置 Canvas 会追加 QQ 输入和回复。若正在编辑或运行，或有多个 Decision Core，请在 Canvas 手动接线。
4. 将 **Channel Request** 和 **Channel Reply** 卡片都选为同一个 QQ 渠道。连接 Request 的数据输出到 Decision Core，连接 Decision Core 的执行口到 Reply。
5. 在 Canvas 开启智能控制。QQ Gateway 连通且 Decision Core 输入订阅已生效时，扫码页显示 **QQ 已连接**。可在渠道列表分别查看 Gateway 与 Canvas 状态。

授权后如果 Gateway 或 Canvas 仍在等待，渠道和凭证已保存。网络恢复后可用渠道的 **Restart** 重新连接。容器重建时保留 `/work/resource` 卷。

## 私聊与群聊

- 打开机器人私聊，发送一条文本并等待被动回复。
- 如需群聊，在渠道列表显式打开 **群聊 @**，并在 QQ 开放平台使用群 @ 消息事件。将机器人添加到测试群，@ 机器人并发送文本。
- 扫码创建的渠道默认开启渠道级 `chat_only` 策略：只能用文本回复触发本轮的那条消息，不能调用设备、执行或其他工具。它只限制能做什么，不改变模型看到的内容——与其他输入共用同一份 system prompt、环境信息和对话历史，QQ 的对话也会进入共享历史。`chat_only` 是渠道配置项，任何平台的渠道都可以开启。

## 运维

- 使用独立的测试机器人验证。一个 AppID 同时只保留一个 Gateway 客户端。
- Gateway 使用 SDK 的 `GROUP_MESSAGES` intent 接收私聊和群 @，应用层独立控制群 @ 开关。Gateway 自身的恢复次数有限，耗尽后由渠道看门狗按 30 → 300 秒退避持续重连，网络恢复后无需手动 **Restart**。
- QQ 的 AppSecret 保存在 `/work/resource/qq-credentials/`，在 Linux 下目录权限为 `0700`、文件权限为 `0600`。SQLite 和 Dashboard 接口只保存或显示渠道配置。将数据卷作为敏感数据管理。
- 出站回复绑定原始平台消息 ID。QQ 请求超时可能发生在消息送达之后，系统会报告未确认并拒绝自动重发。
- 回滚时先在 Channels 停用 QQ，再移除对应的两张 Canvas 卡片。删除 QQ 渠道会删除本地凭证。

SDK 和扫码接口由腾讯维护。 [Tencent QQ Bot Agent SDK](https://github.com/tencent-connect/qqbot-agent-sdk)
