# 企业微信机器人创建与 Channel 接入

本文说明如何创建企业微信智能机器人，并将其接入 Phanthy Motus Agent Core，实现私聊、群聊 @ 和 JPG/PNG 图片收发。Agent Core 使用企业微信官方 Python SDK 建立 WebSocket 长连接，设备需要能够访问企业微信服务，无需配置公网回调地址。

本文面向**企业微信智能机器人 API 模式**，不适用于个人微信机器人或群 Webhook 机器人。

## 1. 准备环境

- 企业微信客户端及可创建智能机器人的账号。已有组织的成员需具备创建权限；没有组织时，可参考官方[团队形态说明](https://open.work.weixin.qq.com/help2/pc/21059)创建自己的测试团队。
- 已部署包含 WeCom Channel 的 Agent Core 版本，并能打开 Web Dashboard。
- 已配置 `decision_core` 使用的模型地址、凭据与模型名称。
- Agent Core 所在设备可访问 `wss://openws.work.weixin.qq.com:443`；扫码授权需要访问 `https://work.weixin.qq.com`，接收图片还需要访问企业微信素材服务。

同一个 Bot ID 只应由一个运行中的进程连接。本地测试前，先停止正式环境中使用同一 Bot ID 的 Channel；测试结束后，再恢复正式环境连接。

## 2. 创建企业微信智能机器人

### 2.1 扫码授权

1. 确认 Dashboard 已启用身份验证：部署配置中的 `ACCESS_TOKEN` 非空，修改后需重启 Agent Core。打开 **设置 → 渠道（Channels）→ + Add Channel**，选择 `WeCom (企业微信)`。
2. 点击 **企业微信扫码授权**，用企业微信扫描二维码，按照手机上的提示创建机器人并确认授权。页面最多等待 5 分钟，超时后可重新扫码。
3. 授权完成后，Agent Core 自动保存 Bot ID 和 Secret、生成 Channel ID 并尝试建立长连接。页面显示已连接后，关闭扫码页面，继续第 4 节的 Canvas 接线。
4. 若提示已保存但未连接，在渠道列表检查连接状态并点击 **Restart**；若该机器人已经配置，请检查已有渠道。

扫码流程参考[企业微信官方 CLI 的二维码实现](https://github.com/WecomTeam/wecom-cli/blob/078e7e10c2c815734892d60b915f155d39e6687b/crates/wecom-cli/src/auth/qrcode.rs)。扫码接口可能随平台调整；如无法使用，可按照以下步骤手动配置。

### 2.2 手动创建

1. 在企业微信客户端打开 **工作台 → 智能机器人 → 创建机器人**，选择手动创建。
2. 选择 **API 模式创建**。部分客户端将此入口放在创建页底部；若已创建机器人，可在编辑页查找 **API 配置**。
3. 将连接方式设为 **使用长连接**。
4. 获取 **Bot ID** 和 **Secret**。Secret 可能需要点击“获取”才会显示，获取后妥善保存。
5. 填写机器人名称，并将**可见范围 / 可使用范围**设为需要与机器人沟通的成员。多人测试或群聊时，确认相关成员均可使用该机器人。
6. 保存配置，在企业微信中找到机器人并打开单聊；需要群聊时，再将机器人加入测试群。

不同客户端的入口名称和填写顺序可能不同，以实际界面及官方[获取 Bot ID 和 Secret 指南](https://open.work.weixin.qq.com/help2/pc/21677)为准。若没有创建入口，先检查组织的[机器人创建权限](https://open.work.weixin.qq.com/help2/pc/21663)。

长连接使用 **Bot ID + Secret**，不需要填写 URL 回调地址、Token 或 EncodingAESKey。入站图片的解密信息由企业微信随消息提供，无需在 Dashboard 额外配置。

扫码时凭据由 Agent Core 后端获取并保存在本机的 Channel 配置中；手动配置时只在 Dashboard 填写 Secret。不要将凭据提交到代码仓库或放入截图、日志、聊天记录。泄露后，在企业微信侧重置并更新 Channel。

## 3. 在 Dashboard 手动添加 Channel

1. 打开 Agent Core Web Dashboard，进入 **设置 → 渠道（Channels）→ + Add Channel**。
2. 填写以下字段：

   | 字段 | 填写内容 |
   |---|---|
   | **Platform** | `WeCom (企业微信)` |
   | **ID** | 稳定、可辨识的名称，例如 `wecom_bot` |
   | **Bot ID** | 企业微信 API 配置页生成的 Bot ID |
   | **Secret** | 同页获取的长连接 Secret |
   | **Enable immediately** | 勾选，保存后立即连接 |

3. 点击 **Add**，确认状态变为 `connected`。若未启用，先启用 Channel。
4. 修改企业微信侧配置后，可在渠道面板点击 **Restart** 重新连接。

`connected` 表示长连接认证已完成。要让消息进入 Agent 并收到回复，还需完成下面的 Canvas 接线。

## 4. 将 Channel 接入 Decision Core

1. 停止当前智能控制，进入 Canvas 编辑状态。
2. 添加 `Channel / channel_request` 卡片，在实例配置中选择刚创建的 Channel。
3. 将 `channel_request` 的 `data/json` 输出连接到 `AgentCore / decision_core` 输入。
4. 添加 `Channel / channel_reply` 卡片，在实例配置中选择**同一个 Channel**。
5. 从 `decision_core` 底部的**执行器**端口连接到 `channel_reply` 卡片。
6. 检查 `decision_core` 的模型配置，启动智能控制，并确认两个 Channel 卡片均通过启动自检。

```text
企业微信私聊 / 群聊 @
  → channel_request
  → decision_core
  → channel_reply
  → 企业微信原会话
```

Agent 通过 `channel_reply` 工具发送回复，普通模型输出不会自动发送到企业微信。工具中的 `source_message_id` 必须使用当前触发消息的 ID，以便回复到正确会话。

如需控制设备，再将相应执行器连接到 `decision_core`。Channel 本身不提供拍照或运动能力，这些能力取决于 Canvas 已绑定的工具。

## 5. 设置用户权限

在默认配置下，新用户首次发送有效消息后会登记为 `viewer`。完成 Canvas 接线并启动智能控制后，让测试者发送一条私聊消息，再到 **设置 → 渠道 → 用户（Users）**查找对应的企业微信用户并调整角色。

| 角色 | 可用能力 |
|---|---|
| `viewer`（默认） | 查询只读信息、读取本轮收到的图片，并向当前消息回复；回复附件限本轮收到的图片 |
| `operator` | 可调用 Canvas 已绑定的工具，包括执行器；可按项目文件访问规则发送本地图片 |
| `owner` | 具有执行权限及用户管理权限 |
| `blocked` | 拒绝其消息，不触发 Agent 或图片下载 |

这些是 Agent Core 的 Channel 角色，与企业微信组织内的管理员、普通成员身份无关。若修改过全局默认角色或关闭自动登记，应按实际配置处理新用户。

只做文字问答和入站图片识别时，保持 `viewer` 即可。需要拍照、执行动作或发送已有本地图片时，给实际操作者设置 `operator`，并确认相应工具已绑定。角色授权不会限制某个执行器内部的具体动作或参数。

## 6. 检查文字和图片收发

### 6.1 私聊文字

1. 在企业微信中私聊机器人，发送：`请只回复：企微链路测试成功`。
2. 在 Dashboard Activity 中确认出现 `channel:wecom` 来源的消息，并触发 `decision_core`。
3. 确认企业微信收到 `企微链路测试成功`。

通过标准是：**Channel 为 `connected`、Activity 有入站消息、Decision Core 实际运行、企业微信收到回复**。

### 6.2 群聊 @

1. 将机器人加入测试群，确认测试者在机器人的可使用范围内。
2. 在群中通过成员选择明确 **@该机器人**，发送：`请只回复：群聊测试成功`。仅输入机器人的名字不等同于 @。
3. 确认 Activity 出现该群的入站消息，且回复出现在原群。

群聊测试应使用明确 @ 消息；普通群消息不作为本指南的触发方式。

### 6.3 用户发图片给机器人

1. 在 `decision_core` 配置中开启 **模型支持图片输入（`vision_input`）**，并选择实际支持图片输入的模型。
2. 先在私聊发送一张小型 JPG/PNG 图片，并询问：`请描述这张图片`。
3. 在 Activity 中确认入站消息包含图片附件，Agent 使用 `Read` 读取该附件，并能回答图片内容。
4. 私聊通过后，再验证群聊 @ 加图片的消息。

图片下载、解密后保存在 Agent Core 的 `resource/channel_files` 目录。只有文件路径不代表模型已看到图片内容；未开启图片输入、读取失败或模型不支持图片时，不能将其当作图片识别通过。

### 6.4 机器人向用户发图片

1. 使用 `operator` 测试账号，准备一张 Agent Core **容器内可访问**、且符合项目文件访问规则的 JPG/PNG 图片。宿主机文件需先通过现有挂载或文件管理方式供容器访问。
2. 私聊请求：`请把 <容器内图片绝对路径> 的图片发给我，并附一句说明`，将占位符替换为实际路径。
3. 在 Activity 中确认 `channel_reply` 调用包含 `files`，其中 `path` 指向该图片；同时使用当前消息的 `source_message_id`。
4. 确认企业微信客户端**实际显示图片及说明**，并且“正在处理…”已结束。只有文字或工具成功提示，不算图片发送通过。
5. 再测试只发送图片，以及群聊 @ 后回复图片。需要拍照时，先确认拍照工具生成有效图片，再重复上述发送检查。

文字和图片会分别发送：适配器先结束文字 / 进度流，再上传图片素材并回复图片消息。如果文字已到而图片失败，应按报错排查图片步骤。

以上步骤用于每次部署后的验收。独立适配器测试成功不能替代完整 Agent 链路、群聊及所部署版本的验证。

## 7. 常见问题

| 表现 | 检查与处理 |
|---|---|
| 找不到创建入口或机器人 | 检查企业微信客户端、组织创建权限、机器人可见范围和可使用范围；参考第 2 节官方指南 |
| 扫码页面提示启用 Dashboard 身份验证 | 先配置并启用 Dashboard 访问码；也可以使用第 2.2 节和第 3 节手动添加 |
| 认证失败或订阅超时 | 核对 API 模式、长连接、Bot ID / Secret，并检查设备到 `openws.work.weixin.qq.com:443` 的出网；不要使用 URL 回调凭据 |
| 连接被其他进程接管 | 停止使用同一 Bot ID 的另一进程，再点击 **Restart**；适配器收到接管事件后会停止自动重连，避免互相抢占 |
| `connected` 但 Activity 没消息 | 确认智能控制已启动、`channel_request` 选择正确 Channel；群聊需检查机器人是否入群及是否真正 @ 到机器人；另检查用户是否被设为 `blocked` |
| Activity 有消息，但企业微信无回复 | 检查模型是否正常运行、`channel_reply` 的执行器连接及所选 Channel，并核对工具报错和 `source_message_id` |
| 能聊天但不能执行动作或发送本地图片 | 检查用户是否仍为 `viewer`，以及相关工具、文件路径是否可用；按第 5 节设置实际需要的权限 |
| 提示“图片处理失败或超限” | 检查 JPG/PNG 格式、大小、素材访问和脱敏失败原因；当前图片下载不接受跳转或未批准的素材主机 |
| 附件已收到，但模型看不到图片 | 开启 `vision_input`，使用支持图片的模型，并确认 `Read` 成功读取本轮图片 |
| 只有文字，没有收到图片 | 按第 6.4 节检查 `files`、容器内文件及上传 / 回复报错；确认运行镜像包含当前图片发送代码 |
| `No reply context` 或回复窗口过期 | 让用户重新发消息，并使用新消息的 `source_message_id`；不要复用旧会话 ID |
| 回复回执超时，提示 `delivery is unknown` | 先检查企业微信是否已收到，再决定是否重试，避免重复发送 |
| 尝试发送音频、视频或普通文件失败 | 当前适配器只支持图片附件，不能使用其他附件代替 |

排查时只分享必要的脱敏日志，例如 `[wecom]` 的错误类型、错误码和消息类型。不要分享 Secret、图片下载 URL、图片解密密钥或完整私密消息。

## 8. 当前限制与版本更新

- 入站支持文字、平台提供的语音转写文本、独立图片和图文混排中的图片；不处理原始音频、视频或普通文件附件。
- 图片仅支持 JPG/PNG，每张不超过 **10 MiB**，每条入站消息或每次回复最多 **10 张**。
- 入站图片仅从已批准的企业微信素材主机下载，且不跟随重定向。当前批准的主机为 `ww-aibot-img-1258476243.cos.ap-guangzhou.myqcloud.com`；若平台改用其他主机，需先核实后更新适配器。
- 回复依赖原消息上下文，不支持无触发的主动推送、任意跨会话发送、Bot @ Bot 或模板卡片交互。
- 更换代码或依赖后，需通过项目镜像构建流程获取包含更新的 Agent Core 镜像并重新部署。只修改本地仓库或重启旧镜像不会安装新代码。
- 当前 Docker 构建沿用仓库方式：根据 `pyproject.toml` 执行 `uv lock` 和依赖安装，未使用已提交的 `uv.lock` 固定镜像解析结果。

## 参考文档

- [企业微信：如何获取 Bot ID 和 Secret](https://open.work.weixin.qq.com/help2/pc/21677)
- [企业微信：智能机器人的创建权限与使用规则](https://open.work.weixin.qq.com/help2/pc/21663)
- [企业微信：如何使用团队形态](https://open.work.weixin.qq.com/help2/pc/21059)
- [企业微信：智能机器人长连接](https://developer.work.weixin.qq.com/document/60904)
- [企业微信官方 Python SDK 与示例](https://github.com/WecomTeam/wecom-aibot-python-sdk)
