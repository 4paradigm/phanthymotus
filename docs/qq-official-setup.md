# Tencent QQ Official private-text Channel

The official QQ SDK runs inside Agent Core. It reuses ChannelManager,
Channel MCP and Canvas channel_request / decision_core / channel_reply.
No NapCat service or Bumi-specific Driver is required.

## V1 limits and authorization

- C2C private text, passive replies bound to a verified source message ID.
- Gateway health/reconnect, inbound deduplication, per-chat history isolation.
- New QQ users default to viewer. Server-side policy permits QQ V1
  only channel_reply and finish, including for operator/owner principals.
- No group chat, proactive messages, media, robot motion, audio, shell,
  or other device control through this Channel.

## Configuration

The normal Agent Core build installs pinned qqbot-agent-sdk==1.2.2.
In Settings / Channels choose QQ Official, assign a Channel ID and the
Tencent Bot AppID. Provide the AppSecret through a managed environment
variable or a secret mount at /run/qq-official-app-secret when deploying.
Do not commit credentials, private QQ IDs, sessions or local test scripts.

Select the same Channel ID for the Canvas channel_request and
channel_reply Cards. Connect channel_request to decision_core, grant
channel_reply as an executable tool, and start the project.

## Verification

Run the QQ Official adapter, SDK contract, Canvas, ACL and session
isolation regression tests. Company CI must validate the final image.
Local private QQ -> LLM -> QQ was verified during development.
Bumi deployment still requires Machine Owner approval.
