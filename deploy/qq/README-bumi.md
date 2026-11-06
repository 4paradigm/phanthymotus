# Bumi EDU Max：QQ Channel 独立部署与验收

> 目标：**Bumi 上的 NapCat + Bumi 上的 Agent Core 独立运行**；Windows 电脑关机后，用测试 QQ 给机器人私聊纯文本，机器人仍能回复。
>
> 此文档是 `deploy/qq/compose.napcat.yaml` 的机器人端执行说明。**需要机器人 Master 授权**后才进行安装/升级；PR 构建成功不等于真机部署获批。
>
> 范围：QQ 私聊纯文本。**不包含** QQ 驱动机器人运动、音频、文件、群聊或其他工具执行；现有 QQ ACL 白名单不得放宽。

## 0. 已实现 vs. 待验收

- Agent Core 的 `QQOneBotAdapter`、ChannelManager 复用、Settings/Channels 与 ACL 已在 QQ Channel PR 中实现。
- 同仓库提供了一个独立 NapCat Compose 模板，**不会随 Core 镜像自动安装**。
- `mlikiowa/napcat-docker:v4.18.37` 在官方发行渠道提供 `linux/arm64` 镜像。存在对应架构不代表已在当前 Bumi 上真实跑通。
- **必须在 Bumi 上独立执行本文件并通过真机测试**，才能称为完成机器人交付。

## 1. 预检：只读，不改现有容器

使用 VS Code Remote-SSH 登录目标 Bumi 后，先让 Master 确认机器人确切身份和部署窗口：

```bash
uname -m
docker version --format '{{.Server.Arch}}'
docker compose version
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker info --format '{{.DockerRootDir}}'
df -h
```

要求：操作的确是 Bumi；主机可用 Docker Compose；CPU 为 ARM64/aarch64；磁盘余量足够；**记录当前 Agent Core 容器名称、镜像标签和运行配置，便于回滚**。如果目标机器人已经运行 NapCat，先确认它是否归其他人管理，不重复启动服务、端口或登录同一账号。

## 2. 在 Bumi 上安装独立 NapCat 服务（需授权）

先把本仓库的 `deploy/qq/compose.napcat.yaml` **复制到 Bumi 自己的独立目录**，不要放进原厂 Docker Compose 项目。两种情况任选其一：

- 如果 Bumi 已有该分支源码，可直接使用 `<仓库目录>/deploy/qq`。
- 如果 Bumi 没有源码，在 Bumi 上执行 `mkdir -p ~/qq-service`；在 Windows 本地 PowerShell 通过 `scp -P 2222 <本机compose.napcat.yaml文件路径> noetix@<Bumi当前IP>:~/qq-service/` 传输 Compose 文件，再在 Bumi 上进入 `~/qq-service`。

然后在 **Bumi 的该独立目录**新建 `.env`（这是本机私有配置，不能上传 GitHub），其中必须包含一行 `WEBUI_TOKEN=<32位以上随机密钥>`。不要直接使用示例占位符或把真实值粘进 PR：

```bash
nano .env
```

保存后执行以下操作（创建时限制 `.env` 的读取权限）：

```bash
chmod 600 .env
docker compose --env-file .env -f compose.napcat.yaml config -q
docker compose --env-file .env -f compose.napcat.yaml pull napcat
docker compose --env-file .env -f compose.napcat.yaml up -d napcat
docker compose --env-file .env -f compose.napcat.yaml ps
```

`docker compose up -d` 仅管理该 Compose 项目的 **NapCat 服务**，不要对机器人当前 Core、Driver 或控制服务执行 `docker compose down`。三个 `./data/` 持久化目录保存 QQ 登录、NapCat 设置和插件状态；这些文件均被 `.gitignore` 排除，不允许加入 PR。

如果机器人外网不能访问 Docker Hub，**先停止在此阶段**，与 Master 协商公司批准的镜像获取方式；不要在未知环境随意运行下载脚本。

## 3. 登录 QQ 和配置 OneBot（不外露管理端口）

Compose 将 NapCat 的端口只绑定在 **Bumi 本机 `127.0.0.1`**：WebUI 为 `6099`，OneBot WS 为 `3001`。

在 **Windows 本地**另开终端（不要在机器人终端执行），通过 SSH 本地转发访问 NapCat WebUI：

```powershell
ssh -p 2222 -N -L 16099:127.0.0.1:6099 noetix@<Bumi当前IP>
```

访问 `http://127.0.0.1:16099/webui`。在 WebUI 登录**专用测试 QQ 账号**，确认登录在线；在 NapCat 连接设置中开启 OneBot v11 **正向 WebSocket 服务器**，容器内部监听 `0.0.0.0:3001`，并配置与 WebUI Token **不同**的高强度 OneBot Access Token。由于宿主机端口只绑定 `127.0.0.1`，无需将 `3001/6099` 暴露到校园 LAN。

不要上传 QQ 登录二维码、账号凭据、OneBot Token、完整私聊内容或 `data/` 到 GitHub。

## 4. 更新 Bumi Agent Core 镜像（需公司审批）

- 取得公司 PR 审批和 Bumi Master 部署授权。
- 在 Bumi Agent Core Dashboard 的 **部署 → 我的服务 → Core** 中选择当前 PR 最新审核通过的 `core` 镜像版本并确认升级。
- **不要在机器人上另起一个 Agent Core 容器，也不要用 Driver/Perception 的 `curl ...run-pr-image.sh` 命令部署 Core**。Agent Core 通过 Dashboard 的 `/api/system/update` 和 restart helper 升级。
- 记录升级前的 Core 版本和成功升级后的版本；如果版本未进入可安装目录，先完成公司审批，不要绕过控制台审核。
- 版本若更新为新的 Commit，需以 PR Review Bot **针对该 Commit** 给出的镜像为准，不能继续声称旧版本代表新代码。

## 5. QQ Channel 指向机器人本地 NapCat

先查明 Core 容器实际 Docker 网络：

```bash
docker inspect -f '{{.HostConfig.NetworkMode}}' <实际Core容器名>
```

仅当 Core 使用 `host` 网络时，Dashboard → **Settings → Channels → QQ** 使用：
- ID：`qq_main`
- WebSocket：`ws://127.0.0.1:3001/`
- Access Token：步骤 3 创建的 OneBot **Access Token**，不是 WebUI Token
- Private only：`true`

如果 Core 不在 host 网络，**不要直接复制 `127.0.0.1`**：它会指向 Core 容器自身。应由 Master 审核安全的容器网络互通方案后，再填写实际可达地址。

然后启用原有 Canvas 的 `channel_request`、`decision_core`、`channel_reply`，两个 Channel Card 选择相同 ID `qq_main`；保持 QQ V1 只允许文本回复。

## 6. 真机独立验收（不可用自动测试替代）

由 Master 共同完成并保存隐去个人信息的结果：

1. **连接**：NapCat 日志无异常、QQ 登录在线；Core 的 QQ Channel 状态为 connected。
2. **真实 E2E**：用另一个 QQ 账号私聊发送唯一测试文本，机器人 Canvas/Activity 收到 `channel:qq`，Agent 调用 `channel_reply`，原 QQ 窗口收到回复。
3. **授权**：未知/blocked 用户和 QQ 工具调用白名单按预期拒绝执行器调用；不能因为角色改成 owner/operator 就绕过 QQ V1 白名单。
4. **断电脑**：关闭 Windows 电脑的 WSL、SSH 隧道和本地 NapCat，**再用手机 QQ 发送消息，机器人仍独立回复**。
5. **恢复**：经 Master 同意，在安全窗口内重启 NapCat 服务并验证恢复；若允许机器人重启，再检查 Docker 和 Core 启动后的自动恢复。
6. **证据**：保存不含账号/Token 的 Core 版本、Compose 服务状态、Canvas/Activity、QQ 收发和 Master 验收记录。自动化测试通过不代替这些证据。

任何一项未通过，PR 标记“仅本地 E2E 完成 / Bumi 验收未完成”，不要称为机器人独立运行成功。

## 7. 回滚与安全

- **Core**：优先通过 Dashboard 的受支持版本切换能力恢复到步骤 1 记录的原 Core 镜像；不得直接删改机器人 Core 容器或其数据。
- **NapCat**：必要时在 `deploy/qq` 目录单独 `docker compose --env-file .env -f compose.napcat.yaml stop napcat`；保留 `data/` 以便调查和恢复。不要执行删除卷/清理持久化数据操作。
- **网络**：端口保持回环地址绑定；WebUI 与 OneBot Token 独立；不要关闭鉴权或开放 LAN。
- **服务归属**：NapCat 为第三方依赖，交付的是机器人端部署配置和 QQ Channel 集成；QQ 非官方机器人登录方式存在协议更新、账号限制等风险，只使用公司认可的测试号。

官方参考：
- https://github.com/NapNeko/NapCat-Docker
- https://hub.docker.com/r/mlikiowa/napcat-docker/tags
