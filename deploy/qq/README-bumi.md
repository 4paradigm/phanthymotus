# Bumi QQ Channel：公司镜像部署与验收契约

> 状态：**规划 / 待审批 / 未完成 Bumi 真机验收**。本仓库 PR #372 交付 QQ OneBot Channel Adapter；NapCat 需作为另一个经公司构建和审核的 Driver 服务交付。两个 PR 均不得以本地测试冒充机器验收。
>
> **禁止在 Bumi 上自行 docker build 或直接部署未经公司批准的第三方镜像。** 只允许机器负责人按部署审批流程操作。本仓库 `deploy/qq/compose.napcat.yaml` 仅用于本地开发调试，不是机器人端正式部署说明。

## 1. 两个 PR 的职责

| 交付物 | 代码仓库 | 构建目标 | 运行职责 |
| --- | --- | --- | --- |
| QQ Channel（现有 PR #372） | `4paradigm/phanthymotus` | `core` | QQOneBotAdapter、ChannelManager/ACL、Canvas 的 channel_request → decision_core → channel_reply |
| NapCat Driver（新 PR，待创建） | `4paradigm/phanthymotus-driver` | `driver`（须符合公司 Driver 契约并获审批） | 持续运行 NapCat + Linux QQ；提供只读 MCP 状态 Card（建议 `napcat_status`） |

新的 Driver 应提供 `driver.yaml`、`Dockerfile`、`deploy/service.yml`、MCP HTTP 服务及完整测试。**不能只放 Dockerfile 就声称 Card 已可安装。** 服务归属和镜像发布仍由公司审核决定。

## 2. 预检与部署批准（只读，不修改机器）

- 确认 Bumi 机器身份、ARM64、当前 Core/Driver 容器清单、剩余磁盘、宿主机网络及已有端口占用。
- 记录现有 Core 的镜像标签、Dashboard 配置、可用回滚版本。不要停止或改动原厂运动/驱动容器。
- PR #372 代码修复后，先以**最新提交**重新触发 `/request_bot_review core`。旧提交 `a0c8dc8` 的 1720 项通过结果不能代表新代码。
- Driver PR 需独立完成公司构建、Card/MCP 校验及部署审批。
- 两个 PR 通过各自的 `/request_deploy` 流程并由 Machine Owner 明确授权后，才进入真机部署。批准之前只允许只读预检。

## 3. 真机部署步骤（需 Machine Owner 执行或批准）

1. 通过 Bumi Agent Core Dashboard 升级**现有** Core 容器为最新获准镜像；不能另外起第二个 Agent Core。
2. 使用公司 Driver Dashboard / 受控部署机制，安装经公司构建的 NapCat Driver 镜像；不得以手工 `docker build` 替代。
3. Driver 必须持久化 NapCat QQ 会话、插件配置和 OneBot 设置，升级容器不能丢失登录态。单独启用、停止和恢复 NapCat，不影响核心运动 Driver。
4. Core 和 NapCat 的通信基于真实 Docker 网络配置。若二者均处于 host 网络，则 OneBot 可用 `ws://127.0.0.1:3001/`；否则按审定的实际网络地址填写，不能直接复制 loopback 地址。
5. **安全验收**：WebUI 和 OneBot API 不对校园局域网公开；两者使用不同的强 Token，禁止把 Token、QQ 二维码、登录态或聊天记录放入 Git、PR 或日志。对于 host 网络，必须验证服务监听地址确实仅为 loopback；Compose 的端口映射不能为 host 网络提供隔离。
6. 使用专用测试 QQ 号登录 NapCat；在 Dashboard Settings → Channels 配置 `qq_main`，Private only，启用 Canvas 的两张 Channel Card 与 `decision_core`。

## 4. 验收矩阵

| 编号 | 必须真实验证的行为 | 通过标准 |
| --- | --- | --- |
| E1 | 服务状态 | Core 和 NapCat Driver 持续在线；OneBot 鉴权及 QQ 登录健康 |
| E2 | QQ 往返 | 用另一 QQ 号发送独特文本，Activity 有对应消息，`channel_reply` 回复原会话 |
| E3 | 权限 | viewer/operator/owner 的 QQ 消息都不能执行运动、Shell、音乐、TTS、其他执行工具或硬件 Hook |
| E4 | 并发隔离 | QQ 到达时本地高权限任务仍按原权限运行；QQ 不得插入该任务；两个 QQ 用户上下文不互通 |
| E5 | 断电脑 | Windows 关机、WSL/NapCat 及 SSH 隧道退出后，手机 QQ 仍能收发 |
| E6 | 异常恢复 | 经负责人批准重启 NapCat / Core 后恢复连接，账号状态持久化；原厂 Driver 正常 |
| E7 | 回滚 | 分别回退 Core 和 NapCat，保留状态目录，原有服务不受影响 |

验收证据应包含经脱敏的 Core/Driver 镜像版本、运行状态、Canvas、收发时间戳、权限拒绝与负责人结论。未通过 E1–E7 前，不得宣称 Bumi 交付完成。

## 5. PR 收尾

- 修复后的 Core PR 和新增 Driver PR **各自保持独立**，代码更新需要相应的重新构建与审批。
- 真机验证成功后，把**脱敏验收结果**补充到现有两个 PR 的描述或证据评论；若没有额外代码修改，不需要重新创建 QQ PR。
- 如果验收发现新缺陷，仅对相应责任仓库提交修复 Commit，并重新触发审核流程；不能用旧镜像的成功记录证明新代码。
- Machine Owner 提交部署验证证据，按公司审批流程推进最终验收、合并。

## 参考

- [公司 Core / Driver 构建与部署约定](https://github.com/4paradigm/phanthymotus/blob/main/PR_REVIEW_AGENT.md)
- [公司 Driver 开发契约](https://github.com/4paradigm/phanthymotus-driver/blob/main/README_dev.md)
- [NapCat 上游项目](https://github.com/NapNeko/NapCat-Docker)
