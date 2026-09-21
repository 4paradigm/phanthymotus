# 天轶 Core Jev 部署

授权：只更新天轶 Orin Core；不更新 Driver、ActuCore、Perception，不启动智能控制或动作。

目标：天轶 Orin。首轮 Core 镜像 `release.260921.7950a7d`，PR #254 的 d2c001cb 构建产物。首轮回滚镜像 `release.260920.b75b6b0`；配置页修订版将以 `release.260921.7950a7d` 为回滚基线。SSH 入口与个人目录从私有设备档案及实时状态核验，不在公共仓库记录。

步骤：

1. 实时核对 Canvas 空闲、project_running=false、auto_start=false、无活动任务、DDS loopback 隔离。
2. 在已核验的机器人个人工作目录下创建本次独立备份目录，保存 Compose、数据库及容器基线；核对 arm64 镜像。
3. 仅修改 `/opt/phanthy-motus/docker-compose.yml` 中 agent-core.image；不使用会替换整个 service 的 restart helper。使用现有容器部署权限写入镜像引用，宿主执行 `docker compose up -d --no-deps agent-core`。
4. 读回版本、Jev 配置状态、安全状态；对照其他容器 ID、启动时间和镜像，确认未变。
5. 失败且安全门禁仍有效时将 image 恢复为旧值并仅重建 Core；否则停止并报告。

边界：不注入密钥、不改变挂载/网络/权限、不更改画布连线。修订版由用户在 Jev 密码框填写密钥，不要求修改宿主环境。部署与现场物理验收分开。

相关文档：沿用 PR 已有真实部署验收 SOP，本次不改变功能范围；此文件记录设备部署差异与验证结果。

## 配置页修订版部署前门禁

- 拒绝 decision_core 实例配置及实例 MCP 调用中的 Jev 字段，防止密钥进入普通配置；实例删除不得重置共享设置或凭据，数据库失败必须显式返回。
- 共享配置保存区分持久化成功与运行时下发结果；下发失败给出可重试警告，不误报数据库回滚。页面显示该警告，重新保存可重试。
- 增加接口失败路径回归，运行 Core 全量测试及前端检查，再提交、构建并申请最新提交审查。
- DDS 沿用 Core 在 ROS 启动前生成/修复 profile 的现有链路；不添加会阻碍修复的只读子挂载。以启动顺序测试和部署后隔离检测验证。
- 仅在最新镜像通过门禁且现场仍空闲时部署；否则保留当前运行镜像，不启动智能控制。

## 执行结果（2026-09-21 16:31，北京时间）

- Core 已切换为目标 arm64 镜像，镜像 ID `sha256:79bcfea5245a5c05d077a19e7d06d0c747c931adc0b60e22391b0a2e443e8e22`，`/work/VERSION` 与 tag 一致，重启计数 0。
- 对比备份逐字验证 Compose 仅 Core image 引用变化；Driver、ActuCore、Perception 的 ID、镜像、启动时间和重启计数均未变化。
- 部署后 project_running=false、auto_start=false、活动任务 0、DDS isolated=true。启动日志有 Application startup complete，最近 1000 行无 Traceback/ERROR。第一次健康查询遇到启动期连接拒绝，随后全部接口成功。
- Jev 关闭，身份文件 `/work/resource/memory/identity.md` 可读；Core 环境与宿主 `.env` 均未配置 TypeSafe key。状态接口提示现有 ASR 仍为 KWS，免唤醒词测试需要操作者改为 vad。
- 原 Compose、数据库和前后容器基线保存在上述个人目录。未调用真实 Jev、播放、机器人控制或开始智能控制；不构成物理验收。
- README 与接口说明无需功能修改；使用文档的部署状态已同步，开启步骤仍适用。
