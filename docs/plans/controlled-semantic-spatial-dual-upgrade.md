# 北京 G1 Core 与 ActuCore 双升级准备

## 当前范围

用户选择先对齐上游，再由 review bot 构建双镜像，最终交付北京 G1 部署脚本。
本轮不在本地构建镜像，不直接执行真机升级。

## 实施

- 合并上游 main 260b34cb，保留导航功能及最新 Core/VLA 上游改动。
- 执行 Core、ActuCore、Web 必要回归，提交并推送，再申请 BOT 构建 Core 与 ActuCore JP5.11/JP6.1 和复审。
- 只有实际构建通过、可拉取的镜像才写入部署命令，不复用失败构建的标签。
- 部署脚本使用北京当前正式 Compose（phanthy-motus 项目），只升级 agent-core 与 actucore。
- 预检 Canvas/自动启动/Driver 任务与 motion_sequence 契约，备份配置与数据库；补齐地图和数采持久化。
- 先启动 ActuCore 并检查 /mcp/sse，再重建 Core，最后验证镜像及空闲状态。
- 保留回滚配置；部署完成和真机导航/完成通知验收分别报告。

脚本作为本地交付文件 `g1-upgrade-navigation.py` 提供，不进入产品构建。
已完成隔离配置测试：保留 Driver/环境变量、持久化挂载幂等、拒绝冲突挂载。
镜像版本与完整部署命令待 BOT 产物确定。

## 合并验证

上游 260b34cb 自动合并无冲突；Core 1356 passed、1 deselected、11 subtests，ActuCore 457 passed、7 skipped、83 subtests，Web 95 passed。Core 排除沿用已知 test_progress_stream；ActuCore 跳过项为宿主 ROS/OpenCV/Linux procfs。README 的 SSE、两输出端口和部署契约仍与合并实现一致，无需额外改写。
