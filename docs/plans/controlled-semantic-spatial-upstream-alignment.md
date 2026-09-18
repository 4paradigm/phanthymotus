# ControlledSemanticSpatial 与上游对齐

## 范围

将 PR #141 的 feat/controlled_semantic_spatial（9f7461c）合并到上游 main 基线 bdeec4e，保留导航能力和上游 VLA、DDS、Canvas 生命周期及 ACP 资源仲裁。只修改本仓；不修改 Driver，不部署或操作机器人。雨强的最终审核为真机导航验收，本轮本地验证不能代替该验收。

## 实施

1. 核对干净工作树及 PR head，fetch 上游并使用 merge 保留分支历史。
2. 按真实调用链解决构建、插件注册、端口绑定、地图预览与完成事件冲突；复用上游资源仲裁。
3. 执行 Core、ActuCore 与前端现有测试，补充合并边界回归；构建脚本在隔离环境验证。
4. 核对相关 README 与构建支持范围，范围化提交并推送当前 PR 分支；读回 PR head。

## 验收与限制

本轮交付为更新后的 PR 和可复现本地检查。镜像构建、北京部署、传感器与 Driver 兼容、连续导航及停车确认分别验证，不提前宣称通过。

## 合并决策

- Core 使用上游的物理资源与调用者顺序仲裁；保留导航暂停、恢复、等待、停止旁路和早到终态回放。导航未声明资源时保守互斥。
- JP5.11 镜像包含导航与 VLA 远端/mock；JP6.1 保留上游 VLA 本地推理构建入口，导航禁用且不接受覆盖。两个平台不混用 ROS/CUDA install space。
- DDS 使用上游主机隔离 profile。导航地图持久卷继续保留。
- README、导航 ACP 说明与 CONTRIBUTING 同步上述契约；真机部署文档暂不新增，因本轮尚未构建/部署更新镜像。

## 本地验证（2026-09-18）

- Core：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -k 'not test_progress_stream'`：988 passed，1 deselected，8 subtests passed。
- ActuCore：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../agent-core/.venv/bin/python -m pytest tests -q`：394 passed，2 skipped，66 subtests passed；包含 6 个隔离执行真实构建 Shell 的场景，Docker 用记录参数的替身，未启动容器或发布镜像。
- Web：`node --test 'agent-core/web/js/*.test.mjs'`：45 passed。
- `bash -n deploy/build_actucore.sh`、`git diff --check` 通过。
- Core 旧虚拟环境缺少上游 Peer 测试依赖 cryptography，已仅在该虚拟环境补齐，未修改项目依赖清单。
- 排除的上游 `tests/test_deploy_progress.py::test_progress_stream` 文件相对 main 无差异。直接用 `asyncio.run` 执行后确认在第 122 行调用 `progress.update('com配置…')` 缺少必填 message，引发 TypeError；未为消除该既有失败修改无关实现。
- README、CONTRIBUTING、导航 ACP 和 RViz DDS 说明已按合并结果复核。未构建新镜像，未部署，未执行真机运动；北京导航仍需配套 Driver 接口及现场验收。

## BOT 复审与修复

用户于 2026-09-18 要求重新发起 BOT review 并处理意见。对 `173084f` 请求 Core、ActuCore JP5.11、JP6.1 的 Build + Review；触发评论为 https://github.com/4paradigm/phanthymotus/pull/141#issuecomment-5724388958 。

流程：等待构建/测试/审查结果 → 对意见核验真实调用链 → 修复成立问题并补最小回归 → 同次提交代码与本事项文档 → 推送并再次请求复审。对不成立意见提供证据，不为消除报告而改错语义。沿用本仓范围，不部署、不驱动硬件、不修改 Driver。

第一轮：三个镜像构建成功，容器内 Core 988 / ActuCore 396 项通过。审查报告
https://github.com/4paradigm/phanthymotus/pull/141#issuecomment-5724471763 指出两项 P1。

- JP6.1 确认漏打包 `utils.security`：补齐目录复制、主入口导入检查和两个镜像的打包契约测试；Dockerfile 默认参数也与 JP6.1 构建入口一致。
- JP5.11 最终镜像只验证 segmented_controller，FAST-LIVO2/Nav2 的检查原在 builder：补齐最终阶段所有运行节点的 ldd、BT 插件 dlopen 与 Python 入口检查。先用真实构建确认平台镜像的库闭包，若缺库再按具体缺失补运行依赖，避免无证据安装开发包。
- Shell 检查在临时目录用替身 ldd 验证正常、缺库及命令失败三条路径；原生库检查仍以 BOT 实际构建为准。相关构建说明同步，不改 Driver 或真机部署步骤。
