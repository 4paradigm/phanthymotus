# 导航卡片公开输出收敛

## 范围

根据雨强反馈，ControlledSemanticSpatial 仅保留原第 1、4 个输出。
公开输出依次为 `map_view`、`motion_sequence`，移除 `status`、
`collection_status`、`costmap`。修改统一卡片契约；停发纯展示的 collection_preview，不删除内部必需 ROS 流或数采功能。
`motion_sequence` 的端口、ROS topic、schema、launch 参数与实现命名全部统一；
QoS、频率、TTL、消息字段及停车语义不变。Driver 由同事同步修改，不兼容旧 schema。

## 实施与兼容

- 修改统一 `topic_out`；`tools/list` 与 `info` 共用该契约。
- 更新模块 README 和已有端口测试，验证自定义 namespace 及 Driver wire contract。
- 旧画布需停止智能控制、刷新定义、移除旧输出连线，并把运动输出重接到新的第 2 个端口。
- 不修改 Driver、Core、部署脚本或已暂缓问题；不执行机器人操作。

## 停发纯展示数据

- 移除数采 JPEG publisher、五路预览订阅和预览 worker 创建；不再渲染预览或导出进度图片。
- 保留 CollectionPostprocessManager、raw 状态订阅、JSON 诊断和 receipt 导出链。
- `status` 由 mapping/backend.py 等待命令回执；停发会造成操作超时。
- `/global_costmap/costmap` 由 planner_command_node.py 校验目标占据与新鲜度；停发会拒绝导航。
- 上述两路仍为内部必需发布，不能仅为去掉展示而关闭。
- 更新 mapping 内部契约，不再声明不存在的 preview 输出；补充真实控制器初始化的 ROS 替身测试，断言无预览 publisher/订阅/worker 且诊断与导出委托仍工作。

## 验证

- `./agent-core/.venv/bin/python -m pytest actucore/tests/test_navigation_plugin.py -q`：37 passed、1 skipped、7 subtests passed。
- `cd actucore && ../agent-core/.venv/bin/python -m pytest tests -q`：436 passed、2 skipped、72 subtests passed。
- `git diff --check` 通过。
- 已复核 ActuCore、统一卡片、mapping、planning README，更新公开输出及旧画布迁移说明；内部协议说明继续保留原 ROS topic/schema。
- 未提交、推送、构建镜像、部署或真机验收；未执行浏览器端到端验证。本地测试不替代上述证据。

停发扩展验证：

- 定向 collection postprocess + FAST-LIVO2 contract：13 passed、1 skipped、2 subtests passed。
- ActuCore 全量：`cd actucore && ../agent-core/.venv/bin/python -m pytest tests -q -rs`，437 passed、2 skipped、72 subtests passed。
- 两个跳过原因：本机缺 OpenCV、需要 Linux procfs。
- 已回读实施计划并核对 README：明确 preview 停发、status/costmap 内部依赖保留；无部署、浏览器端到端或真机证据。

## 本轮提交与北京部署

此前端口收敛版本已提交并部署北京 G1。用户最新要求本地与现场验收通过后再申请 BOT review；
本次完整协议改名先本地验证，待同事的 Driver 配套后才可部署。
部署前只读核验设备身份、当前镜像、Driver 接口、画布占用和任务状态；通过后部署已构建镜像。
部署结果与本地测试、BOT 审查和现场运动验收分别记录。
