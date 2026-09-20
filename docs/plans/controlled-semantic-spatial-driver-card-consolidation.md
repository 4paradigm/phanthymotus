# 导航消费端适配 Driver 卡片合并

## 范围和基线

用户指定 Driver 合并四张独立卡片；本 session 只修改导航消费端说明与契约提示，
不修改 Driver。新增 lidar_imu 承接 LiDAR IMU，原机身 imu 保持原用途。
本地尚未找到新的 Driver 实现；以用户提供的能力映射、既有 topic/schema 为基线，
不声称新 Driver 已完成或联调通过。

## 实施

- lidar_cloud 的标准 PointCloud2 → lidar；lidar_imu → imu。
- camera_rgb 的 RGB PSE1 → rgb；camera_depth 的 Depth PSE1 → depth_frame。
- 保持消费端 ROS 类型、schema、frame、时钟和 freshness 约束。
- 消费端已支持按画布输出索引解析多输出来源和自定义 topic，无需添加旧卡片名兼容分支。
- 更新输入描述、README 和旧画布迁移步骤。
- 用 Core 现有解析函数验证多输出选择、已保存 topic 与实时 topic 两条路径；不改 Core 生产逻辑。

## 验证

- Core：`./agent-core/.venv/bin/python -m pytest agent-core/tests/test_start_project_resolution.py -q`，20 passed；测试自身将 DB_PATH 隔离到临时目录。
- ActuCore：`cd actucore && ../agent-core/.venv/bin/python -m pytest tests/test_navigation_plugin.py tests/test_fast_livo2_contract.py tests/test_vln.py -q -rs`，88 passed、1 skipped（Linux procfs）、26 subtests passed。
- `git diff --check` 通过；已核对统一卡片及 mapping README 与输入描述，按能力选择输出并明确旧画布迁移。
- 真实 Driver tools/list、浏览器及机器人联调未验证。未提交、推送或部署。

## 本轮提交与北京部署

用户已授权提交和部署，目标北京 G1。先范围化提交并推送 PR #141，触发 BOT 构建及复审。
部署前只读核验设备身份、当前镜像、Driver 接口、画布占用和任务状态；通过后部署已构建镜像。
部署结果与本地测试、BOT 审查和现场运动验收分别记录。
