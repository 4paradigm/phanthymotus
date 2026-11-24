# hand_landmarks：手部关键点识别

这是 Perception 的视觉处理卡。输入一幅相机画面，识别一只手的 21 个二维关键点，输出五指弯曲量和绿色骨架预览。它不写电机。L20 Lite 的标定、使能与执行仍归独立的 `hand_visual_follow` 驱动卡。

## 从 ActuCore 版迁移

1. 停止画布项目，移除旧的 ActuCore `handdrive` / `hand_landmarks` 实例。避免新旧实例同时发布同一话题。
2. 部署含此改动的 **Perception** 镜像；保留 L20 Lite 驱动服务。以前保存的画布节点绑定的是 ActuCore 的 MCP 服务，即便工具名相同也要重新添加。
3. 添加 Perception 的 `hand_landmarks`。将相机 `image/jpeg` 接到它的输入，将它的 `control/hand_flex`（五指弯曲量指令流）接到 `hand_visual_follow` 的输入。另一个 `image/jpeg` 输出是绿色骨架预览，仅供查看、不参与控制。
4. 设置 `hand` 与驱动卡的 `bound_side` 一致；左手机械手用 `Left` 和 `left`。RTMPose 整图推理不测左右手，输出明确标记为 `handedness_source=configured`。

默认保留 `/actucore/handdrive/keypoints` 和 `/actucore/handdrive/view` 两个 ROS 话题名，前缀仅为兼容旧订阅方。JSON 协议仍是 `motus.hand/1`：`landmarks`、`flexions`、`score`、`handedness`、`handedness_source`、`stamp_ms`、`obs_stamp_ms`。旧版可选的 `control/joint` 不再发布；若已有项目直接依赖该端口，不能用本卡直接替换。

## 在另一台 Jetson 复用

- 在主仓 `4paradigm/phanthymotus` 提交 `perception/` 的改动。JetPack 6.1 的 PR 构建指令为 `/request_bot_review perception jetson-6.1`。取 bot 返回的镜像和部署说明。
- 沿用 Perception 的 Dockerfile 和 `deploy/service.yml`：安装插件、工具、配置，启用 NVIDIA 容器运行时，挂载 `/models`，连接 ROS/DDS。目标机需有兼容的 JetPack、GPU 运行时和相机。无需从 Mac 复制 Python 文件或单独安装 ActuCore。
- 默认 RTMPose 模型首次使用时下载到 `/models/hand`，由仓库下载器按 TensorRT 版本选择、校验并缓存。离线机器需预置匹配的缓存；不要混用不同 JetPack 的引擎。
- 验收先看画布的绿色骨架，再检查 `keypoints` 的观测时间戳和五指数据。摄像头中没有手时不应继续发布新的手姿。最后在同一台机器验证下游驱动卡的真机跟随。

离线回归：在 `perception/` 下运行 `python3 -m pytest -q tests/test_hand_landmarks.py tests/test_bundle_dispatch.py`。使用 ROS 测试端点；不代替 Jetson 上的 TensorRT 加载、DDS 互通或机械手验收。
