# VOP 物体最近点相对深度与监控验收

vop（Video Object Perception）保留 YOLOv8-World 开放词汇检测。开启 `depth_enabled` 后，SAM 2.1 Tiny 根据同帧检测框生成掩码，YOLO26n-Depth 预测深度；每个物体输出掩码内最小的正数有限预测值。

**这是可见物体的预测掩码内最近相对深度，不是米制距离，也不是完整物体的真实最近点。** 不做逐帧归一化；跨帧绝对尺度未经验证。单像素噪声、掩码越界及漏分割会影响最小值，本版不以分位数替代最小值。不得用于米制制动阈值或证明避障安全。

## 卡片与契约

在 vop 的实例配置中启用 `depth_enabled`，再启动。生产默认值为 `false`；配置修改会停止受影响实例，需重新启动。关闭深度不加载 SAM/深度模型，原有类别扩展和识别字段保留。

- 输入：原有 `sensor_msgs/CompressedImage` JPEG。
- 输出 `{input_topic}/objects`：原有 `std_msgs/String` JSON。
- 输出 `{input_topic}/objects/preview`：`sensor_msgs/CompressedImage` JPEG；通过运行时 `info.topic_out` 发现，在监控面板选择该输出。
- 预览包含原始推理图像、检测框、半透明掩码、轮廓、最近点十字标记；图中编号对应右侧名称、置信度与相对深度。编号仅在当前帧有效。
- JSON 与预览使用同一帧、物体列表和 `sequence`；预览保留源 `header`，图像右侧附加图例不改变 JSON 的原图像素坐标。
- `timestamp` 继续表示发布时间（Unix 秒）；新增 `source_timestamp`、`frame_id` 保留输入头。`frame_age_s` 为从本进程接收该源帧到发布的单调时钟耗时，包含排队和推理，不冒充设备端采集延迟。

物体结果示例：

```json
{
  "id": 1,
  "name": "cup",
  "position": [-0.1, 0.2],
  "confidence": 0.91,
  "bbox_xyxy": [10, 20, 100, 200],
  "obstacle_depth": {
    "value": 1.2345,
    "unit": "relative",
    "method": "mask_min",
    "status": "ok",
    "pixel": [35, 75]
  }
}
```

无掩码为 `empty_mask`，无正数有限深度为 `invalid_depth`，模型失败为 `error`，数值和像素均为 `null`。模型失败保留检测结果，顶层 `status=depth_unavailable`，并给出 `error`；不回退到检测框或旧深度。输入中断超过 3 秒，输出空物体列表和 `stale: no input` 预览；停止发布一次 `stopped`，然后停止业务输出。推理未结束时 `stop` 返回 `stopping`，保留实例并在推理退出后自动清理，不谎报停止完成。

`info.state` 使用 Core 可判定的 `loading/running/error/idle`，细分阶段放在 `phase`；等待首帧不是 ready，输入过期为 `state=error, phase=stale`。

同一输入只允许一个 vop 实例，避免两个实例共用既有 `/objects` 输出路径。不同输入复用模型并串行推理；模型配置更新与推理互斥。

## 天轶 2.0 镜像测试

在 draft PR 评论 `/request_bot_review perception jetson-5.11 jetson-6.1`，bot 分别构建两个 JetPack 版本并 review。只使用与设备 JetPack 匹配、且对应 PR 当前 head SHA 的成功构建镜像；不要使用旧提交的构建结果。Jetson Dockerfile 安装锁定的深度 API 依赖，保留 NVIDIA torch/torchvision 与系统 OpenCV，构建时检查导入和 CUDA torch；模型推理及真机性能仍由设备测试验证。

由测试人在设备上更新 Perception 镜像后，从实际话题列表选择相机 `CompressedImage` 输入，启用 vop 实例的 `depth_enabled` 并启动。在监控面板选择该输入对应的 `/objects/preview`，按照下文人工验收步骤核对图像、框、掩码与相对深度。首次启用需能访问权重下载地址，或预先将校验一致的两个权重放入持久化 `/models/vop-depth`；失败必须显示 N/A，不能算通过。

本 PR 不更改 Driver 或动作链路。记录设备 JetPack、镜像完整标签/摘要、PR SHA 和验收结果；当前仍待天轶 2.0 测试。

## 开发机回放

以下是独立本地开发环境，不执行真机部署。命令从业务仓根目录运行。Core 镜像需已在本机存在；本次复用 `local/phanthy-motus/core:release.260904.4d38a2c`，将当前源码与前端只读挂载覆盖旧实现，不挂载用户配置或 Docker socket。

```bash
# CPU 回放镜像；首次构建和权重下载不计入五分钟人工验收。
docker build -f perception/Dockerfile.vop-replay \
  -t local/phanthy-motus/perception:vop-depth-replay perception

docker network create vop-depth-replay

docker run -d --name vop-depth-dev --network vop-depth-replay \
  -p 127.0.0.1:18720:15720 \
  -e ROS_DOMAIN_ID=73 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e AGENT_CORE_URL=http://127.0.0.1:9 \
  -v vop-depth-models:/models \
  local/phanthy-motus/perception:vop-depth-replay

docker run -d --name vop-depth-ui --network vop-depth-replay \
  -p 127.0.0.1:18778:15678 \
  -e ROS_DOMAIN_ID=73 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -v "$PWD/agent-core/src:/work/src:ro" \
  -v "$PWD/agent-core/web:/work/web:ro" \
  --entrypoint bash local/phanthy-motus/core:release.260904.4d38a2c \
  -c 'source /opt/ros/humble/setup.bash; source /ros_ws/install/setup.bash; .venv/bin/python -m uvicorn start:app --app-dir src --host 0.0.0.0 --port 15678'

docker exec -it vop-depth-dev bash -c \
  'source /opt/ros/humble/setup.bash; python3 tools/replay_vop.py bus \
   --output /tmp/vop-evidence --duration 300 \
   --core-url http://vop-depth-ui:15678 --mcp-url http://vop-depth-dev:15720/mcp'
```

`AGENT_CORE_URL` 的禁用端口只用于隔离本机原有自动注册（它假定所有容器共享 localhost）；回放脚本通过 `--core-url/--mcp-url` 显式注册真实地址。脚本保留已有画布卡片，只追加 `vop-replay`，不会强占其他编辑者。未传这两个参数时仅发布/采集 DDS，适合已有实例。

`bus` 使用安装包自带的真实示例图片；不将图片、人脸或模型权重提交到仓库。也可把有使用权限的图片或视频只读挂载进容器，把 `bus` 换成该文件路径；视频循环回放。`--duration 0` 持续运行至 Ctrl-C。结束后 vop 会显示输入过期；再次运行回放即可恢复。

已有同名验收容器时复用它们，不覆盖别人的容器。源码只读挂载的开发环境，把命令中的 `tools/replay_vop.py` 改成实际挂载路径下的脚本。

## 五分钟人工验收（必须由验收人确认）

1. 打开 `http://localhost:18778`，切换 **监控**，查看 `/vop_replay/camera/objects/preview` 面板，放大面板使图例可读。
2. 核对原图、每个物体框、分割轮廓、编号、名称、置信度和深度是否一一对应；核对十字标记位于对应物体的预测掩码内。
3. 使用近远物体、遮挡场景和物体进出画面的自有视频；确认未用检测框内的其他区域代替物体掩码。发现掩码包含遮挡物时记为模型质量问题，不以数值测试通过掩盖它。
4. 对照同帧 JSON 的 `sequence`、`source_timestamp`、物体编号、`obstacle_depth.value` 与像素坐标；检查预览明确标注 `RELATIVE DEPTH / UNCALIBRATED` 和帧处理耗时。
5. 停止回放，确认面板在输入中断后显示 `stale: no input`，清除旧物体与深度；恢复回放后恢复结果。模型失败时必须看到 `depth_unavailable`、N/A 和 JSON 错误信息。

保存 `docker cp vop-depth-dev:/tmp/vop-evidence ./vop-evidence` 的结果，并自行录制浏览器短视频。目录含结构化逐帧记录、最近预览及至多 20 个源帧预览；异常状态可能覆盖同源帧截图，以 `results.ndjson` 和录屏核对过程。证据仅本地保存，不加入 Git。

验收记录必须填写：验收人、时间、源码 SHA/本地 diff、镜像与模型版本、素材来源、以上各项通过/不通过、截图/录屏路径及问题。**自动测试、API/WS 通过和 agent 查看输出图像，均不能代替用户在监控面板的人工验收。**

## 依赖与验证

Ultralytics 固定 `8.4.144`；CPU 回放验证使用 torch `2.2.2`、torchvision `0.17.2`、NumPy `1.26.4`。默认 `ros:humble` 基础镜像构建路径尚需独立验证；本次 Dockerfile 实测复用了本地已存在的 ROS/torch 基础镜像：

```bash
docker build --build-arg ROS_BASE_IMAGE=local/phanthy-motus/perception:release.260817.e7d5abb-navigation \
  -f perception/Dockerfile.vop-replay -t local/phanthy-motus/perception:vop-depth-replay perception

docker exec vop-depth-dev bash -c 'cd /work; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest tests/test_vop_depth.py tests/test_bundle_dispatch.py tests/test_shared_utils.py \
  -q -p pytest_mock -o cache_dir=/tmp/pytest-vop'
```

新增权重在 `/models/vop-depth` 缓存，下载完成且校验成功后才原子安装，加载前校验 SHA-256：

| 权重 | SHA-256 |
|---|---|
| `yolo26n-depth.pt` | `befed1b8561d8b2eaa66274b070cdfc9c44853bda6d6d62a04759f9383af74e9` |
| `sam2.1_t.pt` | `3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3` |

来源为 Ultralytics assets `v8.4.0` release，固定校验值防止同名资产更新后静默改变模型。原有 YOLO-World/CLIP 加载逻辑继续使用已有缓存约定。

Ultralytics 代码与 YOLO 权重适用 AGPL-3.0 / Enterprise；SAM 2.1 上游为 Apache-2.0，CLIP 为 MIT。本次仅开发与仿真验证，不据此授权闭源分发。官方接口参考：[YOLO26-Depth](https://docs.ultralytics.com/tasks/depth/)、[SAM 2](https://docs.ultralytics.com/models/sam-2/)。

Jetson CUDA wheel、真机性能和米制标定未验证；不要在生产 Jetson 上直接安装本 CPU 回放栈。回滚只需关闭实例 `depth_enabled` 后重启该实例；完整移除此功能可回滚本次业务仓改动。

独立 GPU 容器接入已有 Core 时，设置 `AGENT_CORE_URL`、Core 可访问的 `MCP_ADVERTISE_URL` 和唯一的 `MCP_SERVER_NAME`。Core 按服务身份去重；多个 Perception 容器不能共用默认身份，否则会合并已有服务。未设置这些变量时保留原来的本机注册行为。

本次另在 Linux amd64 / A100 / Torch 2.2.2+cu121 上验证：52 项测试通过，Core 收到 20 帧实际仿真相机的标注输出，接收至发布帧龄中位约 79 ms；独立回放断流后显示 stale 并清空物体。当前模型把仿真红盘识别为 frisbee，色块漏检，尚未通过人工语义验收；这个性能数字也不代表真机端到端延迟。
