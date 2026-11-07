# PiPER 物体目标达成验证 Card

Tool：`piper_object_goal_verifier`；类型：Perception processor。专门服务松灵 PiPER 的推动、抓取放置及其他物体位置调整结果验证，也支持用人工移动物体做独立验收。**不执行运动，不导入 PiPER SDK/CAN/URDF，不依赖旧 Visual Push。**

相机、Topic、ROS域、PiPER安装标识、桌面/工具参数均不写死。首次验收可用固定相机，但不是唯一部署场景。移动相机需要逐帧拍摄几何与有效标定；证据不足时返回 INCONCLUSIVE。

## 能回答什么

- **PASS**：可靠的新观测、合理身份关联，证明所选物体的**可见 bbox 中心**满足目标区域及容差。
- **FAIL**：相同可靠性条件下，证明该中心仍在目标区域之外。
- **INCONCLUSIVE**：没有足够证据，例如目标丢失、遮挡后身份不明、同外观多物体、低可靠度、旧帧、不同相机/视角、无有效标定。

目标支持圆和轴对齐矩形，单位支持 pixels、normalized `[0,1]`。不需要机器人真实空间标定即可使用图像坐标；不会把图像位移解释为物理毫米。`plane_mm` 是可选的实测 PiPER 平面映射，需明确物体参考平面、独立检查点、误差界、相机/视图/尺寸/畸变/有效期绑定；不是通用三维距离或抓取成功证明。

## 输入与关联

1. **PiPER 现有标准 ROS 图像**：订阅 `sensor_msgs/Image` 的 bgr8/rgb8，或 CompressedImage JPEG。使用原 header 采集时间、复制像素，验证 frame_id/相机配置。CPU HSV 后端无需 GPU 或 VOP，能检测多个配置颜色目标；类别来自配置，可靠度是实测 mask 填充率，明确区别于模型概率。
2. **官方 VOP**：复用 `recognize_by_url`、原 bbox 解码和颜色结构。图像以本 Card 的随机短时内存 URL 供同主机 VOP 读取，不假定容器共享路径，不向其他服务上传/清理文件。结果 source 必须匹配该 URL，原采集时间不变。VOP 冷启动/延迟导致过期时拒绝。
3. **规范观测**：MCP 或 `std_msgs/String` 接收 `motus.piper-object-observation/1`。外部 MCP 提交默认关闭，须 owner 配置可信生产者；ROS图像来源不允许 MCP 调用者凭空替换检测。可以重新引用由本 Card 已经接收且散列相同的观测。

官方 VOP 实时 `/objects` 的 timestamp 是**发布时间**；单张图片回复也不含采集时间。不能直接把它们包装成新鲜采集帧，不能用 latency_ms 反推时间。缺帧元数据拒绝，不能补值。

多个物体可以按类别、颜色、检测标识或经验证的持续 track_id 选择。检测ID只在某一 observation_id 内有效。跨操作默认只接受同类别/颜色的唯一候选、有限位移和外观面积一致性；同类同外观多物体不靠最近点强行关联。参与处理的观测中目标曾消失后再次出现，默认要求新基准。只有 owner 明确确认来源提供持续身份、配置 identity_record，并且 track_epoch 一致时才使用 track_id；stock VOP/measure 没有这个承诺。移动视角的毫米验证也要求这一身份依据。

复用 `measure.geometry.normalise_bbox`，不把其 Stabilizer 当成跨操作跟踪器。边缘截断、过期、重放和最新来源错误均不能用旧缓存掩盖。

## 操作流程

`start_session → capture_before → 外部操作 → mark_operation_complete → verify_after → get_result/export_session`。

`mark_operation_complete` 只记录调用方声明操作已经结束的时间屏障，不执行/证明机械臂操作。后观测须在屏障之后采集。每次验证用新 request_id；相同 request_id 和相同参数返回同一历史结果，不重新采集。相同ID换参数拒绝。reset退役旧token/marker，迟到回复不能写入新会话；关闭/停止释放自己的资源。

无论 PASS/FAIL，`physical_motion_confirmed` 都是 false：视觉只证明该观测时刻的结果条件，不证明是谁移动物体，也不保证之后物体仍在区域内。

## 启动与 Dashboard

推荐独立只读服务，使用自己创建的环境和端口，原 PiPER owner 保持原状：

```bash
PYTHONPATH=perception python perception/tools/goal_verifier_cli.py template --mode color --output /own/run/config.json
# 按 ONSITE.md 填入真实来源，不直接启动空模板
PYTHONPATH=perception python -m plugins.goal_verifier.server --config /own/run/config.json --port 15785
```

在官方 Core 注册为 `category: perception`。新分支对 sidebar 增加只读 processor 工作台按钮，复用现有 `openToolConsole`、sandbox、MessageChannel 和 MCP API，**无需开启整个项目**。`x-workbench.readOnly` 是入口元数据，不是安全权限；原认证和宿主能力限制保留。普通工具和 actuator 不增加此入口。

也可在授权维护的 Perception 配置中启用插件；默认 disabled。现场不一定有兼容前端/Perception/VOP，doctor 会检查，不据端口号推断。部署包生成不等于现场安装成功。

The official image copies this directory to `/work/plugins/goal_verifier`.
The existing Perception entrypoint reads `CONFIG_PATH` (default `/work/config.yaml`).
An operator-provided full config may enable `plugins.piper_object_goal_verifier.enabled`
and supply its source bindings. Disabled-by-default means an unchanged image configuration
does not advertise this tool. Enabling advertises it; source subscriptions still require an
explicit `start`. The existing default-enabled Resource Center card list is unchanged.

## Validation and deployment limits

Run `GOAL_TEST_PYTHON=/own/python bash perception/plugins/goal_verifier/deploy/run-tests.sh`.
The browser harness `perception/tools/goal_verifier_e2e.py --full-core-lifespan --ros-input`
uses the actual Core/MCP/native workbench and ROS2 transport with explicitly synthetic images.
Without `--full-core-lifespan` it uses an HTTP-only lifecycle; these are different test scopes.
ROS/image contracts and official VOP decoding use real source; inference substitutes in unit tests
are not evidence of official TensorRT inference.

Actual camera before/after acceptance, official VOP inference and Jetson image execution remain
unverified. A real photograph can exercise the CPU detector, but absent original camera/capture
metadata it must remain inadmissible for trusted goal verification. Image coordinates are not
physical displacement. The default unique-class/color, bounded movement and area association is
conditional, not continuous identity proof across unobserved intervals.

See [API.md](API.md), [ONSITE.md](ONSITE.md) and [DIFFERENCES.md](DIFFERENCES.md).
No robot SDK, CAN owner, old Visual Push code or other unmerged Card is required.
