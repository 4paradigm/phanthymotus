# Mosaic Smooth Demo（独立演示 Card）

> Experimental / consent required / NOT anonymization.
> 这张 Card 只是流畅马赛克图像演示，不是 `privacy_view` 或 `privacy_view_v2` 的替代品，也不能保证人脸匿名。

## 功能与边界

- 独立 `mosaic_smooth_demo` processor，支持多实例、`start/stop/info/config` 生命周期。
- 输入 `image/jpeg`，输出一个处理后 JPEG 和一个 `data/json` 状态话题，话题名包含实例 ID 的 SHA-256 前缀。
- 最新帧覆盖队列；默认每四帧运行一次 YuNet，间隔帧用短时 LK 光流更新马赛克区域。
- 人脸可信时局部 24px 马赛克；检测或跟踪不可靠时使用整帧 40px 粗马赛克，而不是直接显示原 JPEG。
- 异常、畸形/过期 JPEG 停止输出新图像。无本地网页、无录制/文件写入、无运动控制。
- 默认 `enabled: false`；操作员必须明确勾选 `consent_acknowledged: true`，否则 `start` 拒绝执行。
- 状态固定声明 `privacy_guaranteed: false`，漏检人脸、新进入画面者以及极端侧脸仍可能被辨认。
- 不修改任何现有 `privacy_view` / `privacy_view_v2` 代码，也不添加依赖项。

## 部署条件

复用 Perception 镜像现有 OpenCV `FaceDetectorYN`、NumPy、ROS2 和 `sensor_msgs/msg/CompressedImage`。
YuNet 2023mar 模型需要在隔离测试容器中预先作为只读文件提供，默认路径 `/models/mosaic_smooth/yunet.onnx`；不会自动下载、创建或修改模型。
期望模型 SHA-256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。
可以使用 `plugins.mosaic_smooth_demo.model_path` 或环境变量 `MOSAIC_SMOOTH_YUNET_MODEL` 指定路径。缺失或哈希不符将拒绝启动。
**PR 合并或构建镜像不会自动把模型文件装进镜像。** 必须由操作者在独立测试容器中只读挂载。

## MCP 使用方式

1. 在私有的测试配置中将 `plugins.mosaic_smooth_demo.enabled` 改为 `true`，不要直接修改共享正式 Perception。
2. 将同意参与测试的摄像头 JPEG 输出连接至此 Card。打开设置并显式确认 `consent_acknowledged=true`。
3. 通过 `config`、`start`、`info`、`stop` 操作独立实例。连接图像和状态两个输出口进行联调。

```json
{"action":"config","instance_id":"demo_front","config":{"consent_acknowledged":true,"output_width":640,"detect_every_frames":4}}
{"action":"start","instance_id":"demo_front","input_topic":"/robot/camera/color"}
{"action":"info","instance_id":"demo_front"}
{"action":"stop","instance_id":"demo_front"}
```

## 默认配置

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `consent_acknowledged` | `false` | 必须显式同意才能启动 |
| `output_width` | 640 | 输出宽度 320–960 |
| `detect_every_frames` | 4 | YuNet 检测间隔 1–12 帧 |
| `confidence_threshold` | 0.28 | 检测阈值 0.1–0.9 |
| `max_faces` | 6 | 1–8；超额进入整帧马赛克 |
| `pixelate_block` | 24 | 人脸局部马赛克块 |
| `fallback_block` | 40 | 整帧回退马赛克块 |
| `jpeg_quality` | 72 | JPEG 编码质量 |

## 性能与验收状态（2026-11-21）

- 独立 V3 原型在 Bumi Orin 上用实际 ROS2 摄像头测得约 21.33–29.98 FPS（不同短时采样），但当时**未检测到可信真人脸，输出主要是整帧粗马赛克**。
- 60 帧离线合成自测约 49.93 FPS，不是正式 Card 或 Canvas 的性能指标。
- **本 PR 插件版本**在已有 ARM64 Perception 镜像内通过 8 项单元测试 + 1 项真实 ROS2 合成输入/双话题输出/停止生命周期测试（独立网络，无真实摄像头）。
- 正式新镜像构建、MCP 发现、Canvas 渲染、真人局部马赛克 FPS、隐私漏盖率、P95/P99 延迟和资源长测，仍未验收。
- 插件不调用 `cv2.setNumThreads()`，避免修改共享进程级 OpenCV 线程配置；因此正式容器性能需重新实测。

## 复现

```bash
PYTHONPATH=perception python3 -m unittest discover -s perception/tests -p test_mosaic_smooth_demo.py -v
TEST_MOSAIC_YUNET=/models/mosaic_smooth/yunet.onnx PYTHONPATH=perception python3 -m unittest discover -s perception/tests -p test_mosaic_smooth_ros_integration.py -v
```

**只有在获得所有出镜人员同意的演示环境才能启动。** 不得将本卡当作严格隐私遮挡或正式安全功能交付。
