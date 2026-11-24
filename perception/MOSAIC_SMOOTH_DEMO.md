# Mosaic Smooth Demo（独立演示 Card）

> Experimental / consent required / NOT anonymization.
> 这张 Card 只是流畅马赛克图像演示，不是 `privacy_view` 或 `privacy_view_v2` 的替代品，也不能保证人脸匿名。

## 功能与边界

- 独立 `mosaic_smooth_demo` processor，支持多实例、`start/stop/info/config` 生命周期。
- 输入 `image/jpeg`，输出一个处理后 JPEG 和一个 `data/json` 状态话题，话题名包含实例 ID 的 SHA-256 前缀。
- 最新帧覆盖队列；默认每四帧运行一次 YuNet，间隔帧用短时 LK 光流更新马赛克区域。
- 人脸可信时局部 24px 马赛克；检测或跟踪不可靠时使用整帧 40px 粗马赛克，而不是直接显示原 JPEG。
- 处理异常或收到畸形、过期 JPEG 时停止输出新图像。无本地网页、无图像录制或落盘、无运动控制。
- 默认 `enabled: false`；操作员必须明确勾选 `consent_acknowledged: true`，否则 `start` 拒绝执行。
- 状态固定声明 `privacy_guaranteed: false`，漏检人脸、新进入画面者以及极端侧脸仍可能被辨认。
- 不修改任何现有 `privacy_view` / `privacy_view_v2` 代码，也不添加依赖项。

## 部署条件

复用 Perception 镜像现有 OpenCV、NumPy、ROS2 和 `sensor_msgs/msg/CompressedImage`。OpenCV 缺少可运行的 `FaceDetectorYN` 时，使用 Perception 共享 ORT 子进程中的 CPU 会话运行 YuNet 2023mar。该路径的检测输入长边限制为 640 像素，检测框会映射回马赛克输出画面。
模型由 Perception 在运行时准备。默认 `model_path` 为空，首次经知情同意启动时，`ensure_verified_bundle()` 会验证缓存并在缺失或损坏时自动下载。给隔离测试容器挂载持久化、可写的 `/models`，缓存文件为 `/models/mosaic_smooth/face_detection_yunet_2023mar.onnx`。旧路径 `/models/mosaic_smooth/yunet.onnx` 上已有的有效文件也可以直接复用。模型文件随运行时缓存提供，镜像构建流程无需下载权重。

权重来自 OpenCV Zoo 的固定提交，官方 Hugging Face 副本也固定到提交，两处文件逐字节一致。文件大小为 **232,589 字节**，SHA-256 为 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。转存或分发时随权重保留 MIT 版权与许可全文。 [OpenCV Zoo 权重](https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) · [官方 Hugging Face 副本](https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx) · [YuNet MIT 许可](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/LICENSE)

- **自动下载。** MCP `info` 和 ROS 状态话题中的 `model_fetch` 提供阶段、传输字节、百分比和重试次数。并发实例共享文件锁，分别报告等待、校验、缓存命中或下载完成。100% 表示传输进度，下载通过大小和 SHA-256 校验并原子写入缓存后才报告完成。最终失败区分 `model_download_failed` 和 `model_verification_failed`，再次 `start` 可重试。停止时立即关闭图像发布和进度更新，正在等待的文件锁或网络读取按下载器生命周期结束。
- **受限网络。** 可在容器中设置 `MOSAIC_SMOOTH_MODEL_BASE_URL`，以已发布且部署可达的 COS 镜像目录替换两个默认来源。目录应提供 `face_detection_yunet_2023mar.onnx` 和对应的 MIT 许可证。镜像使用同一组校验值。
- **离线部署。** 将校验过的模型和许可证放入挂载目录。可通过 `plugins.mosaic_smooth_demo.model_path` 或 `MOSAIC_SMOOTH_YUNET_MODEL` 指定只读模型文件，配置项优先。显式路径只做本地校验，缺失、大小错误或哈希错误均拒绝启动。旧配置中若填了 `model_path`，清空该项并取消设置模型路径环境变量即可使用自动下载。

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
- 提交 `969189d` 的插件在已有 ARM64 Perception 镜像内通过 8 项单元测试和 1 项真实 ROS2 合成输入、双话题输出与停止测试（独立网络，无真实摄像头）。
- 本次下载接入的 Jetson 镜像构建、Canvas 渲染、真人局部马赛克 FPS、隐私漏盖率、P95/P99 延迟和资源长测仍待验收。
- 插件不调用 `cv2.setNumThreads()`，避免修改共享进程级 OpenCV 线程配置；因此正式容器性能需重新实测。

## 复现

ROS2 合成输入测试应使用独立 DDS 域及仅限回环的传输配置。

```bash
PYTHONPATH=perception python3 -m pytest perception/tests/test_mosaic_smooth_demo.py perception/tests/test_mosaic_smooth_model.py -q
TEST_MOSAIC_YUNET=/models/mosaic_smooth/face_detection_yunet_2023mar.onnx PYTHONPATH=perception python3 -m unittest discover -s perception/tests -p test_mosaic_smooth_ros_integration.py -v
```

**只有在获得所有出镜人员同意的演示环境才能启动。** 不得将本卡当作严格隐私遮挡或正式安全功能交付。
