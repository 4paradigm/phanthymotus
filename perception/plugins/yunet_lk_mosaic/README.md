# YuNet + LK 实时人脸马赛克 Card

MCP 名称 `yunet_lk_mosaic`。这张 processor Card 接收一路摄像头 JPEG，用 YuNet 定位人脸并用 LK 光流跟踪，在原画面上叠加局部像素块。用途是实时视觉效果，**不承诺匿名或严格隐私保护**。测试和使用前须取得出镜人员同意。

## 启动和连接

1. 随项目提供的配置为 `plugins.yunet_lk_mosaic.enabled: true`。Card 默认注册到现有 Perception，公共 Canvas 可以发现它。默认 `consent_acknowledged: false`。只有确认出镜人员同意并调用 `start` 后，Card 才会订阅摄像头、加载模型和开始推理。源码运行且 ROS2 环境已加载时，从仓库根目录执行 `CONFIG_PATH=/path/to/perception.yaml PYTHONPATH=perception python3 perception/main.py`。
2. 在 Canvas 连接相机的 `image/jpeg` 输出，打开实例配置并设置 `consent_acknowledged=true`。输入可来自已经接入项目的通用摄像头、浏览器摄像头或兼容的 ROS2 视频源。
3. 使用 Card 的启动操作。用 `info` 查看模型和画面是否就绪，连接 `image/jpeg` 预览和 `data/json` 状态输出。停止实例后可修改配置并重新启动。

也可通过 Perception 的 MCP HTTP 接口操作。以下是对 `tools/call` 的 `arguments`，`name` 为 `yunet_lk_mosaic`。

```json
{"action":"config","instance_id":"front","config":{"consent_acknowledged":true,"target_fps":24,"publish_delay_ms":300}}
{"action":"start","instance_id":"front","input_topic":"/camera/color"}
{"action":"info","instance_id":"front"}
{"action":"stop","instance_id":"front"}
```

```sh
curl -sS http://127.0.0.1:15720/mcp -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

ROS2 输入和图像输出使用 `sensor_msgs/msg/CompressedImage`，保留原始 `header.stamp` 与 `frame_id`。接受 `jpeg`、`jpg` 和 ROS 的 `bgr8; jpeg compressed bgr8` 等 JPEG 格式写法。源需要递增时间戳，并与 Perception 使用同一 ROS 时钟。状态使用装有 JSON 的 `std_msgs/msg/String`，图像与状态话题由 `info` 返回。每实例单独管理订阅、缓存与输出，最多同时运行 4 个实例。

## 模型

复用 Perception 运行环境的 OpenCV、NumPy、ROS2 和 ONNX Runtime。优先使用 OpenCV `FaceDetectorYN`。如果 OpenCV 不支持该模型，使用项目共享 ORT 子进程的 CPU 会话，检测输入长边限制为 640px 并将框映射回输出画面。

默认使用项目下载器，在 `/models/yunet_lk_mosaic/` 校验和缓存模型。缓存缺失时，从固定版本的 OpenCV Zoo 或官方镜像下载。权重文件为 `face_detection_yunet_2023mar.onnx`，大小 232589 字节，SHA-256 为 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`。需要离线部署时，将已校验的模型与 MIT 许可放在容器可读的挂载目录，用 `plugins.yunet_lk_mosaic.model_path` 或 `YUNET_LK_MOSAIC_MODEL` 指定。受限网络可通过 `YUNET_LK_MOSAIC_MODEL_BASE_URL` 指定含同一文件的镜像目录。启动加载和推理失败会在状态中报错。 [OpenCV Zoo 权重](https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) · [官方镜像](https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx) · [MIT 许可](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/LICENSE)

## 配置

YAML 提供部署默认值，MCP `config` 可按实例覆盖。非法值返回具体字段错误。

| 参数 | 默认值 | 支持值 |
| --- | ---: | --- |
| `enabled` | `true` | 在现有 Perception 中注册 Card，供公共 Canvas 发现 |
| `model_path` | 空 | YAML 指定已校验模型，空时走环境配置或下载 |
| `consent_acknowledged` | `false` | 开启后允许启动 |
| `target_fps` | 30 | 15 / 20 / 24 / 30，限制处理速率 |
| `publish_delay_ms` | 0 | 0 / 200 / 300 / 500，收到帧至最早可发布的时间 |
| `miss_compensation` | `true` | 开关，控制检测漏帧的短时补偿及历史纠错 |
| `adaptive_detection` | `false` | 开关，默认保留旧版固定节奏 |
| `detect_every_frames` | 4 | 1–5，固定模式中每 N 个实际处理帧检测一次 |
| `confidence_threshold` | 0.28 | 0.1–0.9，直接传给 YuNet |
| `stable_expansion` | 0.35 | 0–1，稳定时在每边扩展对应框宽或高的比例 |
| `unstable_expansion` | 0.50 | 0–1，且不低于稳定扩框 |
| `max_buffer_frames` | 20 | 1–32，另有每实例 64 MiB 图像缓存上限 |
| `output_width` | 640 | 320–960，按原宽高比缩放 |
| `max_faces` | 6 | 1–8，检测超额时记录状态并优先处理高置信框 |
| `jpeg_quality` | 72 | 35–95 |
| `pixelate_block` | 24 | 8–64，输出像素 |

## 处理细节

- 正常无脸时输出原场景，沿用 Card 的缩放和 JPEG 编码设置。模型、输入解码或编码错误会跳过故障帧并上报状态。稳定检测沿用旧版每边 35% 扩框。
- 每张脸在各自的框内选点，所有点同批计算 LK。保留旧版窗口 15×15、2 层金字塔、至少 7 个有效点、单轴位移小于 40px 的设置。新增至少一半点有效、LK 误差不超过 20、相对位移中位数残差不超过 3px 的过滤。
- 检测漏掉已有目标时，有效 LK 最长从上次检测延续 550 ms，使用 50% 扩框。LK 丢失时最多保留一次最近可靠框，且它的年龄不超过 120 ms。恢复检测即使用正常扩框。关闭补偿后，固定检测的间隔帧仍用 LK，此时检测帧漏掉的脸即时移除。
- 自适应模式在跟踪退化或移动较快时提前检测。静止且跟踪可靠时最多隔 5 帧，无跟踪目标时最多隔 2 帧。快速移动阈值为 `max(4px, 当前框短边 × 12%)`。启动、分辨率变化和长时间断流后的首帧都运行检测。新脸最快在下一次 YuNet 检测被发现。
- 延迟模式使用可靠的前后检测框与连续 LK 轨迹。末端检测与预测框须有至少 0.45 IoU，存在 0.20 IoU 的竞合候选或轨迹中途重叠时跳过关联纠错。将可靠终点的偏差按源时间戳分配到尚未到期的历史帧，直接以纠正框渲染。首见人脸、已发布帧和缺少连续轨迹的帧没有回溯纠正。
- 输入单槽保留最新帧。回调拒收时间戳乱序或源年龄超过 500 ms 的帧。发布按源时间戳排序，丢弃超过配置延时加 500 ms 的帧。缓存满时丢弃当前准备入队的图像，已排定的帧继续到点发布，避免低容量下持续无输出。0 ms 模式直接渲染发布，无待发布历史缓存。停止立即禁止输出，worker 退出时释放待发布帧。若模型下载或推理仍阻塞，状态会保持 `stopping` 直到 worker 完成退出。

`info.metrics` 区分目标处理 FPS、近 2 秒实际处理 FPS、输出 FPS 和最近 90 个已发布帧的源时间戳至 ROS 发布延迟 P50/P95。看摄像头到浏览器的延迟时，应另计摄像头取样、接收端 DDS 和渲染开销。状态还报告输入覆盖、过期、超容量、处理错误、缓存当前/峰值和已发布纠错帧数。降低目标 FPS 时，输入覆盖是预期的主动抽帧。

弱光、很小或被遮挡的人脸、快速离开/进入、短时间交叉和超额人脸可能暂时露出。不同设备、分辨率及共享推理负载会改变处理和发布速率。测试与复现见 [性能与复现](PERFORMANCE.md) 和 [提取前对照表](AUDIT.md)。
