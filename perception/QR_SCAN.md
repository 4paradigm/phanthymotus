# 通用 JPEG 二维码感知卡片（第一版）

`qr_scan` 位于 perception，工具类型 processor，multiInstance=true。
它与厂商 SDK 无关，也不打开相机：任何能在同一 ROS2 通信环境中提供
`sensor_msgs/CompressedImage` JPEG 的机器人都可连接。

## 输入输出

输入由画布传入 `input_topic`，支持 format 为 jpeg、jpg 或 image/jpeg。
输出为 `std_msgs/String` JSON，格式 data/json。
每个实例输出到 `/perception/qr_scan/instance_<实例标识的哈希>/result`。
`info` 在实例启动前即能推导输出；每个实例独立解码、去重、缓存和生命周期。

结果包括 status、codes、new_events、session_id、frame_sequence、frame_age_s、
frame_received_at 和 processed_at。codes 内包含 text、corners_px、image_width、image_height。
角点对应输入 JPEG 的像素坐标，而非厂商原始图像的坐标。

- waiting_for_frame：输入尚未到达。
- detected：解码到二维码。
- no_qr：图像正常但未解码出码。
- stale：图像过期，内容清空。
- decode_error：图像格式/内容错误。
- error：扫描或发布异常。

new_events 对持续可见的同内容去重。消失超过 rearm_after_s 后再次出现会重新触发。
read 返回缓存而非触发新扫描，同一缓存可以读到同一事件；消费者按 session_id + sequence 去重。
首次 config 更新停止该实例，重新 start 才恢复扫描；stop 不指定实例时停止全部实例。
输入更换会释放旧订阅和工作线程；输入回调只缓存最新帧，扫描在线程中限频执行。

本版不支持 RTSP/H264/H265 或 sensor_msgs/Image；这些应先转换成 JPEG 话题。
不执行二维码内容、不访问其中的网址、不新增 Activity 日志。
可以在同一卡片开启 auto_announce，生成独立的播报文字输出。

## 配置

perception/config.yaml 中默认启用 qr_scan；只有连接并启动实例才订阅相机。
scan_hz 默认 2；stale_after_s 默认 3；rearm_after_s 默认 3。
画布 configSchema 允许每实例配置这些参数。

使用 perception 镜像已有的 OpenCV，不覆盖 Jetson 的 CUDA OpenCV。
Dockerfile 增加真实 QR→JPEG→解码自检；若基础 OpenCV 没有 QR 解码后端，构建会失败并指出原因。

## 开发验证

在仓库根目录，配置好 Python 虚拟环境和 OpenCV 4.x：

```bash
python -m pytest perception/tests/test_qr_scan.py perception/tests/test_bundle_dispatch.py perception/tests/test_shared_utils.py -q
```

这组测试以真实 OpenCV JPEG 解码、模拟 ROS 端点验证内容和多实例接口。

真实 DDS 检查需要 ROS2 环境，隔离 domain 142，无需机器人：

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/perception:$PYTHONPATH" python3 perception/tests/qr_scan_dds_check.py
```

## U1 实机前准备

1. 将平台仓库功能分支推送到自己的 fork，创建平台 PR；确认平台构建方式，取得新 perception 镜像。
   此次不需要更新 Agent Core。完整 Jetson 镜像尚未在本次开发中构建。
2. 核对 U1 JetPack 版本。以前检查到感知镜像标签为 jp6.1，但应按当前实际环境确认。
   仓库本地构建入口为 `./deploy/build_perception.sh --jp-version 6.1`，它可能按 deploy/.env 发布镜像。
   构建前核对 registry 和基础镜像配置，不直接在机器人上盲目构建。
3. 更新前保存当前 perception 容器配置和镜像版本。新旧都叫 embodied-perception，不能并行占用相同端口。
   ASR/TTS 等共用这个容器，替换会暂时影响这些功能；切换应单独安排。
4. 核对服务片段中的 NVIDIA runtime、设备/模型目录挂载及 DDS profile。
   这份新平台代码与 U1 旧平台可能有版本差异，须确认本机通信配置后再部署。

## U1 画布验收

1. 左侧选“感知”服务下的 qr_scan，避免选到旧 U1 driver 下同名卡片。
2. 连接 `U1 camera_left → 感知 qr_scan`，输入应为 `/vision/camera/left`（以相机 info 为准）。
3. 启动实例，查看数据流，展示 U1-001、中文和网址码。
4. 确认 detected/content/corners；持续同一码不产生重复 new_events。
5. 移开超过三秒再展示，应产生新事件；停止相机超过三秒应 stale 且 codes 为空。
6. 停止重启扫码应改变 session；另一扫码实例可以接独立相机输入且结果互不覆盖。
7. 若使用 Decision Core，只接通用扫码这一路结果，避免与旧 U1 qr_scan 同时触发任务。
   需要时通过旧卡片 stop 停止旧扫码插件；不要停止整个 U1 驱动，以免失去相机输入。

成功读取仍不代表大脑会主动说话。任务规则、事件路由和 TTS 属于另一个功能。

## 同一卡片的自动播报

在 qr_scan 配置中开启 `auto_announce: true`，第二个输出口“自动播报文字”发布
`{"text": "扫码成功，内容是：U1-001", "session_id": ..., "sequence": ...}`。
连线为 `camera → qr_scan（自动播报文字端口）→ 感知层 tts → speaker`。
第一个输出口仍是完整识别 JSON，不要把第一个口接到 TTS。
无需 Decision Core、Skill 或额外 qr_announce 卡片。

默认关闭自动播报。announce_cooldown_s 默认 10 秒，冷却期间的新事件跳过，不排队；
多个同时识别的码先播报第一个。普通文本最多朗读 announce_max_chars（默认 80）；
URI 和常见域名链接仅说“识别到一个链接”，不朗读长链接、不访问网址。
请先启动 TTS 和 speaker 并确认工作，再展示二维码；播报文字端口是实时事件，不是持久队列。
停止 qr_scan 不会停止下游 TTS 已接收的语音，停止整个测试时也应停止 TTS。

U1 最小验证脚本 `perception/tests/u1_qr_announce_mvp.py --speak` 使用隔离的测试 JPEG
验证卡片输出，并通过临时测试适配器调用 U1 原生 TTS；不修改画布或服务部署。
这是测试适配器，正式通用卡片的语音输出仍按上面的 TTS 数据流连接。

## U1 / 天弈原生播报模式

现支持可配置的 announce_backend=native_mcp，通过本机 driver 的 tts.speak 播报。
U1 地址为 http://127.0.0.1:15740/mcp，天弈为 http://127.0.0.1:15707/mcp。
tts_mcp_url 是按实例的敏感配置，分享画布时应清空并重新填写。
此模式只需相机连接 qr_scan，不需要感知 TTS 和 PCM speaker。
详细配置、适配范围与验收步骤见 QR_U1_TIANYI.md。
