# U1 / 天弈通用扫码与自动播报验收

扫码卡片属于 perception，processor，多实例。所有机器人复用同一份识别代码。
必要输入为 sensor_msgs/CompressedImage JPEG。RTSP/H264/H265、原始 Image、私有 SDK 要先转成该输入。
相机没有提供标准接口时需要 driver 适配；相机已输出 JPEG 时通常只配置话题。

## 需要更新什么

推送的是平台 phanthymotus 的 feat/perception-qr-scan 分支，不是 U1 driver 分支。
需要新 perception 镜像，不需要为扫码更新 Agent Core。
U1 专用 qr_scan 仍保留作为回退。本次没有替换实机镜像。

perception Dockerfile 已加 QR 解码自检；若 Jetson OpenCV 缺少二维码解码后端，构建会失败。
不要盲目用 pip OpenCV 覆盖 Jetson CUDA OpenCV。完整 Jetson 镜像尚未在本次验证中构建。
两台机器分别核对 JetPack 标签、NVIDIA runtime、模型目录挂载及 DDS 通信配置。
镜像里包含 ASR/TTS 等其他插件，更新前应保留原容器配置与旧镜像。

## 最简原生 TTS 模式（单个扫码卡片）

两台机器分别连接自己的相机，不是跨网络传输：

| 机器人 | 相机卡片 | 输入话题示例 | tts_mcp_url |
|---|---|---|---|
| U1 Pro | camera_left | /vision/camera/left | http://127.0.0.1:15740/mcp |
| 天弈 2.0 Pro | camera_head | /<namespace>/camera/head | http://127.0.0.1:15707/mcp |

输入话题以各自相机的 info 为准，namespace 不要硬编码照抄。

扫码实例配置：

```json
{
  "auto_announce": true,
  "announce_backend": "native_mcp",
  "tts_mcp_url": "http://127.0.0.1:15740/mcp",
  "announce_cooldown_s": 10,
  "announce_max_chars": 80
}
```

天弈把端口换为 15707，其他逻辑一样。
然后只连接 `camera_left / camera_head → 感知服务 qr_scan`。
不需要另接 Decision Core、TTS 或 speaker。
配置修改会停止实例，需要重新启动；驱动本身及原生 TTS 需正常在线。

原生模式只调用本机固定工具 tts、固定动作 speak，参数 force=false，不执行二维码内容。
仅允许本机 localhost/127.0.0.1/::1 的 HTTP MCP 地址，其他机器人本机按其驱动端口配置。
Native TTS 在后台线程调用，有5秒请求超时，队列最多一个，过期队列丢弃。
调用失败不影响继续扫码，read/info 的 speech.state=error；submitted 仅代表服务请求成功，仍需确认听到声音。
停止扫码不再提交新语音，但不强制中断已经被原生服务接受的播放。

### 其他机器人

如果它有同样的 `tts.speak(text)` MCP 接口，修改本机端口即可。
如果工具名、动作或参数不同，需要增加小型 TTS 适配器；扫码算法不用改。
如果没有原生 MCP TTS，可使用下方 topic 模式。

## 标准文字/音频模式

配置 `auto_announce=true, announce_backend=topic`。
连接 `相机 → 感知 qr_scan 第二输出（自动播报文字）→ 感知 TTS → 兼容的 PCM 扬声器`。
第一个输出是完整识别结果，第二个输出是含 text 的播报 JSON，不要选错。
U1 已有 PCM speaker；当前天弈 driver 不含同名 PCM speaker，因此天弈优先使用上面的原生模式。

## 验收

1. U1 若旧 driver qr_scan 自动播报也在运行，先关闭 auto_speak 或停止旧扫码插件。
   不要停止整个 U1 driver，以免失去 camera_left。
2. 在感知列表选择 qr_scan，确认配置及输入端口，先看相机画面正常再启动扫码。
3. 展示 U1-001 / TIANYI-001 / 中文二维码，数据流应 detected，原生模式应说“扫码成功，内容是……”一次。
4. 持续展示同码不重复；移开超过3秒且播报冷却结束后重新展示应重新播报。
5. http/https/wxp/weixin/wx/其他应用链接只说“识别到一个链接”，完整内容保留 codes 中。
6. 冷却期间切换码仍更新识别结果，但不会补播该事件；多码默认先播报第一个。
7. 停相机超过3秒应 stale；扫码停止重启改变 session；不同实例结果不覆盖。
8. 若识别正常无声音，检查 auto_announce/backend/url、原生 TTS 独立是否工作，以及 speech 字段。

## 已验证与未验证

78 项离线/模拟接口/真实 HTTP 合同与回归测试通过。
两路模拟 U1、天弈相机图像的真实 ROS2 DDS 输入、识别输出、每路一次播报文字和断流清空通过。
U1 原生 TTS 此前的一次性最小测试已确认有声音。
本版通用卡片尚未在天弈硬件上验证，也未替换两台机器的 perception 镜像。
