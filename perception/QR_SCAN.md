# 通用二维码感知卡片

qr_scan 是 perception 的 processor 卡片，multiInstance=true。
不依赖机器人品牌、型号、厂商 SDK、固定相机话题或专属 TTS 接口。

## 标准接口

- 输入：任意合法的绝对 input_topic，消息类型 sensor_msgs/CompressedImage，JPEG 数据。
- 输出1：识别结果，std_msgs/String JSON，data/json。
- 输出2：可选播报文字，std_msgs/String JSON，data/json，含 text、session_id、sequence。

每个实例独立维护图像缓存、二维码检测器、去重状态和扫描线程。
输出为 /perception/qr_scan/instance_<实例哈希>/result 和同级 announce，info 可提前推导。
只缓存最新帧；结果包含状态、码内容、JPEG像素角点、图像尺寸、帧接收时间、序号和新事件。

优先直接解码整帧。发现二维码但解码失败时，裁出该区域并保留边缘，
按0.75、1、2、1.5倍重试；单帧最多8次，重试图像最长边最多1600像素。
部分码已解出时保留成功结果并重试剩余候选；重试结果不会把邻近码内容归到当前候选。
输出角点始终使用原图坐标。没有找到二维码为no_qr；找到但均无法解码为
detected_unreadable，undecoded_codes包含未解码候选的原图角点及图像尺寸。
这种候选不产生新事件或播报文字。部分成功时仍为detected，同时保留undecoded_codes。
断流清空两类结果。复杂图案的重试增加处理时间，需在实机检查延迟及stale状态。

支持 start、stop、info、read、config。instance_id 优先定位实例，否则使用 input_topic。
stop 不指定实例时停止全部。config 更新停止该实例，重新 start 生效。

## 配置

- scan_hz：默认2，范围0.2–10。
- stale_after_s：默认3，范围0.5–30。
- rearm_after_s：默认3，范围0.5–300。
- auto_announce：默认false，生成播报文字。
- announce_cooldown_s：默认10，范围0–300。
- announce_max_chars：默认80，范围10–300。

同码连续可见时 codes 持续更新，new_events 只产生一次；消失超过 rearm_after_s 后再出现重新触发。
read 读取缓存，可能读到相同事件，消费者应按 session_id+sequence 去重。
断流超过 stale_after_s 时 codes/new_events 清空，状态 stale。

## 通用播报连接

```text
任意标准相机 → qr_scan 的自动播报文字输出 → 兼容的 TTS → 兼容的音频播放设备
```

开启 auto_announce=true。第一个输出是完整识别JSON，第二个输出才是TTS文字。
扫码卡片不会直接调用硬件或机器人原生TTS，也不生成音频。
先启动下游TTS/播放设备，再展示二维码；播报端口是实时事件，不是持久队列。
TTS与播放设备是否可用由其各自的接口适配负责，不应把机型差异放进扫码算法。

普通文字生成“扫码成功，内容是……”。URI、应用链接及常见域名只说识别到链接，完整内容仍保留在 codes。
同事件不重复播报，冷却期间的新事件跳过，不补播；多码默认先报告第一个。
长文本截断，不执行二维码指令、不访问网址、不触发Activity日志。
停止扫码不会中断下游TTS已经接受的语音，结束测试时须另行停止下游。

## 支持范围与部署

已输出标准JPEG的设备通常只配置输入话题。原始 Image、RTSP/H264/H265 或私有 SDK 需要先转换。
软件可复用不等于所有硬件已验证；需测试各机器的图像质量、通信和下游音频接口。
当前仓库的完整 perception 镜像面向 Jetson；其他计算平台要选用对应运行环境。
本次更新只涉及 perception，未修改 Agent Core 或各机器人 driver。
Dockerfile复用既有OpenCV，并进行JPEG二维码解码自检，避免基础镜像只有检测没有解码支持。

## 验证

```bash
python -m pytest perception/tests/test_qr_scan.py perception/tests/test_bundle_dispatch.py perception/tests/test_shared_utils.py -q
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
FASTRTPS_DEFAULT_PROFILES_FILE=/opt/phanthy-motus/dds-local.xml \
PYTHONPATH="$PWD/perception:$PYTHONPATH" python3 perception/tests/qr_scan_dds_check.py
```

测试包含真实OpenCV解码、两个无机型关联的相机输入、多实例隔离、去重、断流、生命周期、播报文字和链接过滤。
DDS脚本使用平台domain42及现有loopback配置，使用每次运行唯一的测试话题和实例标识，
避免与生产相机或其他检查串流。验证两路模拟图像及输出，不代表硬件实测。
需要构建并部署新的perception镜像后再在画布选择本卡片；原生播报专属配置已删除，需要按标准端口连接下游。
