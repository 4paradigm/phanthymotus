# 功能与性能验证

测量日期 2026-11-22。对照为旧版 PR #382 头部 `c8b6b15` 的 `SmoothMosaic`，新工作树基于 `a642f8a`。提取前已保存旧版首跑数据，另在同一正式对照批次重测旧版。以下对照使用后者。

## 同条件回放

两版在同一台 x86_64 Intel Core i9-13900HX 上运行，使用 Python 3.11.18、OpenCV 4.11.0、NumPy 1.26.4 和同一已验证的 YuNet 权重。每方案单独进程，启动后预热，回放每个方案 10 秒。独立跑分进程把 OpenCV 设为 3 线程，Card 运行时代码沿用宿主线程设置。输入是 640×360 公共新闻简报视频从 30 秒开始的连续 300 个 30 FPS JPEG，所有对照均核对 JPEG 序列 SHA-256。该片段含多人和人脸超额场景。 [白宫简报原始来源](https://obamawhitehouse.archives.gov/photos-and-video/video/2014/06/25/press-briefing)

- 实际处理 FPS 与输出 FPS 分别为成功分析、成功发布帧数除以 10 秒的源时间窗，收齐延迟发布的尾帧后计数。启动和结束处可能多计一个采样边界帧。JSON 同时记录首末输出间的活跃 FPS。
- 延迟是回放源时间至发布回调的实测时间，含抽帧等待、分析、缓存和 JPEG 编码。ROS2 到接收端和浏览器渲染不在该回放测量中。ROS2/MCP 联调另行验证实际消息。
- YuNet 单次耗时含检测器调用。LK 单次耗时只含 OpenCV 光流调用，新版框质量判断和自适应判断分别计时。处理耗时为单帧解码、分析与渲染的计算时间之和。
- 缓存修正数指有已更新位置或扩框状态、并实际发布的帧。含亚像素调整，不能作为新增正确覆盖帧计数。

| 方案 | 目标 FPS / 延迟 ms | 实际处理 FPS | 实际输出 FPS | 源到发布 P50 / P95 ms | YuNet P50 ms / 次数 | LK P50 ms / 次数 | 缓存峰值 / 纠错后发布 | 输入覆盖 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧版 | 30 / 0 | 30.0 | 30.0 | 1.86 / 6.92 | 4.00 / 75 | 0.21 / 129 | 0 / 0 | 0 |
| A | 30 / 0 | 30.0 | 30.0 | 2.17 / 6.83 | 4.10 / 75 | 0.21 / 297 | 0 / 0 | 0 |
| B | 30 / 300 | 30.0 | 30.0 | 301.98 / 306.65 | 4.15 / 75 | 0.20 / 297 | 9 / 219 | 0 |
| C | 24 / 300 | 24.1 | 24.1 | 300.72 / 300.88 | 4.10 / 61 | 0.21 / 237 | 8 / 177 | 59 |
| D | 20 / 300 | 20.1 | 20.1 | 300.73 / 300.85 | 4.02 / 51 | 0.20 / 198 | 7 / 147 | 99 |
| E | 24 / 500 | 24.1 | 24.1 | 500.72 / 500.86 | 3.97 / 61 | 0.21 / 237 | 13 / 177 | 59 |
| F | 20 / 500 | 20.1 | 20.1 | 500.73 / 500.84 | 4.03 / 51 | 0.21 / 198 | 11 / 147 | 99 |
| G | 15 / 200 | 15.1 | 15.1 | 200.74 / 200.88 | 4.11 / 38 | 0.21 / 148 | 4 / 72 | 149 |
| A 自适应 | 30 / 0 | 30.0 | 30.0 | 6.05 / 7.13 | 4.08 / 164 | 0.21 / 299 | 0 / 0 | 0 |

所有组合合计没有过期、缓存超限或处理故障。降帧组的输入覆盖是 30 FPS 输入被主动抽样。缓存峰值最高为 13 帧、约 8.57 MiB。FPS 与延迟参数的 16 种组合均有校验测试。表中性能只覆盖列出的组合。

旧版每帧均处理全画面，其中 128/300 帧使用粗全帧马赛克。新版 A 平均仅 7.51% 的画面进入像素块渲染，压缩造成的像素差异不计入作用面积。旧版 172 帧具备局部框渲染，新版 A 有 299 帧具备局部框渲染。两项计数描述运行时行为，包含误检和超额场景，不能衡量真人脸覆盖。新版 A 的处理 P50/P95 为 2.01/6.68 ms，旧版为 1.69/6.79 ms。新版提高了 LK 调用次数，新增质量判断 P50/P95 为 0.22/0.37 ms，实测 30 FPS 保持不变，未见有意义的 P95 吞吐差异。初次旧版基线的 P95 延迟是 7.97 ms，也说明单轮的亚毫秒差异不宜外推。 [原始测量 JSON](evidence/)

自适应分支自身判断 P50/P95 约 0.001/0.001 ms。在主片段中，由于运动和弱跟踪，自适应调用了 164 次 YuNet，固定模式为 75 次，计算时间均值从 2.96 ms 上升到 4.28 ms。静止可靠目标用测试验证了最长间隔 5 帧，默认保留固定模式。额外重放 `close_speaker`（转头与手部短遮挡）、`greeting_cluster`（多人重叠）和 `profiles`（侧脸后切无脸幻灯片）各 234 帧。三个片段均达到 30 FPS，纠错后发布分别为 172、0、47 帧。多人重叠片段中有 216 次检测超过人脸上限。侧脸片段有 156 帧正常无局部覆盖。这些重放验证流程和负载，场景的实际漏遮挡还需逐帧标注。 [采访公开录像](https://commons.wikimedia.org/wiki/File:The_YouTube_Interview_with_President_Obama.webm) · [多人片段公开录像](https://commons.wikimedia.org/wiki/File:President_Obama_welcomes_President-Elect_Trump_to_the_White_House_(25-12322-F).webm)

## 识别质量与漏遮挡

用相同模型在 11 张对照图复测阈值。样本包含正脸、按固定尺度合成的小脸、衰减亮度的正脸、目视标注的两张侧脸和三个无脸输入。匹配采用至少 0.25 IoU 的一对一框匹配，基于探针脚本中的固定近似参考框。该小样本可同时查看已知目标漏检与未匹配检测框。

| 阈值 | 已匹配目标 / 10 | 漏检参考目标 | 未匹配检测框 |
| --- | ---: | ---: | ---: |
| 0.10 | 10 | 0 | 6 |
| 0.22 | 10 | 0 | 2 |
| 0.28 | 10 | 0 | 2 |
| 0.45 | 10 | 0 | 0 |
| 0.70 | 10 | 0 | 0 |
| 0.90 | 5 | 5 | 0 |

0.90 的漏检包括两个侧脸、最暗合成图与两个最小尺寸样本。0.28 保持旧版基线，现有样本不足以建立通用最佳阈值。合成亮度衰减不包括真实低照度传感器噪声和运动模糊。 [逐样本测量](evidence/threshold-stills.json)

可控漏检测试用独立设定的每帧 4px 位移和带偏差的 LK 结果验证回溯，三帧的框坐标误差由 1/2/3px 修正到 0，含检测漏帧。另验证两张脸独立的真实 LK 方向、歧义目标不插值、固定/自适应检测、跟踪错误后释放目标及恢复、过载期间恢复发布。公开回放视频缺少完备的独立逐帧人脸真值，因此无法严格计算真人脸覆盖率、漏遮挡帧数、最长连续漏遮挡时间和人脸外的无效马赛克面积。

## 验证边界

- 功能已实现。测试覆盖无脸、单脸、进出、快速移动、短时漏检、重检测、扩框、历史纠错、错配拒绝及缓存失效。实拍录像可重放侧脸、转头和遮挡场景。
- 从上游源码归档应用本工作树补丁后，57 项 Card 测试与 82 项共享组件回归全部通过。覆盖真实 YuNet OpenCV、共享 ORT CPU worker 及其重启恢复。隔离的 Fast DDS 与 HTTP MCP 共 3 项集成通过，覆盖 0 ms、300 ms、两个实例的独立启停和本地模型镜像冷下载及复用缓存。整仓全量回归需在项目完整环境或 CI 中执行。
- 性能验证限于表中的 x86 CPU 回放。现场验收仍需真实摄像头和官方 ARM64/Jetson 镜像，覆盖 Canvas 显示、长时间运行、设备端 FPS/P95，以及有标注素材的遮挡质量。
- 干净环境部署需加载项目已有的 Perception 依赖和可读的已验证模型。首次从外网拉取模型的现场可达性、官方 CI 和 ARM64 镜像构建待验。

## 复现

当前镜像的默认配置会将 Card 加载到现有 Perception，启动处理仍须实例确认出镜同意。

从独立工作树根目录运行。需要当前环境的 OpenCV、NumPy、pytest，以及另一个位于 `c8b6b15` 的旧版检出。测试模型用 [README](README.md) 的校验值准备。可选 ORT 用例需要 ONNX Runtime。人像用例默认读取 scikit-image 的示例图，也可将 `TEST_YUNET_FACE_IMAGE` 指向同一张 `astronaut.png`。参考原始数据区的路径由运行者指定。

```sh
CARD=perception/plugins/yunet_lk_mosaic
export PYTHONPATH=perception TEST_YUNET_MODEL=/path/to/face_detection_yunet_2023mar.onnx
python3 -m pytest perception/tests/test_yunet_lk_mosaic.py -q

# 在只有回环接口的独立网络命名空间内运行 Fast DDS。
unshare --user --map-root-user --net sh -eu -c '
  ip link set lo up
  unset FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
  export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
  export TEST_YUNET_LOCAL_DDS=1
  exec python3 -m pytest "$1/tests/test_ros_mcp.py" -q
' sh "$CARD"

VIDEO=/path/to/briefing-20140625.mp4
OLD=/path/to/pr382-checkout
PYTHONPATH="$OLD/perception" python3 "$CARD/benchmark.py" --variant old \
  --video "$VIDEO" --model "$TEST_YUNET_MODEL" --output baseline.json
python3 "$CARD/benchmark.py" --variant new \
  --video "$VIDEO" --model "$TEST_YUNET_MODEL" --fps 30 --delay 300 --output B.json
```

其他回放组合修改 `--fps` 和 `--delay`。增加 `--adaptive` 可启用自适应模式。阈值探针接受 `--portrait`、`--profiles`、`--slide`、`--model`、`--output`，录入文件 SHA-256 与原始测量对应。公开视频及模型权重由运行者准备。
